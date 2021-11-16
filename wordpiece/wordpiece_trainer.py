'''
    POS-TAGGING

    Author : Sangkeun Jung (2021)
'''

# most of the case, you just change the component loading part
# all other parts are almost same
#

import torch
from torch import nn

from conditional_random_field import ConditionalRandomField
import codecs

from measure.save_cr_and_cm import save_cr_and_cm
from measure.get_performance import get_performance

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

import pandas as pd

import numpy as np
from transformers import BartModel, BartConfig, BartTokenizer

from output.ensemble.make_vote_output import get_word_output_for_model


def load_tsv(fn):
    sents = []
    with open(fn, 'r', encoding='utf-8') as f:
        sent = []
        for line in f:
            line = line.rstrip()
            if line == '':
                sents.append(sent)
                sent = []
            else:
                c, t = line.split('\t')
                sent.append((c, t))

    return sents


def load_label(fn):
    label_vocab = {}
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            tag, idx = line.split('\t')
            idx = int(idx)
            label_vocab[tag] = idx

    return label_vocab


class MODU_POS_DataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32):
        super().__init__()
        self.max_len = 200
        self.batch_size = batch_size

        ## read label
        self.label_vocab = load_label('./data/treebank/label.vocab')
        self.num_labels = len(self.label_vocab)

        ## read tsv files, truncate exceeded data
        train_fn = './data/treebank/train.conv.tsv'
        valid_fn = './data/treebank/dev.conv.tsv'
        test_fn = './data/treebank/test.conv.tsv'

        train_data = load_tsv(train_fn)
        valid_data = load_tsv(valid_fn)
        test_data = load_tsv(test_fn)

        # tokenizer
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.pad_id = self.label_vocab['[PAD]']
        from wordpiece.pos_dataset import POS_Wordpiece_Dataset
        self.train_dataset = POS_Wordpiece_Dataset(train_data, self.tokenizer, self.max_len, self.label_vocab)
        self.valid_dataset = POS_Wordpiece_Dataset(valid_data, self.tokenizer, self.max_len, self.label_vocab)
        self.test_dataset = POS_Wordpiece_Dataset(test_data, self.tokenizer, self.max_len, self.label_vocab)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)  # , shuffle=True)  # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


from pytorch_lightning.metrics import functional as FM


class TextReader(nn.Module):
    def __init__(self):
        super(TextReader, self).__init__()

        ## switchable
        # from kobart import get_pytorch_kobart_model
        # from kobert.pytorch_kobert import get_pytorch_kobert_model
        from transformers import BertModel, BartModel, ElectraModel
        # self.text_reader = BartModel.from_pretrained(get_pytorch_kobart_model())
        # self.text_reader = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.text_reader = BartModel.from_pretrained('facebook/bart-large')

        self.config = self.text_reader.config

    def forward(self, input_ids, token_type_ids, attention_mask):
        # bert
        # final_hidden_vecs = self.text_reader(input_ids=input_ids.long(),
        #                                      token_type_ids=token_type_ids.long(),
        #                                      attention_mask=attention_mask.float()
        #                                      ).last_hidden_state
        # # bart
        final_hidden_vecs = self.text_reader(input_ids=input_ids.long(),
                                             attention_mask=attention_mask.float()
                                             ).encoder_last_hidden_state

        # final_hidden_vecs : [B, max_len, hidden_dim]
        # cls_pooler        : [B, hidden_dim]
        return final_hidden_vecs  # , cls_pooler


class BERTEncoder_POS_Tagger(pl.LightningModule):
    # <-- note that nn.module --> pl.LightningModule
    def __init__(self,
                 label_vocab,
                 tokenizer,
                 learning_rate=1e-3,
                 random_seed=1234,
                 dropout=0.1,
                 repeat=5,
                 batch_size=32,
                 test_num=1):
        super().__init__()
        self.save_hyperparameters()  # <-- it store arguments to self.hparams.*
        self.random_seed = random_seed
        self.dropout = nn.Dropout(dropout)
        self.label_vocab = label_vocab
        self.ignore_label_idx = self.label_vocab['[PAD]']
        self.repeat = repeat
        self.batch_size = batch_size
        self.test_num = test_num
        # text reader
        self.text_reader = TextReader()

        ## get -- character level tokenizer -- ##
        # self.bart_tokenizer = get_kobart_tokenizer()
        # self.electra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.tokenizer = self.bart_tokenizer

        # to class
        self.to_class = nn.Linear(self.text_reader.config.hidden_size, len(label_vocab))

        # loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label_idx)
        self.crf = ConditionalRandomField(len(label_vocab))

        self.output_tensor = []

    def get_alignmented_sequence(self, pred_labels, answer):
        # pred_batch1, pred_batch2, pred_batch3, pred_batch4, pred_batch5 = pred_labels
        size = pred_labels[0].size(dim=0)
        new_pred_lst = []
        for i in range(size):  # enumerate(zip(pred_batch1, pred_batch2, pred_batch3, pred_batch4, pred_batch5)):
            pred_lst = []
            for j in range(self.repeat):
                pred_lst.append(pred_labels[j][i])
            sentence_answer = self.wordpiece_to_word_list(answer[i].tolist())
            sentence_pred_lst = [self.tokenized_list(p, answer[i].tolist(), sentence_answer, ref_mode=False) for p in list(pred_lst)]
            new_pred_tags = get_word_output_for_model(sentence_pred_lst)
            new_pred_lst.append(new_pred_tags)
        return new_pred_lst

    def wordpiece_to_word_list(self, char_list):
        sentence = ''
        for i, c in enumerate(char_list):
            word = self.tokenizer.convert_ids_to_tokens(c)
            # if i == 0 and word == '<s>':
            #     continue
            if i != 0 and word in ['<s>', '</s>']:
                break
            if i != 0 and word in ['<pad>', '<unk>']:
                word = word.replace('<pad>', '_').replace('<unk>', '_')
            sentence += word.replace('<s>', '_').replace('</s>', '_')
        return sentence.replace('Ġ', ' Ġ').split(' ')

    def tokenized_list(self, pred, wordpiece_answer, answer, ref_mode=False):
        if not isinstance(pred, list):
            pred = pred.tolist()
        result = []
        sentence = []
        index = 0
        if not ref_mode:
            for node in wordpiece_answer:
                if not ref_mode and node == 0:
                    sentence.extend([pred[index]])
                else:
                    if not ref_mode:
                        word = self.tokenizer.convert_ids_to_tokens(node)
                    else:
                        word = node
                    if word[0] == 'Ġ':
                        sentence.extend([70] + [pred[index]] * (len(word) - 1))
                    else:
                        sentence.extend([pred[index]] * len(word))
                index += 1
        else:
            for node in answer:
                if not ref_mode and node == 0:
                    sentence.extend([pred[index]])
                else:
                    if not ref_mode:
                        word = self.tokenizer.convert_ids_to_tokens(node)
                    else:
                        word = node
                    if word[0] == 'Ġ':
                        sentence.extend([70] + [pred[index]] * (len(word)))
                    else:
                        sentence.extend([pred[index]] * len(word))
                index += 1
            sentence = [pred[0], 70] + sentence[1:]
            word = []
            for nod in sentence:
                if nod == 70:
                    result.append(word)
                    word = []
                    continue
                word.append(nod)
            result.append(word)
            return result

        index = 0
        for node in answer:
            if node in ['<s>', '<pad>', '</s>']:
                length = 1
            else:
                length = len(node)
            result.append(sentence[index:index + length])
            index += length
        return result

    def get_huggingface_char_tokenizer(self, bart_tokenizer, isbart):
        all_special_ids = bart_tokenizer.all_special_ids

        if isbart:
            cls_token = bart_tokenizer._bos_token
            cls_token_id = bart_tokenizer.bos_token_id
            sep_token = bart_tokenizer._eos_token
            sep_token_id = bart_tokenizer.eos_token_id
        else:
            cls_token = bart_tokenizer._cls_token
            cls_token_id = bart_tokenizer.cls_token_id
            sep_token = bart_tokenizer._sep_token
            sep_token_id = bart_tokenizer.sep_token_id

        pad_token = bart_tokenizer.pad_token
        pad_token_id = bart_tokenizer.pad_token_id

        unk_token = bart_tokenizer._unk_token
        unk_token_id = bart_tokenizer.unk_token_id

        ## charcter 만 기존 사전에서 얻어온다.
        char_token2idx = {idx: x for idx, x in enumerate(bart_tokenizer.vocab) if len(x) == 1}
        ## append special symbol
        char_token2idx[cls_token_id] = cls_token
        char_token2idx[pad_token_id] = pad_token
        char_token2idx[sep_token_id] = sep_token
        char_token2idx[unk_token_id] = unk_token
        return char_token2idx

    def cal_loss(self, step_logits, step_labels, ignore_idx):
        B, S, C = step_logits.shape
        predicted = step_logits.view(-1, C)
        reference = step_labels.view(-1)
        loss = self.criterion(predicted, reference.long())
        return loss

    def get_crf_loss(self, logits, path, weights):
        # logits : [batch_size, num_steps, dim]
        # path   : [batch_size, num_steps], a sequence of given tags
        # weights : [batch_size, num_steps], a sequence of 1 or 0 for marking non-padding and padding symbols

        log_likelihood = self.crf(logits, path, weights)
        negative_log_likelihood = -1.0 * log_likelihood
        loss = negative_log_likelihood
        return loss

    def forward(self, input_ids, token_type_ids, attention_mask):
        # text reader() -> to_class
        final_hidden_vecs = self.text_reader(input_ids=input_ids.long(),
                                             token_type_ids=token_type_ids.long(),
                                             attention_mask=attention_mask.float()
                                             )
        final_hidden_vecs = self.dropout(final_hidden_vecs)
        step_label_logits = self.to_class(final_hidden_vecs)  # [B, max_len, num_labels]
        return step_label_logits

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        pred_step_logits = self(input_ids, token_type_ids, attention_mask)

        ## loss calculation
        loss = self.cal_loss(pred_step_logits, labels, ignore_idx=self.ignore_label_idx)
        # loss = self.get_crf_loss(pred_step_logits, labels, attention_mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # NOTE : "validation_step" is "RESERVED"
        input_ids, token_type_ids, attention_mask, labels = batch
        pred_step_logits = self(input_ids, token_type_ids, attention_mask)

        ## loss calculation
        loss = self.cal_loss(pred_step_logits, labels, ignore_idx=self.ignore_label_idx)
        # loss = self.get_crf_loss(pred_step_logits, labels, attention_mask)
        # metrics = {'val_loss': loss}
        metrics = {'val_loss': loss,
                   'input_ids': input_ids.detach().cpu(),
                   'ne_output': labels.detach().cpu(),
                   'best_seq': pred_step_logits.argmax(-1).detach().cpu()}

        self.log('val_loss', loss, prog_bar=True, logger=True)
        # self.log_dict(matrics)
        return metrics

    def validation_epoch_end(self, val_step_outputs):
        val_loss = []
        input_ids = []
        ne_output = []
        best_seq = []
        for output in val_step_outputs:
            for node in output['input_ids'].tolist():
                input_ids.append(node)
            for node in output['ne_output'].tolist():
                ne_output.append(node)
            for node in output['best_seq'].tolist():
                best_seq.append(node)
        total_best_seq = best_seq
        total_ne_output = ne_output
        total_input_ids = input_ids

        self.performance_eval(total_input_ids, total_ne_output, total_best_seq)

    def test_step(self, batch, batch_idx):
        # NOTE : "validation_step" is "RESERVED"
        input_ids, token_type_ids, attention_mask, labels = batch
        pred_step_logits = self(input_ids, token_type_ids, attention_mask)

        ## loss calculation
        loss = self.cal_loss(pred_step_logits, labels, ignore_idx=self.ignore_label_idx)

        # pred_labels = self.predict(pred_step_logits, labels)
        # metrics = {'test_loss': loss}
        metrics = {'test_loss': loss,
                   'input_ids': input_ids.detach().cpu(),
                   'ne_output': labels.detach().cpu(),
                   'best_seq': pred_step_logits.argmax(-1).detach().cpu()}
                   # 'best_seq': pred_labels}

        self.log('test_loss', loss)
        # self.log_dict(matrics)
        return metrics

    def test_epoch_end(self, outputs):
        val_loss = []
        input_ids = []
        ne_output = []
        best_seq = []
        for output in outputs:
            for node in output['input_ids'].tolist():
                input_ids.append(node)
            for node in output['ne_output'].tolist():
                ne_output.append(node)
            for node in output['best_seq']:
                best_seq.append(node)
        total_best_seq = best_seq
        total_ne_output = ne_output
        total_input_ids = input_ids

        self.performance_eval(total_input_ids, total_ne_output, total_best_seq)
        # self.log('test_loss', val_loss)

    # def test_epoch_end(self, outputs):
    #     val_loss = []
    #     input_ids = []
    #     ne_output = []
    #     best_seq = []
    #     for output in outputs:
    #         for node in output['input_ids'].tolist():
    #             input_ids.append(node)
    #         for node in output['ne_output'].tolist():
    #             ne_output.append(node)
    #         for node in output['best_seq']:
    #             best_seq.append(node)
    #     total_best_seq = best_seq
    #     total_ne_output = ne_output
    #     total_input_ids = input_ids
    #     self.output_tensor = total_best_seq

    def get_output_tensor(self):
        return self.output_tensor

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def predict(self, logits, mask):
        # input : logits = [batch_size, num_steps, num_target_class]
        #         mask   = weights [batch_size, num_steps]
        #
        # do viterbi tags with transition matrix
        #

        predicted_tags = self.crf.viterbi_tags(logits, mask)
        return predicted_tags

    def performance_eval(self, input_ids, labels, total_best_seq):

        dict_id2label = {idx: label for idx, label in enumerate(self.label_vocab)}

        sent_list = []
        list_of_y_real = []
        list_of_y_pred = []
        for bs_idx, bs in enumerate(input_ids):
            # 문장 별 분리
            text = input_ids[bs_idx]  # .tolist()
            reference = labels[bs_idx]  # .tolist()
            prediction = total_best_seq[bs_idx]  # .tolist()
            a_sent = []
            for char_idx, char in enumerate(text):
                # 문장 속 char별 분리
                text_char = self.tokenizer.convert_ids_to_tokens(text[char_idx])
                if text_char in ('[CLS]', '<s>'):
                    continue
                elif text_char in ('[UNK]', '<unk>'):
                    text_char = ' '
                elif text_char in ('[SEP]', '</s>'):
                    break

                refer_char = dict_id2label[int(reference[char_idx])]
                if char_idx >= len(prediction):
                    pred_char = '<pad>'
                elif int(prediction[char_idx]) >= len(dict_id2label):
                    pred_char = '<pad>'
                else:
                    pred_char = dict_id2label[int(prediction[char_idx])]
                _diff = 'O' if refer_char == pred_char else 'X'

                a_sent.append((_diff, text_char, refer_char, pred_char))
                list_of_y_real.append(refer_char)
                list_of_y_pred.append(pred_char)

            sent_list.append(a_sent)
        perf = get_performance(sent_list)
        import os
        os.makedirs(f'./output/treebank3/wordpiece/bart/{self.random_seed}/{self.dropout}/',
                    exist_ok=True)
        n2n_result_fn = f'./output/treebank3/wordpiece/bart/{self.random_seed}/{self.dropout}/n2n_result.txt'
        with codecs.open(n2n_result_fn, 'w', encoding='utf-8') as f:
            for sent_idx, a_sent in enumerate(sent_list):
                for _diff, text_char, refer_char, pred_char in a_sent:
                    print("[{}]\t{}\t{}\t{}\t".format(_diff, text_char, refer_char, pred_char), file=f)
                print("{}".format("-" * 50), file=f)

            print("\n{}".format('=' * 50))
            print("\n Readable N2N result format is dumped at {}".format(n2n_result_fn))
            print("\n{}\n".format('=' * 50))

        save_cr_and_cm(dict_id2label, list_of_y_real, list_of_y_pred)
        return perf

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BERTEncoder_POS_Tagger")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser


from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping

def set_random_seed(random_seed):
    import random
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--dropout', default=1, type=int)
    parser.add_argument('--gpu_name', default=1, type=int)
    parser.add_argument('--repeat_num', default=1, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BERTEncoder_POS_Tagger.add_model_specific_args(parser)
    parser = MODU_POS_DataModule.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # ------------
    # data
    # ------------
    dm = MODU_POS_DataModule.from_argparse_args(args)
    x = iter(dm.train_dataloader()).next()

    # ------------
    # model
    # ------------
    model = BERTEncoder_POS_Tagger(dm.label_vocab, dm.tokenizer, args.learning_rate, args.seed,
                                   dropout=args.dropout / 100, repeat=args.repeat_num, batch_size=args.batch_size)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='./output/treebank3/wordpiece/',
        filename='BART' + f'-{args.seed}-{args.dropout / 100}'
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor='val_loss'), checkpoint_callback],
        gpus=str(args.gpu_name)  # if you have gpu -- set number, otherwise zero
    )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    model = BERTEncoder_POS_Tagger.load_from_checkpoint(
        f'./output/treebank3/wordpiece/BART-{args.seed}-{args.dropout / 100}.ckpt')
    # for i in range(9):
    model.eval()
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())
    print(result)


if __name__ == '__main__':
    cli_main()
