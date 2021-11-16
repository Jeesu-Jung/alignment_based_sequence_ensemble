import json
import pandas as pd


def get_vocab_out(tags):
    tagset = list(set(tags))

    for tag in tagset:
        with open('/home/tmp/pycharm_project_61/data/treebank/DP/vocab.raw.out', 'a', encoding='utf-8') as f:
            f.write(tag + '\n')


def converter(mode):
    path = f'/home/tmp/pycharm_project_61/data/treebank/ptb_{mode}_3.3.0.sd.clean'
    output_path = '/home/tmp/pycharm_project_61/data/treebank/DP/' + mode + '.conv.tsv'

    data = pd.read_csv(path, header=None, sep='\t')

    all_tags = []
    sentences = []

    # if mode == 'test':
    #     with open('/home/tmp/pycharm_project_61/data/treebank/DP/vocab.raw.out', 'r', encoding='utf-8') as f:
    #         vocabs = f.readlines()
    #         vocab_out = []
    #         for tag in vocabs:
    #             vocab_out.append(tag.rstrip())

    sentence = ""
    tags = []
    for idx, row in data.iterrows():
        if row[1] == '':
            continue
        if idx > 0 and row[0] == 1:
            assert len(sentence) == len(tags), 'INVALID Sentence'
            sentences.append([sentence[:-1], tags[:-1]])
            sentence = ''
            tags = []
        context = row[1]
        entity = row[7].upper()
        if context == '``' or context == '\'\'':
            context = '\"'
        if not entity.isalpha():
            entity = 'S'

        tag = ['B-' + entity] + ['I-' + entity] * (len(context) - 1) + ['O']
        assert len(context) + 1 == len(tag), 'INVALID LINE'
        # if mode == 'test':
        #     if 'B-' + entity not in vocab_out:
        #         continue
        # if entity == 'S' and context not in ['\"', '\'']:
        #     sentence = sentence[:-1] + context + ' '
        #     tags = tags[:-1] + tag
        #     continue
        sentence += context + ' '
        tags += tag

    all_tags.extend(tag)

    with open(output_path, 'w', encoding='utf-8') as f:
        for context, tags in sentences:
            # for c, t in zip(list(context), tags):
            for c, t in zip(context, tags):
                if c == '\t':
                    c = '_'
                f.write(c + '\t' + t + '\n')
            f.write('\n')
    return all_tags


all_tags = converter('train')
all_tags += converter('dev')
# get_vocab_out(all_tags)
converter('test')
