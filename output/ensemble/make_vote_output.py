import os
from collections import Counter

import output.ensemble.needleman_wunsch as needelman_wunsch
import output.ensemble.Smith_waterman as smith_waterman
import output.ensemble.simple_overlap as simple_overlap
import output.ensemble.edit_distance as edit_distance
from tqdm import tqdm

import pandas as pd
from scipy.stats import entropy
import numpy as np


def extract_parse(parsed_str):
    tokenized_sentence = []
    sent = parsed_str.split(' ')
    for word in sent:
        if ')' in word:
            idx = word.find(')')
            w = word[:idx]
            if '-LRB-' in w:
                w = w.replace('-LRB-', '(')
            if '-RRB-' in w:
                w = w.replace('-RRB-', ')')
            if '-LSB-' in w:
                w = w.replace('-LSB-', '[')
            if '-RSB-' in w:
                w = w.replace('-RSB-', ']')
            if len(w) == 0:
                tokenized_sentence.append(')')
            else:
                tokenized_sentence.append(w)
    return tokenized_sentence


def vocab2idx(lst):
    with open('./vocab.out', 'r', encoding='utf-8') as f:
        vocab_out = f.readlines()
    vocab = {label.strip(): idx for idx, label in enumerate(vocab_out)}
    new_lst = []
    for lab in lst:
        new_lst.append(vocab[lab])
    return new_lst


def write(path, words, right_tags, pred_tags, sentence=False):
    if sentence:
        with open(path + 'n2n_result.txt', 'a', encoding='utf-8') as f:
            for idx, ch in enumerate(words):
                if right_tags[idx] == pred_tags[idx]:
                    correction = '[O]'
                else:
                    correction = '[X]'
                f.write(correction + '\t' + ch + '\t' +
                        right_tags[idx] + '\t' +
                        pred_tags[idx] + '\n')
            f.write('--------------------------------------------------\n')
    else:
        right_tags = sum(right_tags, [])
        pred_tags = sum(pred_tags, [])
        words = ''.join(words).replace('[UNK]', '_')
        with open(path + 'n2n_result.txt', 'a', encoding='utf-8') as f:
            for idx, ch in enumerate(words):
                if right_tags[idx] == pred_tags[idx]:
                    correction = '[O]'
                else:
                    correction = '[X]'
                f.write(correction + '\t' + ch + '\t' +
                        right_tags[idx] + '\t' +
                        pred_tags[idx] + '\n')
            f.write('--------------------------------------------------\n')


def tag_converter(tag):
    if tag == 'O':
        return tag
    bio = tag[0:2]
    tag = tag[2:]
    new_tag = ''
    for t in tag.split('+'):
        new_tag += t[0] + '+'
    return bio + new_tag[:-1]


def get_word_sequence(path):
    with open(path + 'n2n_result.txt', 'r', encoding='utf-8') as f:
        result = f.readlines()
    all_sent = []
    sentence = []
    answer = []
    preds = []
    word = ''
    word_tag = []
    pred_tag = []
    for line_idx, line in enumerate(result):
        line = line.rstrip()
        if line == '--------------------------------------------------':
            sentence.append(word)
            answer.append(word_tag)
            preds.append(pred_tag)
            if len(pred_tag) < 1:
                sentence = sentence[:-1]
                answer = answer[:-1]
                preds = preds[:-1]
                x = 2
            all_sent.append([sentence, answer, preds])
            sentence = []
            answer = []
            preds = []
            word = ''
            word_tag = []
            pred_tag = []
            continue

        correction, ch, right, pred = line.split('\t')
        before_right = ''
        if len(word_tag) > 0:
            before_right = word_tag[-1]
        word += ch
        word_tag.append(right)
        pred_tag.append(pred)
        is_new = False
        if before_right != '' and before_right != 'O':
            if before_right[2:] != right[2:]:
                is_new = True
        if right == 'O':
            is_new = True

        if is_new:
            sentence.append(word)
            answer.append(word_tag)
            preds.append(pred_tag)
            word = ''
            word_tag = []
            pred_tag = []
    return all_sent


def get_sentence_sequence(path):
    with open(path + 'n2n_result.txt', 'r', encoding='utf-8') as f:
        result = f.readlines()
    all_sent = []
    sentence = ''
    answer = []
    preds = []
    for line_idx, line in enumerate(result):
        line = line.rstrip()
        if line == '--------------------------------------------------':
            # answer = [y for x in answer for y in x]
            # preds = [y for x in preds for y in x]
            all_sent.append([sentence, answer, preds])
            sentence = ''
            answer = []
            preds = []
            continue
        correction, ch, right, pred = line.split('\t')
        sentence += ch
        answer.append(right)
        preds.append(pred)
    return all_sent


def get_chunk_sequence(path, chunk_size=5, tokenizer=None):
    with open(path + 'n2n_result.txt', 'r', encoding='utf-8') as f:
        result = f.readlines()
    all_sent = []
    sentence = ''
    answer = []
    preds = []
    for line_idx, line in enumerate(result):
        line = line.rstrip()
        if line == '--------------------------------------------------':
            # answer = [y for x in answer for y in x]
            # preds = [y for x in preds for y in x]
            if tokenizer is not None:
                tokenized_sentence = tokenizer.tokenize(sentence)
                chunk_sentence = []
                chunk_answer = []
                chunk_pred = []
                c_index = 0
                for w in tokenized_sentence:
                    if w.startswith('##'):
                        w = w[2:]
                    elif w.startswith('Ġ') or w.startswith('▁'):
                        w = w[1:]
                    if c_index >= len(answer):
                        x = 3
                    if answer[c_index] == 'O':
                        w = ' ' + w
                    chunk_sentence.append(w)
                    chunk_answer.append(answer[c_index:c_index + len(w)])
                    chunk_pred.append(preds[c_index:c_index + len(w)])
                    c_index += 1 if '[UNK]' in w else len(w)

            else:
                chunk_sentence = [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]
                chunk_answer = [answer[i:i + chunk_size] for i in range(0, len(answer), chunk_size)]
                chunk_pred = [preds[i:i + chunk_size] for i in range(0, len(preds), chunk_size)]

            all_sent.append([chunk_sentence, chunk_answer, chunk_pred])
            sentence = ''
            answer = []
            preds = []
            continue
        correction, ch, right, pred = line.split('\t')
        sentence += ch
        answer.append(right)
        preds.append(pred)
    return all_sent


def get_dp_sequence(path, mode='eng'):
    with open(path + 'n2n_result.txt', 'r', encoding='utf-8') as f:
        result = f.readlines()
    all_sent = []
    sentence = ''
    answer = []
    preds = []
    for line_idx, line in enumerate(tqdm(result)):
        line = line.rstrip()
        if line == '--------------------------------------------------':
            if mode == 'kor':
                nlp = spacy.load("xx_sent_ud_sm")
                # try:
                #     _create_unverified_https_context = ssl._create_unverified_context
                # except AttributeError:
                #     pass
                # else:
                #     ssl._create_default_https_context = _create_unverified_https_context
                # benepar.download('benepar_ko2')
                nlp.add_pipe("benepar", config={"model": "benepar_ko2"})
            elif mode == 'eng':
                nlp = spacy.load('en_core_web_md')
                nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
            doc = nlp(sentence)
            sent = list(doc.sents)[0]
            sent_parse = sent._.parse_string
            tokenized_sentence = extract_parse(sent_parse)
            chunk_sentence = []
            chunk_answer = []
            chunk_pred = []
            c_index = 0
            for w in tokenized_sentence:
                if len(answer) <= c_index:
                    x = 3
                if answer[c_index] == 'O':
                    w = ' ' + w
                chunk_sentence.append(w)
                chunk_answer.append(answer[c_index:c_index + len(w)])
                chunk_pred.append(preds[c_index:c_index + len(w)])
                c_index += 1 if '[UNK]' in w else len(w)

            all_sent.append([chunk_sentence, chunk_answer, chunk_pred])
            sentence = ''
            answer = []
            preds = []
            continue
        correction, ch, right, pred = line.split('\t')
        sentence += ch
        answer.append(right)
        preds.append(pred)
    return all_sent


def cal_most_similar_word(pred_lst, mode='smith_waterman', penalty=False, window=0):
    max_score = -10000000
    pred_tag = []
    right_tag = []

    for idx, pred in enumerate(pred_lst):
        score = 0
        for idx2, pred2 in enumerate(pred_lst):
            if mode == 'smith_waterman':
                score += smith_waterman.smith_waterman(pred, pred2, penalty=penalty)
            if mode == 'smith_waterman_affine':
                score += smith_waterman.smith_waterman_affine(pred, pred2)
            if mode == 'needleman_wunsch':
                score += needelman_wunsch.needelman_wunsch(pred, pred2, penalty=penalty)
            if mode == 'simple_overlap':
                if window > 0:
                    score += simple_overlap.window_simple_overlap(pred, pred2, window=window)
                else:
                    score += simple_overlap.simple_overlap(pred, pred2, penalty)
        if max_score < (score / len(pred_lst)):
            max_score = score / len(pred_lst)
            pred_tag = pred
            right_tag = idx
    return right_tag, pred_tag


def get_wordpiece_word_sequence(path):
    with open(path + 'n2n_result.txt', 'r', encoding='utf-8') as f:
        result = f.readlines()
    all_sent = []
    sentence = []
    answer = []
    preds = []
    word = ''
    word_tag = []
    pred_tag = []
    before_character = ''
    for line_idx, line in enumerate(result):
        line = line.rstrip()
        if line == '--------------------------------------------------':
            sentence.append(word)
            answer.append(word_tag)
            preds.append(pred_tag)
            if len(pred_tag) < 1:
                sentence = sentence[:-1]
                answer = answer[:-1]
                preds = preds[:-1]
            all_sent.append([sentence[1:], answer[1:], preds[1:]])
            sentence = []
            answer = []
            preds = []
            word = ''
            word_tag = []
            pred_tag = []
            before_character = ''
            continue

        correction, ch, right, pred = line.split('\t')

        if ch.startswith('##'):
            ch = ch[2:]
        if ch.startswith('Ġ') or ch.startswith('▁'):
            ch = ch[1:]
        r_tag = right[2:]
        right_li = []
        for idx in range(len(ch)):
            if idx == 0:
                right_li.append(right)
            else:
                right_li.append('I-' + r_tag)
        p_tag = pred[2:]
        pred_li = []
        for idx in range(len(ch)):
            if idx == 0:
                pred_li.append(pred)
            else:
                pred_li.append('I-' + p_tag)

        is_new = False
        if before_character == '\'' and ch == 't' \
                or p_tag == 'S' \
                or pred[0] == 'I':
            word = word[:-1] + ch + ' '
            word_tag = word_tag[:-1] + right_li + ['O']
            pred_tag = pred_tag[:-1] + pred_li + ['O']
            before_character = ch
        else:
            sentence.append(word)
            answer.append(word_tag)
            preds.append(pred_tag)
            word = ''
            word_tag = []
            pred_tag = []
            before_character = ''
            word = ch + ' '
            word_tag = right_li + ['O']
            pred_tag = pred_li + ['O']
        # if before_character == '\'' and ch == 't' \
        #         or p_tag == 'S' \
        #         or pred[0] == 'I':
        #         is_new = True
        # if right == 'O':
        #     is_new = True

        # if is_new:
        #     sentence.append(word)
        #     answer.append(word_tag)
        #     preds.append(pred_tag)
        #     word = ''
        #     word_tag = []
        #     pred_tag = []
        #     before_character = ''
    return all_sent


def get_wordpiece_sentence_sequence(path):
    with open(path + 'n2n_result.txt', 'r', encoding='utf-8') as f:
        result = f.readlines()
    all_sent = []
    sentence = []
    answer = []
    preds = []
    before_character = ''
    for line_idx, line in enumerate(result):
        line = line.rstrip()
        if line == '--------------------------------------------------':
            # answer = [y for x in answer for y in x]
            # preds = [y for x in preds for y in x]
            all_sent.append([''.join(sentence)[:-1], answer[:-1], preds[:-1]])
            sentence = []
            answer = []
            preds = []
            before_character = ''
            continue
        correction, ch, right, pred = line.split('\t')
        if ch.startswith('##'):
            ch = ch[2:]
        if ch.startswith('Ġ') or ch.startswith('▁'):
            ch = ch[1:]
        r_tag = right[2:]
        right_li = []
        for idx in range(len(ch)):
            if idx == 0:
                right_li.append(right)
            else:
                right_li.append('I-' + r_tag)
        p_tag = pred[2:]
        pred_li = []
        for idx in range(len(ch)):
            if idx == 0:
                pred_li.append(pred)
            else:
                pred_li.append('I-' + p_tag)

        if before_character == '\'' and ch == 't' \
                or p_tag == 'S' \
                or pred[0] == 'I':
            word = [sentence[-1][:-1], ch + ' '] if len(sentence) > 0 else [ch + ' ']
            sentence = sentence[:-1] + word
            answer = answer[:-1] + right_li + ['O']
            preds = preds[:-1] + pred_li + ['O']
            before_character = ch
            continue
        sentence.extend([ch + ' '])
        answer.extend(right_li + ['O'])
        preds.extend(pred_li + ['O'])
        before_character = ch
    return all_sent


def cal_double_similar_word(pred_lst, mode='smith_waterman', penalty=False):
    max_score = -10000000
    pred_tag = []
    for idx, pred in enumerate(pred_lst):
        score = 0
        for idx2, pred2 in enumerate(pred_lst):
            bio = [c[0] for c in pred]
            bio2 = [c[0] for c in pred2]
            if mode == 'smith_waterman':
                score += smith_waterman.smith_waterman(pred, pred2) + \
                         smith_waterman.smith_waterman(bio, bio2)
            if mode == 'smith_waterman_affine':
                score += smith_waterman.smith_waterman_affine(pred, pred2) + \
                         smith_waterman.smith_waterman_affine(bio, bio2)
            if mode == 'needleman_wunsch':
                score += needelman_wunsch.needelman_wunsch(pred, pred2, penalty=penalty) + \
                         needelman_wunsch.needelman_wunsch(bio, bio2, penalty=penalty)
            if mode == 'simple_overlap':
                score += simple_overlap.simple_overlap(pred, pred2, penalty) + \
                         simple_overlap.simple_overlap(bio, bio2, penalty)
        if max_score < (score / len(pred_lst)):
            max_score = score / len(pred_lst)
            pred_tag = pred

    return pred_tag


def get_sent_entropy(pred_lst):
    entropy_score = 0
    sent_len = len(pred_lst[0])

    for i in range(sent_len):
        data = []
        for j in range(len(pred_lst)):
            data.append(pred_lst[j][i])
        if len(set(data)) == 1:
            continue
        pd_series = pd.Series(data)
        counts = pd_series.value_counts()
        entropy_score += entropy(counts)
    return entropy_score / sent_len


def get_word_output(path1, path2, path3, path4, path5, output_path, mode='smith_waterman', chunk_size=0,
                    tokenizer='',
                    DP_mode='',
                    double=False,
                    penalty=False,
                    window=0):
    if len(tokenizer) > 0:
        from transformers import AutoTokenizer
        tokeniz = AutoTokenizer.from_pretrained(tokenizer)
        result = get_chunk_sequence(path1, tokenizer=tokeniz)
        result2 = get_chunk_sequence(path2, tokenizer=tokeniz)
        result3 = get_chunk_sequence(path3, tokenizer=tokeniz)
        result4 = get_chunk_sequence(path4, tokenizer=tokeniz)
        result5 = get_chunk_sequence(path5, tokenizer=tokeniz)
    elif chunk_size > 0:
        result = get_chunk_sequence(path1, chunk_size=chunk_size)
        result2 = get_chunk_sequence(path2, chunk_size=chunk_size)
        result3 = get_chunk_sequence(path3, chunk_size=chunk_size)
        result4 = get_chunk_sequence(path4, chunk_size=chunk_size)
        result5 = get_chunk_sequence(path5, chunk_size=chunk_size)
    elif 'wordpiece' in path1:
        result = get_wordpiece_word_sequence(path1)
        result2 = get_wordpiece_word_sequence(path2)
        result3 = get_wordpiece_word_sequence(path3)
        result4 = get_wordpiece_word_sequence(path4)
        result5 = get_wordpiece_word_sequence(path5)
    elif len(DP_mode) > 0:
        result = get_dp_sequence(path1, DP_mode)
        result2 = get_dp_sequence(path2, DP_mode)
        result3 = get_dp_sequence(path3, DP_mode)
        result4 = get_dp_sequence(path4, DP_mode)
        result5 = get_dp_sequence(path5, DP_mode)
    else:
        result = get_word_sequence(path1)
        result2 = get_word_sequence(path2)
        result3 = get_word_sequence(path3)
        result4 = get_word_sequence(path4)
        result5 = get_word_sequence(path5)
    last_folder = output_path.split('/')[0]
    output_path = output_path + '/' + mode + '/'
    if DP_mode:
        output_path = output_path + '/DP/'
    elif tokenizer != '':
        output_path = output_path + '/' + str(tokenizer) + '/'
    elif chunk_size > 0:
        output_path = output_path + '/chunk_size_' + str(chunk_size) + '/'
    if double:
        output_path += '/double/'
    if penalty:
        output_path += '/penalty/'
    if window > 0:
        output_path += '/window/'
    os.makedirs(output_path, exist_ok=True)
    select_dict = [0, 0, 0, 0, 0]
    for line, line2, line3, line4, line5 in zip(result, result2, result3, result4, result5):
        word, right_tag, pred_tag1 = line
        w2, right_tag2, pred_tag2 = line2
        w3, right_tag3, pred_tag3 = line3
        w4, right_tag4, pred_tag4 = line4
        w5, right_tag5, pred_tag5 = line5
        r_tags = [right_tag, right_tag2, right_tag3, right_tag4, right_tag5]
        words = [word, w2, w3, w4, w5]
        new_pred_tags = []
        new_right_tags = []
        new_words = []
        for i, p in enumerate(zip(pred_tag1, pred_tag2, pred_tag3, pred_tag4, pred_tag5)):
            pred_lst = list(p)
            if double:
                new_pred_tags.append(cal_double_similar_word(pred_lst, mode=mode))
            elif penalty:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode, penalty=penalty)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            elif window > 0:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode, window=window)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            else:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            for m in range(len(pred_lst)):
                if pred == pred_lst[m]:
                    select_dict[m] += 1

        # if len(sum(new_right_tags, [])) != len(sum(new_pred_tags, [])):
        #     is_write = False
        #     for node in r_tags:
        #         if len(sum(new_pred_tags, [])) == len(sum(node, [])):
        #             write(output_path + '/', new_words, node, new_pred_tags)
        #             is_write = True
        #             continue
        #     if not is_write:
        #         write(output_path + '/', word, right_tag, pred_tag1)
        #     continue
        #
        # write(output_path + '/', new_words, new_right_tags, new_pred_tags)
    return select_dict


def get_word_output_for_model(pred_list):
    new_pred_tags = []
    # pred_tag1, pred_tag2, pred_tag3, pred_tag4, pred_tag5 = pred_list
    sent_len = len(pred_list[0])
    for i in range(sent_len):
        # enumerate(zip(pred_tag1, pred_tag2, pred_tag3, pred_tag4, pred_tag5)):
        pred_lst = []
        for node in pred_list:
            pred_lst.append(node[i])
        r_idx, pred = cal_most_similar_word(pred_lst)
        new_pred_tags.append(pred)
    return new_pred_tags


def get_entropy(path1, path2, path3, path4, path5):
    result = get_sentence_sequence(path1)
    result2 = get_sentence_sequence(path2)
    result3 = get_sentence_sequence(path3)
    result4 = get_sentence_sequence(path4)
    result5 = get_sentence_sequence(path5)

    entropy_list = []
    for line, line2, line3, line4, line5 in zip(result, result2, result3, result4, result5):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        w, right_tag3, pred_tag3 = line3
        w, right_tag4, pred_tag4 = line4
        w, right_tag5, pred_tag5 = line5
        pred_lst = [pred_tag1, pred_tag2, pred_tag3, pred_tag4, pred_tag5]
        sent_entrpy = get_sent_entropy(pred_lst)
        entropy_list.append(sent_entrpy)
    print(np.mean(entropy_list))
    return np.mean(entropy_list)


def get_entropy4(path1, path2, path3, path4):
    result = get_sentence_sequence(path1)
    result2 = get_sentence_sequence(path2)
    result3 = get_sentence_sequence(path3)
    result4 = get_sentence_sequence(path4)

    entropy_list = []
    for line, line2, line3, line4 in zip(result, result2, result3, result4):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        w, right_tag3, pred_tag3 = line3
        w, right_tag4, pred_tag4 = line4
        pred_lst = [pred_tag1, pred_tag2, pred_tag3, pred_tag4]
        sent_entrpy = get_sent_entropy(pred_lst)
        entropy_list.append(sent_entrpy)
    print(np.mean(entropy_list))
    return np.mean(entropy_list)


def get_entropy3(path1, path2, path3):
    result = get_sentence_sequence(path1)
    result2 = get_sentence_sequence(path2)
    result3 = get_sentence_sequence(path3)

    entropy_list = []
    for line, line2, line3 in zip(result, result2, result3):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        w, right_tag3, pred_tag3 = line3
        pred_lst = [pred_tag1, pred_tag2, pred_tag3]
        sent_entrpy = get_sent_entropy(pred_lst)
        entropy_list.append(sent_entrpy)
    print(np.mean(entropy_list))
    return np.mean(entropy_list)


def get_entropy2(path1, path2):
    result = get_sentence_sequence(path1)
    result2 = get_sentence_sequence(path2)

    entropy_list = []
    for line, line2 in zip(result, result2):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        pred_lst = [pred_tag1, pred_tag2]
        sent_entrpy = get_sent_entropy(pred_lst)
        entropy_list.append(sent_entrpy)
    print(np.mean(entropy_list))
    return np.mean(entropy_list)


def get_sentence_output(path1, path2, path3, path4, path5, output_path, mode='smith_waterman',
                        double=False,
                        penalty=False,
                        window=0):
    if 'wordpiece' in path1:
        result = get_wordpiece_sentence_sequence(path1)
        result2 = get_wordpiece_sentence_sequence(path2)
        result3 = get_wordpiece_sentence_sequence(path3)
        result4 = get_wordpiece_sentence_sequence(path4)
        result5 = get_wordpiece_sentence_sequence(path5)
    else:
        result = get_sentence_sequence(path1)
        result2 = get_sentence_sequence(path2)
        result3 = get_sentence_sequence(path3)
        result4 = get_sentence_sequence(path4)
        result5 = get_sentence_sequence(path5)

    output_path = output_path + '/' + mode + '/'
    if double:
        output_path += '/double/'
    if penalty:
        output_path += '/penalty/'
    if window > 0:
        output_path += '/window/'
    os.makedirs(output_path, exist_ok=True)

    for line, line2, line3, line4, line5 in zip(result, result2, result3, result4, result5):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        w, right_tag3, pred_tag3 = line3
        w, right_tag4, pred_tag4 = line4
        w, right_tag5, pred_tag5 = line5
        pred_lst = [pred_tag1, pred_tag2, pred_tag3, pred_tag4, pred_tag5]
        if double:
            new_pred_tags = cal_double_similar_word(pred_lst, mode=mode)
        elif penalty:
            new_pred_tags = cal_most_similar_word(pred_lst, mode=mode, penalty=penalty)
        elif window > 0:
            new_pred_tags = cal_most_similar_word(pred_lst, mode=mode, window=window)
        else:
            new_pred_tags = cal_most_similar_word(pred_lst, mode=mode)

        write(output_path + '/sent_', word, right_tag, new_pred_tags, sentence=True)


def get_sentence_2_output(path1, path2, output_path, mode='smith_waterman'):
    result = get_sentence_sequence(path1)
    result2 = get_sentence_sequence(path2)

    output_path = output_path + '/' + mode + '/'
    os.makedirs(output_path, exist_ok=True)
    for line, line2 in zip(result, result2):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        pred_lst = [pred_tag1, pred_tag2]
        idx, new_pred_tags = cal_most_similar_word(pred_lst)
        write(output_path + '/sent_2_', word, right_tag, new_pred_tags, sentence=True)


def get_sentence_3_output(path1, path2, path3, output_path, mode='smith_waterman'):
    result = get_sentence_sequence(path1)
    result2 = get_sentence_sequence(path2)
    result3 = get_sentence_sequence(path3)

    output_path = output_path + '/' + mode + '/'
    os.makedirs(output_path, exist_ok=True)
    for line, line2, line3 in zip(result, result2, result3):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        w, right_tag3, pred_tag3 = line3
        pred_lst = [pred_tag1, pred_tag2, pred_tag3]
        idx, new_pred_tags = cal_most_similar_word(pred_lst)
        write(output_path + '/sent_3_', word, right_tag, new_pred_tags, sentence=True)


def get_sentence_4_output(path1, path2, path3, path4, output_path, mode='smith_waterman'):
    result = get_sentence_sequence(path1)
    result2 = get_sentence_sequence(path2)
    result3 = get_sentence_sequence(path3)
    result4 = get_sentence_sequence(path4)

    output_path = output_path + '/' + mode + '/'
    os.makedirs(output_path, exist_ok=True)
    for line, line2, line3, line4 in zip(result, result2, result3, result4):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        w, right_tag3, pred_tag3 = line3
        w, right_tag4, pred_tag4 = line4
        pred_lst = [pred_tag1, pred_tag2, pred_tag3, pred_tag4]
        idx, new_pred_tags = cal_most_similar_word(pred_lst)
        write(output_path + '/sent_4_', word, right_tag, new_pred_tags, sentence=True)


def get_word_2_output(path1, path2, output_path, mode='smith_waterman', chunk_size=0,
                      tokenizer='',
                      DP_mode='',
                      double=False,
                      penalty=False,
                      window=0):
    if len(tokenizer) > 0:
        from transformers import AutoTokenizer
        tokeniz = AutoTokenizer.from_pretrained(tokenizer)
        result = get_chunk_sequence(path1, tokenizer=tokeniz)
        result2 = get_chunk_sequence(path2, tokenizer=tokeniz)
    elif chunk_size > 0:
        result = get_chunk_sequence(path1, chunk_size=chunk_size)
        result2 = get_chunk_sequence(path2, chunk_size=chunk_size)
    elif 'wordpiece' in path1:
        result = get_wordpiece_word_sequence(path1)
        result2 = get_wordpiece_word_sequence(path2)
    elif len(DP_mode) > 0:
        result = get_dp_sequence(path1, DP_mode)
        result2 = get_dp_sequence(path2, DP_mode)
    else:
        result = get_word_sequence(path1)
        result2 = get_word_sequence(path2)

    output_path = output_path + '/' + mode + '/'
    if DP_mode:
        output_path = output_path + '/DP/'
    elif tokenizer != '':
        output_path = output_path + '/' + str(tokenizer) + '/'
    elif chunk_size > 0:
        output_path = output_path + '/chunk_size_' + str(chunk_size) + '/'
    if double:
        output_path += '/double/'
    if penalty:
        output_path += '/penalty/'
    if window > 0:
        output_path += '/window/'
    os.makedirs(output_path, exist_ok=True)

    for line, line2 in zip(result, result2):
        word, right_tag, pred_tag1 = line
        w2, right_tag2, pred_tag2 = line2
        r_tags = [right_tag, right_tag2]
        words = [word, w2]
        new_pred_tags = []
        new_right_tags = []
        new_words = []
        for i, p in enumerate(zip(pred_tag1, pred_tag2)):
            pred_lst = list(p)
            if double:
                new_pred_tags.append(cal_double_similar_word(pred_lst, mode=mode))
            elif penalty:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode, penalty=penalty)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            elif window > 0:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode, window=window)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            else:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]

        if len(sum(new_right_tags, [])) != len(sum(new_pred_tags, [])):
            is_write = False
            for node in r_tags:
                if len(sum(new_pred_tags, [])) == len(sum(node, [])):
                    write(output_path + '/test_2_', new_words, node, new_pred_tags)
                    is_write = True
                    continue
            if not is_write:
                write(output_path + '/test_2_', word, right_tag, pred_tag1)
            continue

        write(output_path + '/test_2_', new_words, new_right_tags, new_pred_tags)


def get_word_3_output(path1, path2, path3, output_path, mode='smith_waterman', chunk_size=0,
                      tokenizer='',
                      DP_mode='',
                      double=False,
                      penalty=False,
                      window=0):
    if len(tokenizer) > 0:
        from transformers import AutoTokenizer
        tokeniz = AutoTokenizer.from_pretrained(tokenizer)
        result = get_chunk_sequence(path1, tokenizer=tokeniz)
        result2 = get_chunk_sequence(path2, tokenizer=tokeniz)
        result3 = get_chunk_sequence(path3, tokenizer=tokeniz)
    elif chunk_size > 0:
        result = get_chunk_sequence(path1, chunk_size=chunk_size)
        result2 = get_chunk_sequence(path2, chunk_size=chunk_size)
        result3 = get_chunk_sequence(path3, chunk_size=chunk_size)
    elif 'wordpiece' in path1:
        result = get_wordpiece_word_sequence(path1)
        result2 = get_wordpiece_word_sequence(path2)
        result3 = get_wordpiece_word_sequence(path3)
    elif len(DP_mode) > 0:
        result = get_dp_sequence(path1, DP_mode)
        result2 = get_dp_sequence(path2, DP_mode)
        result3 = get_dp_sequence(path3, DP_mode)
    else:
        result = get_word_sequence(path1)
        result2 = get_word_sequence(path2)
        result3 = get_word_sequence(path3)

    output_path = output_path + '/' + mode + '/'  # +'xgboost/'
    if DP_mode:
        output_path = output_path + '/DP/'
    elif tokenizer != '':
        output_path = output_path + '/' + str(tokenizer) + '/'
    elif chunk_size > 0:
        output_path = output_path + '/chunk_size_' + str(chunk_size) + '/'
    if double:
        output_path += '/double/'
    if penalty:
        output_path += '/penalty/'
    if window > 0:
        output_path += '/window/'
    os.makedirs(output_path, exist_ok=True)

    for line, line2, line3 in zip(result, result2, result3):
        word, right_tag, pred_tag1 = line
        w2, right_tag2, pred_tag2 = line2
        w3, right_tag3, pred_tag3 = line3
        r_tags = [right_tag, right_tag2, right_tag3]
        words = [word, w2, w3]
        new_pred_tags = []
        new_right_tags = []
        new_words = []
        for i, p in enumerate(zip(pred_tag1, pred_tag2, pred_tag3)):
            pred_lst = list(p)
            if double:
                new_pred_tags.append(cal_double_similar_word(pred_lst, mode=mode))
            elif penalty:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode, penalty=penalty)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            elif window > 0:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode, window=window)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            else:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]

        if len(sum(new_right_tags, [])) != len(sum(new_pred_tags, [])):
            r_tags = [right_tag, right_tag2, right_tag3]
            is_write = False
            for node in r_tags:
                if len(sum(new_pred_tags, [])) == len(sum(node, [])):
                    write(output_path + '/test_3_', new_words, node, new_pred_tags)
                    is_write = True
                    continue
            if not is_write:
                write(output_path + '/test_3_', word, right_tag, pred_tag1)
            continue

        write(output_path + '/test_3_', new_words, new_right_tags, new_pred_tags)


def get_word_4_output(path1, path2, path3, path4, output_path, mode='smith_waterman', chunk_size=0,
                      tokenizer='',
                      DP_mode='',
                      double=False,
                      penalty=False,
                      window=0):
    if len(tokenizer) > 0:
        from transformers import AutoTokenizer
        tokeniz = AutoTokenizer.from_pretrained(tokenizer)
        result = get_chunk_sequence(path1, tokenizer=tokeniz)
        result2 = get_chunk_sequence(path2, tokenizer=tokeniz)
        result3 = get_chunk_sequence(path3, tokenizer=tokeniz)
        result4 = get_chunk_sequence(path4, tokenizer=tokeniz)
    elif chunk_size > 0:
        result = get_chunk_sequence(path1, chunk_size=chunk_size)
        result2 = get_chunk_sequence(path2, chunk_size=chunk_size)
        result3 = get_chunk_sequence(path3, chunk_size=chunk_size)
        result4 = get_chunk_sequence(path4, chunk_size=chunk_size)
    elif 'wordpiece' in path1:
        result = get_wordpiece_word_sequence(path1)
        result2 = get_wordpiece_word_sequence(path2)
        result3 = get_wordpiece_word_sequence(path3)
        result4 = get_wordpiece_word_sequence(path4)
    elif len(DP_mode) > 0:
        result = get_dp_sequence(path1, DP_mode)
        result2 = get_dp_sequence(path2, DP_mode)
        result3 = get_dp_sequence(path3, DP_mode)
        result4 = get_dp_sequence(path4, DP_mode)
    else:
        result = get_word_sequence(path1)
        result2 = get_word_sequence(path2)
        result3 = get_word_sequence(path3)
        result4 = get_word_sequence(path4)

    output_path = output_path + '/' + mode + '/'
    if DP_mode:
        output_path = output_path + '/DP/'
    elif tokenizer != '':
        output_path = output_path + '/' + str(tokenizer) + '/'
    elif chunk_size > 0:
        output_path = output_path + '/chunk_size_' + str(chunk_size) + '/'
    if double:
        output_path += '/double/'
    if penalty:
        output_path += '/penalty/'
    if window > 0:
        output_path += '/window/'
    os.makedirs(output_path, exist_ok=True)

    for line, line2, line3, line4 in zip(result, result2, result3, result4):
        word, right_tag, pred_tag1 = line
        w2, right_tag2, pred_tag2 = line2
        w3, right_tag3, pred_tag3 = line3
        w4, right_tag4, pred_tag4 = line4
        r_tags = [right_tag, right_tag2, right_tag3, right_tag4]
        words = [word, w2, w3, w4]
        new_pred_tags = []
        new_right_tags = []
        new_words = []
        for i, p in enumerate(zip(pred_tag1, pred_tag2, pred_tag3, pred_tag4)):
            pred_lst = list(p)
            if double:
                new_pred_tags.append(cal_double_similar_word(pred_lst, mode=mode))
            elif penalty:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode, penalty=penalty)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            elif window > 0:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode, window=window)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]
            else:
                r_idx, pred = cal_most_similar_word(pred_lst, mode=mode)
                new_pred_tags.append(pred)
                new_right_tags.append(r_tags[r_idx][i])
                new_words += words[r_idx][i]

        if len(sum(new_right_tags, [])) != len(sum(new_pred_tags, [])):
            r_tags = [right_tag, right_tag2, right_tag3, right_tag4]
            is_write = False
            for node in r_tags:
                if len(sum(new_pred_tags, [])) == len(sum(node, [])):
                    write(output_path + '/test_4_', new_words, node, new_pred_tags)
                    is_write = True
                    continue
            if not is_write:
                write(output_path + '/test_4_', word, right_tag, pred_tag1)
            continue

        write(output_path + '/test_4_', new_words, new_right_tags, new_pred_tags)


def get_hard_voting_output(path1, path2, path3, path4, path5, output_path):
    if 'wordpiece' in path1:
        result = get_wordpiece_sentence_sequence(path1)
        result2 = get_wordpiece_sentence_sequence(path2)
        result3 = get_wordpiece_sentence_sequence(path3)
        result4 = get_wordpiece_sentence_sequence(path4)
        result5 = get_wordpiece_sentence_sequence(path5)
    else:
        result = get_sentence_sequence(path1)
        result2 = get_sentence_sequence(path2)
        result3 = get_sentence_sequence(path3)
        result4 = get_sentence_sequence(path4)
        result5 = get_sentence_sequence(path5)

    for line, line2, line3, line4, line5 in zip(result, result2, result3, result4, result5):
        word, right_tag, pred_tag1 = line
        w, right_tag2, pred_tag2 = line2
        w, right_tag3, pred_tag3 = line3
        w, right_tag4, pred_tag4 = line4
        w, right_tag5, pred_tag5 = line5
        new_pred_tags = []

        if len(right_tag) != len(pred_tag1):
            print(right_tag)
            print(pred_tag1)

        for pred1, pred2, pred3, pred4, pred5 in zip(pred_tag1, pred_tag2, pred_tag3, pred_tag4, pred_tag5):
            c = Counter([pred1, pred2, pred3, pred4, pred5])
            pred_tag = c.most_common(1)
            pred_tag = [key for key, _ in pred_tag]
            new_pred_tags.append(pred_tag[0])

        if len(right_tag) != len(new_pred_tags):
            continue
        os.makedirs(output_path + '/hard_voting/', exist_ok=True)
        write(output_path + '/hard_voting/', word, right_tag, new_pred_tags, sentence=True)


if __name__ == '__main__':
    output_path = './output/DP2/wordpiece/bart/'

    for random_seed in rand_seeds:
        print(f'random: {random_seed}/ drop: {drop_outs}')
        di = get_word_output(output_path + f'{random_seed}/{drop_outs[0]}/',
                             output_path + f'{random_seed}/{drop_outs[1]}/',
                             output_path + f'{random_seed}/{drop_outs[2]}/',
                             output_path + f'{random_seed}/{drop_outs[3]}/',
                             output_path + f'{random_seed}/{drop_outs[4]}/',
                             output_path + f'{random_seed}')

        di = [str(x) for x in di]
        print('\t'.join(di))

    for drop_out in drop_outs:
        print(f'random: {rand_seeds}/ drop: {drop_out}')
        di = get_word_output(output_path + f'{rand_seeds[0]}/{drop_out}/',
                             output_path + f'{rand_seeds[1]}/{drop_out}/',
                             output_path + f'{rand_seeds[2]}/{drop_out}/',
                             output_path + f'{rand_seeds[3]}/{drop_out}/',
                             output_path + f'{rand_seeds[4]}/{drop_out}/',
                             output_path + f'{drop_out}')

        di = [str(x) for x in di]
        print('\t'.join(di))

    for random_seed in rand_seeds:
        for drop_out in drop_outs:
            print(f'random: {random_seed}/ drop: {drop_out}')
            di = get_word_output(output_path + f'{random_seed}/{drop_out}/1_',
                                 output_path + f'{random_seed}/{drop_out}/2_',
                                 output_path + f'{random_seed}/{drop_out}/3_',
                                 output_path + f'{random_seed}/{drop_out}/4_',
                                 output_path + f'{random_seed}/{drop_out}/5_',
                                 output_path + f'{random_seed}/{drop_out}/same/')
            di = [str(x) for x in di]
            print('\t'.join(di))