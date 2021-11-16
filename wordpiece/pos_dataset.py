"""
    DL-based NER Engine
        with pytorch lightning


    in this engine, I do not use CRF for simplicity.
    (TODO)


    Author : Sangkeun Jung (2021)
"""

from argparse import ArgumentParser

import os, sys, glob, platform
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

import numpy as np
from transformers import (AdamW)
import torch.nn

from torchmetrics import Accuracy, F1

import json
import pandas as pd
import codecs

from transformers import logging

import json
from tqdm import tqdm
import random
from torch.utils.data import Dataset, TensorDataset, DataLoader

logging.set_verbosity_error()  ## disable warning message


def result_collapse(outputs, target):
    if len([x[target] for x in outputs][0].shape) == 0:
        target_value = torch.stack([x[target] for x in outputs])
    else:
        target_value = torch.cat([x[target] for x in outputs])
    return target_value


def convert_to_wordpiece_form(data, tokenizer):
    sentence = "".join([x[0] for x in data])
    sentence_labels = [x[1] for x in data]

    words = tokenizer.tokenize(sentence)
    labels = []

    space_dict = []
    index = -1

    while True:
        index = sentence.find(' ', index + 1)
        if index == -1:
            break
        space_dict.append(index)

    if len(space_dict) == 0:
        space_dict.append(len(sentence))
    end_pt = space_dict[0]
    space_idx = 0
    t_idx = 0
    flag = True

    for w in words:
        if w in tokenizer.all_special_tokens:
            labels.append('[PAD]')
            continue
        if w.startswith('##'):
            w = w[2:]
        if w.startswith('Ġ') or w.startswith('▁'):
            w = w[1:]
        if t_idx >= len(sentence_labels):
            return [], sentence
        tag = sentence_labels[t_idx]
        labels.append(tag)
        t_idx += len(w)
        if t_idx == end_pt and space_idx < len(space_dict):
            space_idx += 1
            t_idx += 1
            end_pt = space_dict[space_idx] if len(space_dict) > space_idx else end_pt
    for i, tag in enumerate(labels):
        if i == 0:
            continue
        if labels[i - 1] == 'O' and tag[0] == 'I':
            labels[i] = 'B-' + tag[2:]

    if not (flag == True and len(words) == len(labels)):
        print("WARNING -- ")

    converted_data = [(token, label) for token, label in zip(words, labels)]
    return converted_data, sentence


class FCC_NER_Character_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len, label_vocab):
        self.pad_id = label_vocab['[PAD]']

        ## extract text
        self.inputs = []
        self.outputs = []
        ## TODO -- 다시 살펴야 함
        for sent in data:
            chars = [x[0] for x in sent]
            labels = [x[1] for x in sent]

            if len(chars) > max_len - 2:
                chars = chars[0: max_len - 2]
                labels = labels[0: max_len - 2]

            _input = tokenizer("".join(chars), padding='max_length', max_length=max_len, return_tensors='pt')

            _label_ids = [label_vocab[x] for x in labels]
            label_ids = [self.pad_id] + _label_ids
            pad_ids = [self.pad_id] * (max_len - len(label_ids))
            padded_label_ids = label_ids + pad_ids

            self.inputs.append(_input)
            self.outputs.append(padded_label_ids)

    def __getitem__(self, idx):
        result = {}
        for key, val in self.inputs[idx].items():
            result[key] = val.clone().detach().squeeze()
        result['labels'] = torch.tensor(self.outputs[idx])
        return result

    def __len__(self):
        return len(self.inputs)


class POS_Wordpiece_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len, label_vocab):
        self.pad_id = label_vocab['[PAD]']

        ## extract text
        self.inputs = []
        self.outputs = []
        self.data = []
        self.special_tokens = tokenizer.all_special_tokens

        for a_sent_char_and_label in data:
            # sent = list of character level data (character, label)
            # need conversion
            key_list = [c for c, t in a_sent_char_and_label]
            converted_sent, sentence = convert_to_wordpiece_form(a_sent_char_and_label, tokenizer)
            if len(converted_sent) == 0:
                continue
            tokens = [x[0] for x in converted_sent]
            labels = [x[1] for x in converted_sent]

            _input = tokenizer([sentence], padding='max_length', max_length=max_len, return_tensors='pt')

            _label_ids = [label_vocab[x] for x in labels]
            label_ids = [self.pad_id] + _label_ids
            if len(label_ids)>=max_len:
                padded_label_ids = label_ids[:max_len-1]+[self.pad_id]
            else:
                pad_ids = [self.pad_id] * (max_len - len(label_ids))
                padded_label_ids = label_ids + pad_ids

            self.inputs.append(_input)
            self.outputs.append(padded_label_ids)
            self.data.append(((_input['input_ids'].reshape(-1), np.zeros(_input['input_ids'].reshape(-1).shape), #_input['token_type_ids'].reshape(-1),
                               _input['attention_mask'].reshape(-1)), padded_label_ids))

    def __len__(self):
        return len(self.data)  # <-- this is important!!

    def __getitem__(self, idx):  # <-- !!!! important function.
        input, label = self.data[idx]
        chars, token_type_ids, attention_mask = input

        input_ids = np.array(chars)
        token_type_ids = np.array(token_type_ids)
        attention_mask = np.array(attention_mask)

        label = np.array(label)
        item = [input_ids, token_type_ids, attention_mask, label]
        return item
