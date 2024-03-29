#!/usr/bin/env python

"""Random baseline for the PAN19 hyperpartisan news detection task"""
# Version: 2018-09-24

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputDir=<directory>
#   Directory to which the predictions will be written. Will be created if it does not exist.

from __future__ import division

import os
import sys
from lxml import etree
import codecs
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import argparse
import logging
import datetime

import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert.tokenization import printable_text, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(filename = '{}_log.txt'.format(datetime.datetime.now()),
                                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                        datefmt = '%m/%d/%Y %H:%M:%S',
                                        level = logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))


clean_data = False
runOutputFileName = "prediction.txt"
max_seq_length = 512
bert_model = "bert-base-uncased"
batch_size = 32
model_path = "/home/howard-beale/model_article.pt"

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(example, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class HyperpartisanData(Dataset):
    """"""
    def __init__(self, examples, max_seq_length, tokenizer):
        self.examples = examples
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        feats = convert_examples_to_features(ex, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor(feats.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feats.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feats.segment_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, ex.guid

def to_out(row):
    if row.prediction == 1:
        row.prediction = "true"
    else:
        row.prediction = "false"

    return row

def main():
    """Main method of this module."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--inputDataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir")
    parser.add_argument("-o", "--outputDir", default=None, type=str, required=True,
                        help="Output dir for predictions")

    args = parser.parse_args()

    inputDataset = args.inputDataset
    outputDir = args.outputDir
    
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertForSequenceClassification.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    logger.info("Model state has been loaded.")

    for filename in glob.glob(inputDataset + "/*.xml"):

        tree = ET.parse(filename)
        root = tree.getroot()
        df = pd.DataFrame([el for el in root], columns=['xml'])
        tree = None
        root = None

        df['id'] = None
        df['text'] = None

        def get_stuff(row):

            row.id = row.xml.attrib['id']
            row.text = "".join([x for x in row.xml.itertext()])

            return row

        df = df.apply(get_stuff, axis=1)
        df.drop(columns=['xml'], inplace=True)

        if clean_data:
            df.text = df.text.str.replace(r"&#160;",r" ")
            df.text = df.text.str.replace(r"\n_{1,}\s?\n",r"\n")
            df.text = df.text.str.replace(r"\n\s?\*{2,}\s?\n",r"\n")
            df.text = df.text.str.replace(r"&amp;",r"&")
            df.text = df.text.str.replace(r"\n_{1,}\s?$","")
            df.text = df.text.str.replace(r"^_{1,}\s?\n","")
            df.text = df.text.str.replace(r"\nADVERTISEMENT\s?\n","\n")

        examples = []
        for (i, line) in df.iterrows():
            guid = line.id
            text_a = line.text
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        logger.info("Created examples from xml.")
        df = None
        df = pd.DataFrame(columns=["id","prediction"])

        test_dataloader = DataLoader(dataset=HyperpartisanData(examples, max_seq_length, tokenizer), batch_size=batch_size)

        for input_ids, input_mask, segment_ids, doc_ids in test_dataloader:

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)
                logits = logits.detach().cpu().numpy()
                labels = np.argmax(logits, axis=1)

            df = df.append([{"id":doc_ids[i], "prediction":labels[i]} for i in range(len(labels))], ignore_index=True)
            logger.info("Done 1 batch")

    df = df.apply(to_out, axis=1)
    df.to_csv(outputDir + "/predictions.txt", sep=" ", index=False, header=None)
    logger.info("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main()
