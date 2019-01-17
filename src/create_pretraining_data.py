# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import random
import pandas as pd
import argparse
from dask import dataframe as dd
from dask.multiprocessing import get
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_labels = masked_lm_labels

def create_training_instances2(input_file, output_file, tokenizer, max_seq_length,
                               dupe_factor, short_seq_prob, masked_lm_prob,
                               max_predictions_per_seq, rng, random_seed):
    """Create `TrainingInstance`s from raw text."""
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.

    f = open(input_file, "r", encoding="utf-8")
    copy_f = open(input_file, "r", encoding="utf-8")
    file_size = f.tell()

    vocab_words = list(tokenizer.vocab.keys())

    for i in range(dupe_factor):
        for _, line in enumerate(tqdm(f, desc="Creating pretraining data")):
            line = line.strip().replace("\\n", " ")
            line = tokenizer.tokenize(line)
            instances = create_instances_from_document(line, max_seq_length, short_seq_prob,
                                                       masked_lm_prob, max_predictions_per_seq,
                                                       vocab_words, rng, random_seed, file_size,
                                                       copy_f, tokenizer)

            for instance in instances:

                input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = list(instance.segment_ids)
                masked_lm_ids = instance.masked_lm_labels
                assert len(input_ids) <= max_seq_length

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    masked_lm_ids.append(-1)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                while len(masked_lm_ids) > max_seq_length:
                    print("Don't know why!")
                    masked_lm_ids.pop()

                assert len(masked_lm_ids) == max_seq_length

                for i,x in enumerate(masked_lm_ids):
                    if masked_lm_ids[i] != -1:
                        masked_lm_ids[i] = tokenizer.convert_tokens_to_ids([x])[0]

                next_sentence_label = 1 if instance.is_random_next else 0

                with open(output_file, "a", encoding="utf-8") as g:
                    g.write(str(input_ids) + "\t" + str(input_mask) + "\t" + str(segment_ids) + "\t" + str(masked_lm_ids) + "\t" + str(next_sentence_label) + "\n")

    f.close()
    copy_f.close()

def get_random_sentence(file_size, rng, copy_f, tokenizer, min_sent_length, max_length):

    while True:
        copy_f.seek(rng.randint(0,file_size))
        copy_f.readline() # We can be in the middle of the line
        rand_line = copy_f.readline().strip()
        rand_line = tokenizer.tokenize(rand_line)
        if len(rand_line) >= min_sent_length:
            break

    if len(rand_line) > max_length:
        b_start = rng.randint(0,len(rand_line) - max_length)
        tokens_b = rand_line[b_start:b_start + max_length]
    else:
        tokens_b = rand_line

    return tokens_b

def create_instances_from_document(
        document, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq,
        vocab_words, rng, random_seed, file_size,
        copy_f, tokenizer):
    """Creates `TrainingInstance`s for a single document."""

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3
    min_sent_length = int(max_seq_length / 6)

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2*min_sent_length + 1, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    
    i = 0
    while i < len(document) - min_sent_length:
        is_random_next = False
        if len(document) - i >= target_seq_length:
            a_end = rng.randint(i + min_sent_length, i + target_seq_length - min_sent_length)
            tokens_a = document[i:a_end+1]

            if rng.random() < 0.5:
                is_random_next = True
                tokens_b = get_random_sentence(file_size, rng, copy_f, tokenizer, min_sent_length, target_seq_length - len(tokens_a))
                i -= len(tokens_b)
            else:
                tokens_b = document[a_end:i+target_seq_length]

        elif len(document) - i >= 2*min_sent_length:
            a_end = rng.randint(i + min_sent_length, len(document) - min_sent_length)
            tokens_a = document[i:a_end]
            tokens_b = document[a_end:]
        else:
            tokens_a = document[i:]
            is_random_next = True
            tokens_b = get_random_sentence(file_size, rng, copy_f, tokenizer, min_sent_length, target_seq_length - len(tokens_a))
            i -= len(tokens_b)

        i += target_seq_length

        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)


        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            if token != " ":
                tokens.append(token)
                segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            if token != " ":
                tokens.append(token)
                segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if len(tokens) < 2*min_sent_length + 3:
            continue

        (tokens,masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(tokens=tokens, segment_ids=segment_ids,
                                    is_random_next=is_random_next,
                                    masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    all_indexes = list(range(0,len(tokens)))

    rng.shuffle(all_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lm_labels = [-1] * len(tokens)
    masked_lms = 0
    covered_indexes = set()
    for index in all_indexes:
        if masked_lms >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        if tokens[index] == "[CLS]" or tokens[index] == "[SEP]":
            continue
        #Don't know if this is needed
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token
        masked_lm_labels[index] = tokens[index]

        masked_lms += 1

    return (output_tokens, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model",
                        default=None,
                        type=str,
                        required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="Input raw text file (or comma-separated list of files).")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="Output TF example file (or comma-separated list of files).")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    parser.add_argument("--max_predictions_per_seq",
                        default=20,
                        type=int,
                        help="Maximum number of masked LM predictions per sequence.")
    parser.add_argument("--random_seed",
                        default=12345,
                        type=int,
                        help="Random seed for data generation.")
    parser.add_argument("--dupe_factor",
                        default=10,
                        type=int,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--masked_lm_prob",
                        default=0.15,
                        type=float,
                        help="Masked LM probability.")
    parser.add_argument("--short_seq_prob",
                        default=0.1,
                        type=float,
                        help="Probability of creating sequences which are shorter than the maximum length")

    args = parser.parse_args()


    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    rng = random.Random(args.random_seed)
    create_training_instances2(args.input_file, args.output_file, tokenizer, args.max_seq_length, args.dupe_factor,
                               args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq, rng, args.random_seed)


if __name__ == "__main__":
    main()
