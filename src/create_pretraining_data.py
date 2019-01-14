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
#import spacy
import codecs
import argparse
from dask import dataframe as dd
from dask.multiprocessing import get

from pytorch_pretrained_bert.tokenization import BertTokenizer

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_labels = masked_lm_labels

def instance_to_columns(row,tokenizer,max_seq_length):

    instance = row.instance['instance']
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

    row["input_ids"] = input_ids
    row["input_mask"] = input_mask
    row["segment_ids"] = segment_ids
    row["masked_lm_labels"] = masked_lm_ids
    row["next_sent_label"] = next_sentence_label

    return row

# def write_instance_to_example_files(instances, tokenizer, max_seq_length,
#                                     max_predictions_per_seq, output_file):
#     """Create TF example files from `TrainingInstance`s."""
#     df = pd.DataFrame(columns=["input_ids","input_mask","segment_ids","masked_lm_labels","next_sent_label"])
#     for (inst_index, instance) in enumerate(instances):
#         input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
#         input_mask = [1] * len(input_ids)
#         segment_ids = list(instance.segment_ids)
#         assert len(input_ids) <= max_seq_length

#         while len(input_ids) < max_seq_length:
#             input_ids.append(0)
#             input_mask.append(0)
#             segment_ids.append(0)

#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length

#         masked_lm_ids = instance.masked_lm_labels
#         for i,x in enumerate(masked_lm_ids):
#             masked_lm_ids[i] = tokenizer.convert_tokens_to_ids([x])[0]

#         next_sentence_label = 1 if instance.is_random_next else 0

#         features = {}
#         features["input_ids"] = input_ids
#         features["input_mask"] = input_mask
#         features["segment_ids"] = segment_ids
#         features["masked_lm_labels"] = masked_lm_ids
#         features["next_sent_label"] = next_sentence_label

#         df = df.append(features, ignore_index=True)

#     df.to_csv(output_file, sep="\t", index=False)

# Cumlelere bolme islemini ayri bi yerde paralel olarak yap!!!! OSMAN
# def create_training_instances(input_file, tokenizer, max_seq_length,
#                               dupe_factor, short_seq_prob, masked_lm_prob,
#                               max_predictions_per_seq, rng):
#     """Create `TrainingInstance`s from raw text."""
#     all_documents = []

#     # Input file format:
#     # (1) One sentence per line. These should ideally be actual sentences, not
#     # entire paragraphs or arbitrary spans of text. (Because we use the
#     # sentence boundaries for the "next sentence prediction" task).
#     # (2) Blank lines between documents. Document boundaries are needed so
#     # that the "next sentence prediction" task doesn't span between documents.
#     df = pd.read_csv(input_file)

#     nlp = spacy.load("en")

#     for ind,row in df.iterrows():
#         doc = nlp(row.text)
#         sentences = [sent.string.strip() for sent in doc.sents]
#         document = []
#         for sent in sentences:
#             sent = sent.strip()
#             if sent:
#                 document.append(sent)
#         if len(document) > 0:
#             all_documents.append(document)

#     print("Created all_documents!!")
#     # with codecs.open(input_file, "r", "utf-8") as reader:
#     #     while True:
#     #         line = reader.readline()

#     #         if not line:
#     #             all_documents.append([])
#     #         else:
#     #             line = line.strip()

#     #             # Empty lines are used as document delimiters
#     #             if not line:
#     #                 all_documents.append([])
#     #             tokens = tokenizer.tokenize(line)
#     #             if tokens:
#     #                 all_documents[-1].append(tokens)

#     # # Remove empty documents
#     # all_documents = [x for x in all_documents if x]
#     rng.shuffle(all_documents)

#     vocab_words = list(tokenizer.vocab.keys())
#     instances = []
#     for _ in range(dupe_factor):
#         for document_index in range(len(all_documents)):
# #            print("Doc " + str(document_index + 1))
#             instances.extend(
#                 create_instances_from_document(
#                     all_documents, document_index, max_seq_length, short_seq_prob,
#                     masked_lm_prob, max_predictions_per_seq, vocab_words, rng))
# #            print("--------------------------------")
#     rng.shuffle(instances)
#     return instances

def get_tokens(row,tokenizer):
    row.tokens = tokenizer.tokenize(row.text)
    return row

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
    df = pd.read_csv(input_file)
    print("Loaded")
    df.hyperpartisan = None
    df['tokens'] = ""
    df['instances'] = ""
    df_instances = pd.DataFrame(columns=["instance","input_ids","input_mask","segment_ids","masked_lm_labels","next_sent_label"])
    df = dd.from_pandas(df,npartitions=8).map_partitions(lambda x : x.apply(lambda row : get_tokens(row,tokenizer), axis=1),meta=df).compute(get=get)
    print("Tokenized")
    df.text = None
#    df = df.apply(lambda row : get_tokens(row,tokenizer), axis=1)
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle
    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for i in range(dupe_factor):
        # for ind,row in df.iterrows():
        #     df_instances = df_instances.append(create_instances_from_document2(
        #         df, ind, max_seq_length, short_seq_prob,
        #         masked_lm_prob, max_predictions_per_seq, vocab_words, rng), ignore_index=True)

        # df = df.apply(lambda row : create_instances_from_document2(df, row, max_seq_length, short_seq_prob,
        #                                                            masked_lm_prob, max_predictions_per_seq, vocab_words, rng), axis=1)
        df = dd.from_pandas(df,npartitions=8).map_partitions(lambda x : x.apply(lambda row : create_instances_from_document2(df, row, max_seq_length, short_seq_prob,
                                                                                                                             masked_lm_prob, max_predictions_per_seq,
                                                                                                                             vocab_words, rng, random_seed),
                                                                                axis=1),meta=df).compute(get=get)

        print("Created instances " + str(i + 1))

        for ind,row in df.iterrows():
            if row.instances:
                df_instances = df_instances.append([{"instance" : x} for x in row.instances], ignore_index=True)

        print("Got instances" + str(i + 1))

        df.instances = None
        

    df = None
    df_instances = df_instances.sample(frac=1).reset_index(drop=True) # Shuffle
    print("instance to columns")
#    df_instances = df_instances.apply(lambda row : instance_to_columns(row,tokenizer,max_seq_length), axis=1)
    df_instances = dd.from_pandas(df_instances,npartitions=8).map_partitions(lambda x : x.apply(lambda row : instance_to_columns(row,tokenizer,max_seq_length),
                                                                                                axis=1),meta=df_instances).compute(get=get)
    print("Done")
    df_instances = df_instances.drop(columns=["instance"])
    
    df_instances.to_csv(output_file, sep="\t", index=False)

# def create_instances_from_document(
#     all_documents, document_index, max_seq_length, short_seq_prob,
#     masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
#     """Creates `TrainingInstance`s for a single document."""
#     document = all_documents[document_index]

#     # Account for [CLS], [SEP], [SEP]
#     max_num_tokens = max_seq_length - 3

#     # We *usually* want to fill up the entire sequence since we are padding
#     # to `max_seq_length` anyways, so short sequences are generally wasted
#     # computation. However, we *sometimes*
#     # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
#     # sequences to minimize the mismatch between pre-training and fine-tuning.
#     # The `target_seq_length` is just a rough target however, whereas
#     # `max_seq_length` is a hard limit.
#     target_seq_length = max_num_tokens
#     if rng.random() < short_seq_prob:
#         target_seq_length = rng.randint(2, max_num_tokens)

#     # We DON'T just concatenate all of the tokens from a document into a long
#     # sequence and choose an arbitrary split point because this would make the
#     # next sentence prediction task too easy. Instead, we split the input into
#     # segments "A" and "B" based on the actual "sentences" provided by the user
#     # input.
#     instances = []
#     current_chunk = []
#     current_length = 0
#     i = 0
#     while i < len(document):
#         segment = document[i]
#         current_chunk.append(segment)
#         current_length += len(segment)
# #        print("Current chunk length : " + str(len(current_chunk)))
#         if i == len(document) - 1 or current_length >= target_seq_length:
#             if current_chunk:
# #                print("Inside")
#                 # `a_end` is how many segments from `current_chunk` go into the `A`
#                 # (first) sentence.
#                 a_end = 1
#                 if len(current_chunk) >= 2:
#                     a_end = rng.randint(1, len(current_chunk) - 1)

#                 tokens_a = []
#                 for j in range(a_end):
#                     tokens_a.extend(current_chunk[j])

#                 tokens_b = []
#                 # Random next
#                 is_random_next = False
#                 if len(current_chunk) == 1 or rng.random() < 0.5:
# #                    print("Random")
# #                    print("Current i : " + str(i))
# #                    print("a_end is : " + str(a_end))
#                     is_random_next = True
#                     target_b_length = target_seq_length - len(tokens_a)

#                     # This should rarely go for more than one iteration for large
#                     # corpora. However, just to be careful, we try to make sure that
#                     # the random document is not the same as the document
#                     # we're processing.
#                     for _ in range(10):
#                         random_document_index = rng.randint(0, len(all_documents) - 1)
#                         if random_document_index != document_index:
#                             break

#                     random_document = all_documents[random_document_index]
#                     random_start = rng.randint(0, len(random_document) - 1)
#                     for j in range(random_start, len(random_document)):
#                         tokens_b.extend(random_document[j])
#                         if len(tokens_b) >= target_b_length:
#                             break
#                     # We didn't actually use these segments so we "put them back" so
#                     # they don't go to waste.
#                     num_unused_segments = len(current_chunk) - a_end
#                     i -= num_unused_segments
# #                    print("End i : " + str(i))
#                 # Actual next
#                 else:
# #                    print("Actual")
#                     is_random_next = False
#                     for j in range(a_end, len(current_chunk)):
#                         tokens_b.extend(current_chunk[j])
#                 truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

#                 assert len(tokens_a) >= 1
#                 assert len(tokens_b) >= 1

#                 tokens = []
#                 segment_ids = []
#                 tokens.append("[CLS]")
#                 segment_ids.append(0)
#                 for token in tokens_a:
#                     if token != " ":
#                         tokens.append(token)
#                         segment_ids.append(0)

#                 tokens.append("[SEP]")
#                 segment_ids.append(0)

#                 for token in tokens_b:
#                     if token != " ":
#                         tokens.append(token)
#                         segment_ids.append(1)
#                 tokens.append("[SEP]")
#                 segment_ids.append(1)
# #                print("Before masked lm preds")
#                 (tokens,masked_lm_labels) = create_masked_lm_predictions(
#                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
# #                print("After masked lm preds")
#                 instance = TrainingInstance(
#                   tokens=tokens,
#                   segment_ids=segment_ids,
#                   is_random_next=is_random_next,
#                   masked_lm_labels=masked_lm_labels)
#                 instances.append(instance)
#             current_chunk = []
#             current_length = 0
#         i += 1

#     return instances


def create_instances_from_document2(
        df, row, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq,
        vocab_words, rng, random_seed):
    """Creates `TrainingInstance`s for a single document."""
    document_index = row.name
    document = row.tokens

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
                while True:
                    rand_document = df[df.index != document_index].sample(n=1, random_state=random_seed).tokens.iloc[0]
                    if len(rand_document) >= min_sent_length:
                        break

                if len(rand_document) > target_seq_length - len(tokens_a):
                    b_start = rng.randint(0,len(rand_document) - target_seq_length + len(tokens_a))
                    tokens_b = rand_document[b_start:b_start+target_seq_length-len(tokens_a)]
                else:
                    tokens_b = rand_document

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
            while True:
                rand_document = df[df.index != document_index].sample(n=1, random_state=random_seed).tokens.iloc[0]
                if len(rand_document) >= min_sent_length:
                    break

            if len(rand_document) >= target_seq_length - len(tokens_a):
                b_start = rng.randint(0,len(rand_document) - target_seq_length + len(tokens_a))
                tokens_b = rand_document[b_start:b_start+target_seq_length - len(tokens_a)]
            else:
                tokens_b = rand_document

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
        instance = {"instance" : TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_labels=masked_lm_labels)}
        instances.append(instance)

    row.instances = instances
    return row


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

#    print("Now to writing")
    # write_instance_to_example_files(instances, tokenizer, args.max_seq_length,
    #                                 args.max_predictions_per_seq, args.output_file)


if __name__ == "__main__":
    main()
