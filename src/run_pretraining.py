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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

logger.info(PYTORCH_PRETRAINED_BERT_CACHE)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def get_rates(out, labels):
    outputs = np.argmax(out, axis=1)
    adder = outputs + labels
    TP = len(adder[adder == 2])
    TN = len(adder[adder == 0])
    subtr = labels - outputs
    FP = len(subtr[subtr == -1])
    FN = len(subtr[subtr == 1])

    return np.array([TP, TN, FP, FN])

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def get_scores(rates):

    [TP, TN, FP, FN] = rates

    balanced_acc = ((TP / (TP+FN)) + (TN / (TN+FP))) / 2
    mcc = (TP*TN - FP*FN) / math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

    precision_2 = TP / (TP + FP)
    precision_1 = TN / (TN + FN)
    recall_2 = TP / (TP + FN)
    recall_1 = TN / (TN + FP)
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
    f1_2 = (2 * precision_2 * recall_2) / (precision_2 + recall_2)

    return balanced_acc, f1_1, f1_2, mcc

class BertDataset(Dataset):
    def __init__(self, input_file, input_length=0):
        self.file = open(input_file, "r", encoding="utf-8")
        self.input_file = input_file
        if input_length == 0:
            self.input_length = sum([1 for line in self.file])
        else:
            self.input_length = input_length
        self.sample_count = 0

    def __len__(self):
        return self.input_length

    def __getitem__(self, item):

        self.sample_count += 1
        if self.sample_count == self.input_length:
            self.file = open(self.input_file, "r", encoding="utf-8")
            self.sample_count = 0

        line = self.file.__next__().strip()
        line = line.split("\t")
        input_ids = eval(line[0])
        input_mask = eval(line[1])
        segment_ids = eval(line[2])
        masked_lm_ids = eval(line[3])
        next_sent_label = eval(line[4])

        tensors = (torch.tensor(input_ids),
                   torch.tensor(input_mask),
                   torch.tensor(segment_ids),
                   torch.tensor(masked_lm_ids),
                   torch.tensor(next_sent_label))

        return tensors

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input file. Every line is an instance")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_input_file",
                        default=None,
                        type=str,
                        help="The eval input file. Every line is an instance")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--input_length",
                        default=0,
                        type=int,
                        help="Length of the input.")
    parser.add_argument("--eval_input_length",
                        default=0,
                        type=int,
                        help="Length of the eval input.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # os.makedirs(args.output_dir, exist_ok=True)

    # Prepare model
    model = BertForPreTraining.from_pretrained(args.bert_model, PYTORCH_PRETRAINED_BERT_CACHE)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
#    logger.info(optimizer_grouped_parameters)
#    logger.info([str(n) for n,p in param_optimizer if p.grad is not None])

    global_step = 0
    if args.do_train:
        if args.input_length == 0:
            with open(args.input_file, "r", encoding="utf-8") as f:
                input_length = sum([1 for line in f])
        else:
            input_length = args.input_length

        num_train_steps = int(input_length / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", input_length)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
#        logger.info("inpput_id size = %d", len(train_features[0].input_ids))

        train_dataset = BertDataset(args.input_file, input_length)
        eval_dataset = BertDataset(args.eval_input_file, args.eval_input_length)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = RandomSampler(eval_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

        logger.info("Created DataLoader")
        best_loss = 5.0
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                input_ids, input_mask, segment_ids, masked_lm_ids, next_sent_label = tuple(t.to(device) for t in batch)
                loss = model(input_ids, segment_ids, input_mask, masked_lm_ids, next_sent_label)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                # if (step + 1) % args.gradient_accumulation_steps == 0:
                #     # modify learning rate with special warm up BERT uses
                #     lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = lr_this_step
                #     optimizer.step()

                lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()

                optimizer.zero_grad()

                if (step + 1) % 5000 == 0:
                    # input_ids=input_mask=segment_ids=masked_lm_ids=next_sent_label=loss=None
                    # del input_ids,input_mask,segment_ids,masked_lm_ids,next_sent_label,loss
                    # torch.cuda.empty_cache()
                    model.eval()
                    bcount = 0
                    total_loss = 0.0
                    for eval_batch in eval_dataloader:
                        eval_batch = tuple(t.to(device) for t in eval_batch)
                        input_ids, input_mask, segment_ids, masked_lm_ids, next_sent_label = eval_batch
                        cur_loss = model(input_ids, segment_ids, input_mask, masked_lm_ids, next_sent_label)
                        if n_gpu > 1:
                            cur_loss = cur_loss.mean() # mean() to average on multi-gpu.

                        total_loss += cur_loss.item()
                        bcount += 1

                    curr_loss = total_loss / bcount
                    logger.info("Loss = " + str(curr_loss))
                    if best_loss > curr_loss:
                        best_loss = curr_loss
                        logger.info("** Saving model - Loss = " + str(best_loss) + " **")
                        model_to_save = model.module if hasattr(model, 'module') else model  # To handle multi gpu
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)

                    # del input_ids,input_mask,segment_ids,masked_lm_ids,next_sent_label,cur_loss
                    # torch.cuda.empty_cache()
                    model.train()

                global_step += 1

        # save_model = model.module if hasattr(model, 'module') else model  # To handle multi gpu
        # output_file = os.path.join(args.data_dir, "pytorch_model.bin")
        # torch.save(save_model.state_dict(), output_file)

    # if args.do_eval:
    #     eval_examples = processor.get_dev_examples(args.data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, args.max_seq_length, tokenizer)
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #     if args.local_rank == -1:
    #         eval_sampler = SequentialSampler(eval_data)
    #     else:
    #         eval_sampler = DistributedSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     total_rates = np.array([0,0,0,0])
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #     for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    #         input_ids = input_ids.to(device)
    #         input_mask = input_mask.to(device)
    #         segment_ids = segment_ids.to(device)
    #         label_ids = label_ids.to(device)

    #         with torch.no_grad():
    #             tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

    #         logits = logits.detach().cpu().numpy()
    #         label_ids = label_ids.to('cpu').numpy()
    #         tmp_eval_accuracy = accuracy(logits, label_ids)
    #         tmp_rates = get_rates(logits, label_ids)

    #         eval_loss += tmp_eval_loss.mean().item()
    #         eval_accuracy += tmp_eval_accuracy
    #         total_rates += tmp_rates

    #         nb_eval_examples += input_ids.size(0)
    #         nb_eval_steps += 1

    #     eval_loss = eval_loss / nb_eval_steps
    #     eval_accuracy = eval_accuracy / nb_eval_examples

    #     balanced_acc, f1_neg, f1_pos, mcc = get_scores(total_rates.tolist())

    #     result = {'eval_loss': eval_loss,
    #               'eval_accuracy': eval_accuracy,
    #               'global_step': global_step,
    #               'balanced_accuracy' : balanced_acc,
    #               'f1_neg' : f1_neg,
    #               'f1_pos' : f1_pos,
    #               'mcc' : mcc,
    #               'loss': tr_loss/nb_tr_steps}

    #     output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         logger.info("***** Eval results *****")
    #         for key in sorted(result.keys()):
    #             logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
