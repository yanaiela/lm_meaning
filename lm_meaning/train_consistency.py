import argparse
import glob
import json
import logging
import os
import pickle
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(block_size) + '_' + filename)
        print(cached_features_file)
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


# This file originates from HuggingFace's run_language_modeling.py and was adapted to our needs.

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        print(file_path)
        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file_h = os.path.join(directory, "bert_cached_lm_h" + str(block_size) + "_" + filename)
        cached_features_file_p = os.path.join(directory, "bert_cached_lm_p" + str(block_size) + "_" + filename)
        if args.overwrite_cache:
            print(args.overwrite_cache)
        if os.path.exists(cached_features_file_p) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file_p)
            with open(cached_features_file_p, "rb") as handle:
                examples_p = pickle.load(handle)
        if os.path.exists(cached_features_file_h) and not args.overwrite_cache:
            with open(cached_features_file_h, "rb") as handle:
                examples_h = pickle.load(handle)

        else:
            logger.info("Creating features from datasets file at %s", directory)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            lines = tokenizer.batch_encode_plus(lines, add_special_tokens=True, padding=True, max_length=block_size)[
                "input_ids"]

            examples_p = lines[::2]
            examples_h = lines[1::2]

            logger.info("Saving features into cached file %s", cached_features_file_p)
            with open(cached_features_file_p, "wb") as handle:
                pickle.dump(examples_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cached_features_file_h, "wb") as handle:
                pickle.dump(examples_h, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.examples = [(p,h) for p,h in zip(examples_p, examples_h)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, file_path, tokenizer):
    return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)

def load_and_cache_examples_natural(args, file_path, tokenizer):
    return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
               ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def mask_objs(inputs: torch.Tensor, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    indices_replace_x, indices_replace_y = torch.where(inputs[:,0]==tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
    #indices_replace_y = indices_replace_y - 2
    #inputs[:,0][indices_replace_x, indices_replace_y] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    model.eval()
    inputs = inputs.to(args.device)
    outputs = model(inputs[:,0])
    predicted_index = torch.argmax(outputs[0][indices_replace_x, indices_replace_y], 1)

    labels = inputs[:, 1].clone()
    labels.to(args.device)
    """indices_replace_x, indices_replace_y = torch.where(inputs[:,1]==tokenizer.convert_tokens_to_ids("[SEP]"))
    indices_replace_y = indices_replace_y - 2
    inputs[:,1][indices_replace_x, indices_replace_y] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)"""
    masked_indices = inputs[:,1] == tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    labels[masked_indices] = predicted_index

    return inputs[:,1], labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_dataset_mlm=[]) -> Tuple[int, float]:
    """ Train the model """
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    """log_dir = os.path.join(config.output_dir, 'runs', args.relation,
                           os.path.basename(args.output_dir) + '_' + current_time)
    tb_writer = SummaryWriter(log_dir=log_dir)"""

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size, collate_fn=collate
    )

    train_sampler = RandomSampler(train_dataset_mlm)
    train_dataloader_mlm = DataLoader(
        train_dataset_mlm, sampler=train_sampler, batch_size=args.batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproducibility
    language_modeling = True
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            batch_mlm = next(iter(train_dataloader_mlm))
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs, labels = mask_tokens(batch_mlm, tokenizer, args)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()



            inputs_consistency, labels_consistency = mask_objs(batch, model, tokenizer, args)

            model.train()
            outputs = model(inputs_consistency, masked_lm_labels=labels_consistency)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            language_modeling = True


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if global_step%int(len(epoch_iterator)/10)==0:
                checkpoint_prefix = "checkpoint"
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
                _rotate_checkpoints(args, checkpoint_prefix)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

    # tb_writer.close()

    return global_step, tr_loss / global_step


def batchify_dict(d, args, tokenizer):
    masked_queries = [f'{query} {tokenizer.mask_token}' for query in d]
    masked_queries = [torch.Tensor(seq) for seq in tokenizer.batch_encode_plus(masked_queries)['input_ids']]
    masked_queries = pad_sequence(masked_queries, batch_first=True, padding_value=tokenizer.pad_token_id)
    batches = np.split(masked_queries, list(range(0, len(masked_queries), args.batch_size))[1:])
    return d, batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', '-d', required=True, type=str, help='dataset used for train')
    parser.add_argument('--lm', '-lm', default="bert-large-cased", type=str, help='which model should be trained')
    parser.add_argument('--output_dir', '-o', type=str, default='',
                        help='folder to save the model.')

    parser.add_argument('--epochs', type=int, default='3', help='Default is 2000 epochs')
    parser.add_argument('--batch_size', type=int, default='180', help='Default is batch size of 256')
    parser.add_argument('--logging_steps', type=int, default='200', help='After how many batches metrics are logged')
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--block_size",
        default=128,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training datasets will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=6e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=-1,
        help="Saves this many checkpoints and deletes older ones",
    )
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--mlm_LAMA", action="store_true",
                        help="If MLM should be done using natural text")
    parser.add_argument('--dataset_name_mlm', '-dn', type=str, help='wikipedia dataset used for training')
    #parser.add_argument("--gpu_device", type=int, default=0, help="gpu number")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    metadata = args.dataset_name.split("/")[-2]
    with open("/".join(args.dataset_name.split("/")[:-1]) + "/log.txt") as f_log:
        metadata += "_"
        for line in f_log:
            metadata+= line.strip()
            metadata+="-"

    metadata = metadata.strip("-")
    metadata += "_"
    metadata += args.lm

    metadata += "_1ranked"

    metadata += "_no-typed"


    if args.mlm_LAMA:
        metadata += "_no-wiki"
    else:
        metadata += "_wiki"


    if args.mlm_LAMA:
        metadata += "_lama_"
    else:
        metadata += "_no-lama_"

    metadata+="1_1.0_none"
    """metadata+=str(args.loss_scaling)
    metadata+="_none"""
    args.output_dir += metadata
    args.output_dir += "/"
    print(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    args.train_data_file = args.dataset_name + "train.txt"

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # Set seed
    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    # Load pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.lm, use_fast=True)

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    model = BertForMaskedLM.from_pretrained(args.lm)
    model.to(args.device)


    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    logger.info("Training/evaluation parameters %s", args)



    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    train_dataset = LineByLineTextDataset(tokenizer, args, args.train_data_file, block_size=args.block_size)

    train_dataset_mlm = TextDataset(tokenizer, args, args.dataset_name_mlm, block_size=args.block_size)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # train
    global_step, tr_loss = train(args, train_dataset, model, tokenizer, train_dataset_mlm)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using
    # from_pretrained()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (model.module if hasattr(model, "module") else model)
    # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()
