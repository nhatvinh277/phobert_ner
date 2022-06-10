import os
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from commons import NERdataset, logger, init_logger
from processor import NERProcessor
from transformers import PhobertTokenizer
from tqdm import tqdm
from model import *
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score

def build_dataset(args, processor, data_type='train', feature=None, device=torch.device('cpu')):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        examples, labels = processor.get_example(data_type, feature is not None)
        
        # features = processor.convert_examples_to_features(examples, labels, args.max_seq_length, feature)
        # print("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)

    return NERdataset(processor.tokenizer, processor.max_seq_length, examples, tags=labels)

def train_function(iterator, model, optimizer, device, scheduler):
    model.train()
    tr_loss = 0
    for data in tqdm(iterator, total=len(iterator)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        tr_loss += loss.item()
    return tr_loss / len(iterator)


def evaluate(iterator, model, device):
    model.eval()
    eval_loss = 0
    for data in tqdm(iterator, total=len(iterator)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss = model(**data)
        eval_loss += loss.item()
    return eval_loss / len(iterator)

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    summary_writer = SummaryWriter(args.log_dir)
    init_logger(f"{args.output_dir}/vner_trainning.log")
    
    tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    processor = NERProcessor(args.data_dir, tokenizer)

    num_labels = processor.get_num_labels()
    config, model = model_builder(model_name_or_path=args.model_name_or_path,
                                    num_labels=num_labels)
    model.to(device)

    logger.info("Prepare dataset ...")

    # train_data = build_dataset(args, processor, data_type='train', feature=None, device=device)
    train_examples, train_labels = processor.get_example(data_type='train')
    train_data = NERdataset(processor.tokenizer, args.max_seq_length, train_examples, train_labels)
    # train_sampler = RandomSampler(train_data)
    train_iterator = DataLoader(train_data, batch_size=args.train_batch_size)

    # eval_data = build_dataset(args, processor, data_type='test', feature=None, device=device)
    eval_examples, eval_labels = processor.get_example(data_type='test')
    eval_data = NERdataset(processor.tokenizer, args.max_seq_length, eval_examples, eval_labels)
    # eval_sampler = RandomSampler(eval_data)
    eval_iterator = DataLoader(eval_data, batch_size=args.eval_batch_size)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = len(train_iterator) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    logger.info("="*30 + f"Summary" + "="*30)
    logger.info("MODEL:")
    logger.info(f"\tBERT model: {args.model_name_or_path}")
    logger.info(f"\tNumber of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("DATASET:")
    logger.info(f"\tNumber of train Examples: {len(train_data)}")
    logger.info(f"\tNumber of eval Examples: {len(eval_data)}")
    logger.info(f"\tNumber of labels: {len(processor.labels)}")
    logger.info("Hyper-Parameters:")
    logger.info(f"\tMax sequence length: {args.max_seq_length}")
    logger.info(f"\tLearning rate: {args.learning_rate}")
    logger.info(f"\tNumber of epochs: {args.num_train_epochs}")
    logger.info(f"\tTrain batch size: {args.train_batch_size}")
    logger.info(f"\tEval batch size: {args.eval_batch_size}")
    logger.info(f"\tAdam epsilon: {args.adam_epsilon}")
    logger.info(f"\tWeight decay: {args.weight_decay}")
    logger.info(f"\tWarmup Proportion: {args.warmup_proportion}")
    logger.info(f"\tMax grad norm: {args.max_grad_norm}")
    logger.info(f"\tGradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"\tSeed: {args.seed}")
    logger.info(f"\tCuda: {args.cuda}")
    logger.info(f"\tFeat config: {args.feat_config}")
    logger.info(f"\tUse one-hot embbeding: {args.one_hot_emb}")
    logger.info(f"\tOutput directory: {args.output_dir}")
    logger.info(f"\tLog directory: {args.log_dir}")


    model.train()
    best_loss = -1
    for e in range(int(args.num_train_epochs)):
        logger.info("="*30 + f"Epoch {e}" + "="*30)
        tr_loss = train_function(train_iterator, model, optimizer, device, scheduler)
        eval_loss = evaluate(eval_iterator, model, device)

        print(f"Epoch {e}: Train Loss = {tr_loss} Valid Loss = {eval_loss}")
        if eval_loss < best_loss:
            model_path = f"{args.output_dir}/vner_model.bin"
            torch.save(model.state_dict(), model_path)
            best_loss = eval_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--log_dir", default=None, type=str, required=True)

    # Other parameters
    parser.add_argument("--feat_config", default=None, type=str)
    parser.add_argument("--one_hot_emb", action='store_true')
    parser.add_argument("--use_lstm", action='store_true')
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run(args)