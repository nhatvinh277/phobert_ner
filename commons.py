from logging.handlers import RotatingFileHandler
from torch.utils.data.dataset import Dataset
import os
import re
import string
import json
import logging
import torch

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = RotatingFileHandler(
            log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

# class NERdataset(Dataset):
#     def __init__(self, features, device):
#         self.features = features
#         self.device = device

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         sample = self.features[0]
#         token_id_tensors = torch.tensor(sample.token_ids, dtype=torch.long).to(device=self.device)
#         attention_mask_tensors = torch.tensor(sample.attention_masks, dtype=torch.long).to(device=self.device)
#         token_type_ids_tensors = torch.tensor(sample.token_type_ids, dtype=torch.long).to(device=self.device)
#         token_mask_tensors = torch.tensor(sample.token_masks, dtype=torch.long).to(device=self.device)

#         return token_id_tensors, attention_mask_tensors, token_type_ids_tensors, token_mask_tensors

class NERdataset:
    def __init__(self, tokenizer, max_seq_length, examples, labels):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = examples
        self.labels = labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        example = self.examples[item]
        label = self.labels[item]

        ids = []
        target_tag = []

        for i, s in enumerate(example):
            inputs = self.tokenizer.encode(s, add_special_tokens=False)
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([label[i]] * input_len)

        ids = ids[:self.max_seq_length - 2]
        target_tag = target_tag[:self.max_seq_length - 2]

        ids = [0] + ids + [2] # <s>: 0, </s>: 2, <pad>: 1
        target_tag = [12] + target_tag + [12] # O: 12

        mask = [1] * len(ids) # 1: not masked, 0: masked
        token_type_ids = [0] * len(ids)

        padding_len = self.max_seq_length - len(ids)

        ids = ids + ([1] * padding_len) # padding
        mask = mask + ([0] * padding_len) # 1: not masked, 0: masked
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([12] * padding_len) # O: 12

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.tensor(mask, dtype=torch.long),
            "attention_mask": torch.tensor(token_type_ids, dtype=torch.long),
            "target_label": torch.tensor(target_tag, dtype=torch.long),
        }