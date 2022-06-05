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

class NERdataset(Dataset):
    def __init__(self, features, device):
        self.features = features
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[0]
        token_id_tensors = torch.tensor(sample.token_ids, dtype=torch.long).to(device=self.device)
        token_mask_tensors = torch.tensor(sample.token_masks, dtype=torch.long).to(device=self.device)
        attention_mask_tensors = torch.tensor(sample.attention_masks, dtype=torch.long).to(device=self.device)
        label_id_tensors = torch.tensor(sample.label_ids, dtype=torch.long).to(device=self.device)
        label_mask_tensors = torch.tensor(sample.label_masks, dtype=torch.long).to(device=self.device)
        segment_id_tensors = torch.tensor(sample.segment_ids, dtype=torch.long).to(device=self.device)
        
        feat_tensors = {}
        return sample.tokens, token_id_tensors, attention_mask_tensors, token_mask_tensors, segment_id_tensors, label_id_tensors, label_mask_tensors, feat_tensors