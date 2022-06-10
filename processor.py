import re
import os
import csv
import sys
import argparse
import tensorflow as tf
from transformers import PhobertTokenizer
from tqdm import tqdm
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary


maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def normalize_text(txt):
    txt = re.sub("\xad|\u200b", "", txt)
    return txt.strip()

class Example:
    def __init__(self, eid: int, tokens: str, token_ids: list, attention_masks: list, token_type_ids: list, token_masks: list):
        self.eid = eid
        self.tokens = tokens
        self.token_ids = token_ids
        self.token_masks = token_masks
        self.attention_masks = attention_masks
        self.token_type_ids=token_type_ids

class NERProcessor:
    def __init__(self, data_dir: str or None, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        self.label_map = {label: i for i, label in enumerate(self.labels, 1)}

    def get_num_labels(self):
        return len(self.labels) + 1

    def get_example(self, data_type: str = "train"):
        if data_type == "train":
            return self._read_file(os.path.join(self.data_dir, 'train.csv'))
        elif data_type == "dev":
            return self._read_file(os.path.join(self.data_dir, 'dev.csv'))
        elif data_type == "test":
            return self._read_file(os.path.join(self.data_dir, 'test.csv'))
        else:
            print(f"ERROR: {data_type} not found!!!")

    def _read_file(self, file_path: str):
        """Reads a tab separated value file."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            words = []
            examples = []
            labels = []
            labels_ids = []
            for line in reader:
                if len(line) >= 2:
                    words.append(line[0].strip())
                    labels_ids.extend([self.label_map[line[-1].strip()]])                    
                else:
                    examples.append(words)
                    labels.append(labels_ids)
                    labels_ids = []
                    words = []

            return examples, labels

    def convert_examples_to_features(self, examples, labels, max_seq_length, feature=None):
        features = []
        for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
            ex_id, ex_words, ex_labels, ex_feats = example
            # Init Example features
            ids = []
            attention_masks = []

            for i, (word, label) in enumerate(zip(ex_words, ex_labels)):
                inputs = self.tokenizer.encode(example, add_special_tokens=False)
                input_len = len(inputs)
                ids.extend(inputs)
                attention_masks.extend([label] * input_len)

            ids = ids[:max_seq_length - 2]
            attention_masks = attention_masks[:max_seq_length - 2]


            ids = [0] + ids + [2] # <s>: 0, </s>: 2, <pad>: 1
            attention_masks = [12] + attention_masks + [12] # O: 12

            token_masks = [1] * len(ids) # 1: not masked, 0: masked
            token_type_ids = [0] * len(ids)

            padding = max_seq_length - len(ids)

            
            ids = ids + ([1] * padding) # padding
            token_masks = token_masks + ([0] * padding) # 1: not masked, 0: masked
            token_type_ids = token_type_ids + ([0] * padding)
            attention_masks = attention_masks + ([12] * padding) # O: 12

            if(ex_id < 3):
                print("token_ids: ",len(ids), ids)
                print("attention_masks:", len(attention_masks),attention_masks)
                print("token_type_ids: ",len(token_type_ids), token_type_ids)
                print("token_masks: ",len(token_masks), token_masks)
            
            features.append(
                Example(eid=example[0],
                        tokens="",
                        token_ids=ids,
                        attention_masks=attention_masks,
                        token_type_ids=token_type_ids,
                        token_masks=token_masks))

        return features

if __name__ == "__main__":
    # from transformers import BertTokenizer
    # tokenzier = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    processor = NERProcessor("./data/raw_data", tokenizer)
    a,b = processor.get_example("test")
    print("examples: ", a)
    print("labels: ", b)
    # features = processor.convert_examples_to_features(a, 126)