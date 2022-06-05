import re
import os
import csv
import sys
import argparse
import tensorflow as tf
from tqdm import tqdm
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
    def __init__(self, eid: int, tokens: str, token_ids: list, token_masks: list, segment_ids: list,
                 label_ids: list, label_masks: list, attention_masks: list, feats: dict):
        self.eid = eid
        self.tokens = tokens
        self.token_ids = token_ids
        self.token_masks = token_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_masks = label_masks
        self.attention_masks = attention_masks
        self.feats = feats

class NERProcessor:
    def __init__(self, data_dir: str or None, bpe, vocab):
        print(data_dir)
        self.data_dir = data_dir
        self.tokenizer = bpe
        self.vocab = vocab
        self.labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        self.label_map = {label: i for i, label in enumerate(self.labels, 1)}

    def get_num_labels(self):
        return len(self.labels) + 1

    def get_example(self, data_type: str = "train", use_feats: bool = False):
        if data_type == "train":
            return self._read_file(os.path.join(self.data_dir, 'train.csv'), use_feats)
        elif data_type == "dev":
            return self._read_file(os.path.join(self.data_dir, 'dev.csv'), use_feats)
        elif data_type == "test":
            return self._read_file(os.path.join(self.data_dir, 'test.csv'), use_feats)
        else:
            print(f"ERROR: {data_type} not found!!!")

    @staticmethod
    def _read_file(file_path: str, use_feats: bool = False):
        """Reads a tab separated value file."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            eid = 0
            words = []
            feats = []
            labels = []
            examples = []
            for line in reader:
                if len(line) >= 2:
                    words.append(line[0].strip())
                    labels.append(line[-1].strip())
                    if use_feats:
                        feat = []
                        for item in line[1:-1]:
                            k, v = item.split("]")
                            feat.append((f"{k}]", v))
                        feats.append(feat)
                else:
                    examples.append((eid, words, labels, feats))
                    words = []
                    feats = []
                    labels = []
                    eid += 1
            return examples

    def convert_examples_to_features(self, examples, max_seq_length, feature=None):
        features = []
        for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
            ex_id, ex_words, ex_labels, ex_feats = example
            # Init Example features
            tokens = []
            labels = []
            feats = {}
            token_masks = []

            ntokens = []

            input_ids = []

            label_ids = []
            for i, (word, label) in enumerate(zip(ex_words, ex_labels)):
                #token = self.tokenizer.tokenize(word)
                token = self.tokenizer.encode(word)
                subwords = token.split()
                words = []
                for subword in subwords:
                    words.append(subword)
                
                tokens.extend(words)              
                for m in range(len(words)):
                    if m == 0:
                        labels.append(label)
                        token_masks.append(1)
                    else:
                        token_masks.append(0)
                        labels.append("[PAD]")


            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                token_masks = token_masks[0:(max_seq_length - 2)]

            # Add [CLS] token
            ntokens.append("[CLS]")
            token_masks.insert(0, 0)
            
            for i, token in enumerate(tokens):
                ntokens.append(token)
                if len(labels) > i and not labels[i] == "[PAD]":
                    label_ids.append(self.label_map[labels[i]])

            # Add [SEP] token
            ntokens.append("[SEP]")
            token_masks.append(0)
            
            sent = " ".join(ntokens)
            # print("sentence: ", sent)
             
            encoded_sent = self.vocab.encode_line(sent, add_if_not_exist=False).long().tolist()
            input_ids.append(encoded_sent)

            attention_masks = [1] * len(input_ids[0])
            segment_ids = [0] * max_seq_length

            label_masks = [1] * len(label_ids)

            padding = [0] * (max_seq_length - len(input_ids[0]))
            input_ids[0].extend(padding)
            attention_masks.extend(padding)
            padding = [0] * (max_seq_length - len(token_masks))

            token_masks.extend(padding)
            
            padding = [0] * (max_seq_length - len(label_ids))
            label_ids.extend(padding)
            label_masks.extend(padding)

            input_ids = pad_sequences(input_ids, maxlen=max_seq_length, dtype="long", value=0, truncating="post", padding="post")

            # print("input_ids: ",len(input_ids[0]), input_ids[0])
            # print("input_ids: ",len(input_ids), input_ids)
            # print("token_masks: ",len(token_masks), token_masks)
            # print("attention_masks:", len(attention_masks),attention_masks)
            # print("segment_ids: ", len(segment_ids), segment_ids)
            # print("labels: ", len(labels), labels)
            # print("label_ids: ", len(label_ids), label_ids)
            # print("label_masks: ", len(label_masks), label_masks)


            # assert len(input_ids[0]) == max_seq_length
            # assert len(attention_masks) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(label_masks) == max_seq_length
            assert len(token_masks) == max_seq_length
            assert sum(token_masks) == sum(label_masks)

            if ex_index < 5:
                print("*** Example ***")
                print("guid: %s" % (example[0]))
                print("tokens: %s" % " ".join([str(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids[0]]))
                print("token_masks: %s" % " ".join([str(x) for x in token_masks]))
                print("attention_masks: %s" % " ".join([str(x) for x in attention_masks]))
                print("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                print("label_mask: %s" % " ".join([str(x) for x in label_masks]))
                print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                print("feats:")
                for k, v in feats.items():
                    print(f"\t{k}: {v}")

            features.append(
                Example(eid=example[0],
                        tokens="",
                        token_ids=input_ids[0],
                        attention_masks=attention_masks,
                        segment_ids=segment_ids,
                        label_ids=label_ids,
                        label_masks=label_masks,
                        token_masks=token_masks,
                        feats=feats))

        return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe-codes', 
        default="PhoBERT_base_transformers/bpe.codes",
        required=False,
        type=str,
        help='path to fastBPE BPE'
    )
    bpe_args, unknown = parser.parse_known_args()
    bpe = fastBPE(bpe_args)
    # from transformers import BertTokenizer
    # tokenzier = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    vocab = Dictionary()
    vocab.add_from_file("PhoBERT_base_transformers/dict.txt")
    processor = NERProcessor("./data/raw_data", bpe, vocab)
    a = processor.get_example("test")
    print("example: ", a)
    features = processor.convert_examples_to_features(a, 126)