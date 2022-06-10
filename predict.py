import argparse
import os
import torch
import torch.nn as nn
import itertools
import numpy as np

from transformers import RobertaConfig, PhobertTokenizer
from commons import NERdataset
from model import PhoBertLstmCrf
from constant import LABEL2ID, ID2LABEL
from processor import normalize_text
from vncorenlp import VnCoreNLP
from tqdm import tqdm
from processor import Example
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PhobertNER(object):
    def __init__(self, model_path: str = None, max_seq_length: int = 256, no_cuda=False):
        self.max_seq_len = max_seq_length
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'
        self.rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
        self.model, self.tokenizer, self.vocab =self.load_model(model_path, device=self.device)


    @staticmethod
    def load_model(model_path=None, model_clss='vinai/phobert-base', device='cpu'):
        # tokenizer = PhobertTokenizer.from_pretrained(model_clss, use_fast=False)
        parser = argparse.ArgumentParser()
        parser.add_argument('--bpe-codes', 
            default="PhoBERT_base_transformers/bpe.codes",
            required=False,
            type=str,
            help='path to fastBPE BPE'
        )
        bpe_args, unknown = parser.parse_known_args()
        tokenizer = fastBPE(bpe_args)

        vocab = Dictionary()
        vocab.add_from_file("PhoBERT_base_transformers/dict.txt")
        config = RobertaConfig.from_pretrained(model_clss, num_labels=len(LABEL2ID))
        model = PhoBertLstmCrf(config=config)

        if model_path is not None:
            if device == 'cpu':
                checkpoint_data = torch.load(model_path, map_location='cpu')
            else:
                checkpoint_data = torch.load(model_path)
                print("checkpoint_data:", checkpoint_data)
            model.load_state_dict(checkpoint_data)
        model.to(device)
        model.eval()
        return model, tokenizer, vocab

    def convert_sentences_to_features(self, examples, feature=None):
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


            if len(tokens) >= self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
                labels = labels[0:(self.max_seq_length - 2)]
                token_masks = token_masks[0:(self.max_seq_length - 2)]

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
            segment_ids = [0] * self.max_seq_length

            label_masks = [1] * len(label_ids)

            padding = [0] * (self.max_seq_length - len(input_ids[0]))
            input_ids[0].extend(padding)
            attention_masks.extend(padding)
            padding = [0] * (self.max_seq_length - len(token_masks))

            token_masks.extend(padding)
            
            padding = [0] * (self.max_seq_length - len(label_ids))
            label_ids.extend(padding)
            label_masks.extend(padding)

            input_ids = pad_sequences(input_ids, maxlen=self.max_seq_length, dtype="long", value=0, truncating="post", padding="post")

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
            assert len(label_ids) == self.max_seq_length
            assert len(label_masks) == self.max_seq_length
            assert len(token_masks) == self.max_seq_length
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

    def preprocess(self, in_raw: str):
        norm_text = normalize_text(in_raw)
        sentences = self.rdrsegmenter.tokenize(norm_text)
        features = self.convert_sentences_to_features(sentences)
        data = NERdataset(features, self.device)
        return DataLoader(data, batch_size=self.batch_size)
        
    def predict(self, text):
        entites = []
        iterator = self.preprocess(text)
        for step, batch in enumerate(iterator):
            sents, token_ids, attention_masks, token_masks, segment_ids, label_ids, label_masks, feats = batch
            logits = self.model(token_ids, attention_masks, token_masks, segment_ids, label_masks, feats)
            logits = torch.argmax(nn.functional.softmax(logits, dim=-1), dim=-1)
            pred = logits.detach().cpu().numpy()
            entity = None
            words = []
            for sent in sents:
                words.extend(sent.split())
            for p, w in list(zip(pred, words)):
                label = self.label_list[p-1]
                if not label == 'O':
                    prefix, label = label.split('-')
                    if entity is None:
                        entity = (w, label)
                    else:
                        if entity[-1] == label:
                            if prefix == 'I':
                                entity = (entity[0] + f' {w}', label)
                            else:
                                entites.append(entity)
                                entity = (w, label)
                        else:
                            entites.append(entity)
                            entity = (w, label)
                elif entity is not None:
                    entites.append(entity)
                    entity = None
                else:
                    entity = None
        return entites

    def convert_tensor(self, sent):
        # encoding = self.tokenizer(sent,
        #                           padding='max_length',
        #                           truncation=True,
        #                           max_length=self.max_seq_len)
        encoding = self.vocab.encode_line(sent, add_if_not_exist=False).long().tolist()

        valid_id = np.zeros(len(encoding), dtype=int)
        i = 0
        # subwords = self.tokenizer.tokenize(sent)
        subwords = self.tokenizer.encode(sent)
        print("subwords: ", subwords)

        for idx, sword in enumerate(subwords):
            if not sword.endswith('@@'):
                valid_id[idx+1] = 1
                i += 1
            elif idx == 0 or not subwords[idx-1].endswith('@@'):
                valid_id[idx + 1] = 1
                i += 1
            else:
                continue
        label_masks = [1] * i
        label_masks.extend([0] * (self.max_seq_len - len(label_masks)))
        encoding.pop('offset_mapping', None)
        item = {key: torch.as_tensor([val]).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['valid_ids'] = torch.as_tensor([valid_id]).to(self.device, dtype=torch.long)
        item['label_masks'] = torch.as_tensor([label_masks]).to(self.device, dtype=torch.long)
        return item

    def __call__(self, in_raw: str):
        sents = self.preprocess(in_raw)
        entites = []
        for sent in sents:
            item = self.convert_tensor(sent)
            with torch.no_grad():
                tags = self.model(**item)
            entity = None
            for w, l in list(zip(sent.split(), list(itertools.chain(*tags)))):
                tag = ID2LABEL[l]
                if not tag == 'O':
                    prefix, tag = tag.split('-')
                    if entity is None:
                        entity = (w, tag)
                    else:
                        if entity[-1] == tag:
                            if prefix == 'I':
                                entity = (entity[0] + f' {w}', tag)
                            else:
                                entites.append(entity)
                                entity = (w, tag)
                        else:
                            entites.append(entity)
                            entity = (w, tag)
                elif entity is not None:
                    entites.append(entity)
                    entity = None
                else:
                    entity = None
        return entites


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str, required=True)
    parser.add_argument("--max_seq_length", default=128, type=int)

    args = parser.parse_args()
    
    predictor = PhobertNER(args.model_dir, args.max_seq_length)
    while True:
        in_raw = input('Enter text:')
        print(predictor.predict(in_raw))

