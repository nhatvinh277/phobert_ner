import re
import os
import csv
import sys
import logging
from transformers import PreTrainedTokenizer
from typing import List
from typing import List, TextIO, Union
from commons import InputExample, Split, TokenClassificationTask

logger = logging.getLogger(__name__)

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

class NERProcessor(TokenClassificationTask):
    def __init__(self, label_idx=-1):
        self.label_idx = label_idx

    def get_num_labels(self):
        return len(self.labels)

    def get_examples(self, data_dir, mode: Union[Split, str]):
        if isinstance(mode, Split):
            mode = mode.value
        """Reads a tab separated value file."""
        file_path = os.path.join(data_dir, f"{mode}.csv")
        guid_index = 1
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split("\t")
                    words.append(splits[0].strip())
                    if len(splits) > 2:
                        labels.append(splits[self.label_idx].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for line in test_input_reader:
            if line == "" or line == "\n":
                writer.write(line)
                if not preds_list[example_id]:
                    example_id += 1
            elif preds_list[example_id]:
                output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                writer.write(output_line)
            else:
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

if __name__ == "__main__":
    # from transformers import BertTokenizer
    # tokenzier = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer = PreTrainedTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    processor = NERProcessor("./data/raw_data", tokenizer)
    a,b = processor.get_example("test")
    print("examples: ", a)
    print("labels: ", b)
    # features = processor.convert_examples_to_features(a, 126)