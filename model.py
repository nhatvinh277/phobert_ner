from transformers import logging, RobertaForTokenClassification
from transformers import RobertaConfig
from torchcrf import CRF

from constant import LABEL2ID, ID2LABEL
import torch
import torch.nn as nn

logging.set_verbosity_error()

def loss_fn(label, target, attention_mask, num_labels):
    loss_function = nn.CrossEntropyLoss()
    active_loss = attention_mask.view(-1) == 1
    active_logits = label.view(-1, num_labels)
    active_labels = torch.where(active_loss, target.view(-1), torch.tensor(loss_function.ignore_index).type_as(target))
    loss = loss_function(active_logits, active_labels)
    return loss, (active_logits, active_labels)

class PhoBertLstmCrf(RobertaForTokenClassification):
    def __init__(self, config):
        super(PhoBertLstmCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size // 2,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.crf = CRF(config.num_labels, batch_first=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, target_label=None):
        seq_output = self.roberta(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  head_mask=None)[0]
        seq_output, _ = self.lstm(seq_output)

        d_label = self.dropout(seq_output)
        label = self.classifier(d_label)
        loss, _ = loss_fn(label, target_label, attention_mask, self.num_labels)
        return label, loss

def model_builder(model_name_or_path: str,
                    num_labels:int):
    config = RobertaConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = PhoBertLstmCrf(config=config)
    return config, model

# DEBUG
if __name__ == "__main__":
    from transformers import RobertaConfig

    model_name = 'vinai/phobert-base'
    config = RobertaConfig.from_pretrained(model_name, num_labels=7)
    model = PhoBertLstmCrf.from_pretrained(model_name, config=config, from_tf=False)

    input_ids = torch.randint(0, 2999, [2, 20], dtype=torch.long)
    mask = torch.ones([2, 20], dtype=torch.long)
    labels = torch.randint(1, 6, [2, 20], dtype=torch.long)
    new_labels = torch.zeros([2, 20], dtype=torch.long)
    valid_ids = torch.ones([2, 20], dtype=torch.long)
    label_mask = torch.ones([2, 20], dtype=torch.long)
    valid_ids[:, 0] = 0
    valid_ids[:, 13] = 0
    labels[:, 0] = 0
    label_mask[:, -2:] = 0
    for i in range(len(labels)):
        idx = 0
        for j in range(len(labels[i])):
            if valid_ids[i][j] == 1:
                new_labels[i][idx] = labels[i][j]
                idx += 1
    output = model.forward(input_ids,
                           labels=new_labels,
                           attention_mask=mask,
                           valid_ids=valid_ids, label_masks=label_mask)
    print(labels)
    print(new_labels)
    print(label_mask)
    print(valid_ids)
    print(output)
