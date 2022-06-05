from transformers import logging, RobertaForTokenClassification
from transformers import RobertaConfig
from torchcrf import CRF

from constant import LABEL2ID, ID2LABEL
import torch
import torch.nn as nn

logging.set_verbosity_error()


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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        seq_output = self.roberta(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  head_mask=None)[0]
        seq_output, _ = self.lstm(seq_output)

        batch_size, max_len, feat_dim = seq_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=seq_output.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = seq_output[i][j]

        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        seq_tags = self.crf.decode(logits, mask=label_masks != 0)
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=label_masks.type(torch.uint8))
            return -1.0 * log_likelihood, seq_tags
        else:
            return seq_tags

    def calculate_loss(self, input_ids, attention_masks, token_masks, segment_ids, label_ids, label_masks, feats):
            seq_output = self.roberta(input_ids=input_ids,
                                attention_mask=attention_masks,
                                token_type_ids=segment_ids)[0]
            seq_output, _ = self.lstm(seq_output)

            batch_size, max_len, feat_dim = seq_output.shape
            valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=seq_output.device)

            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                    if token_masks[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = seq_output[i][j]

            sequence_output = self.dropout(valid_output)
            logits = self.classifier(sequence_output)
            loss_function = nn.CrossEntropyLoss()

            mask = label_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[mask]
            active_labels = label_ids.view(-1)[mask]
            loss = loss_function(active_logits, active_labels)

            return loss, (active_logits, active_labels)

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
