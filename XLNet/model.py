from transformers import XLNetTokenizer, XLNetModel, AdamW
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


class XLNetForMultiLabelSequenceClassification(torch.nn.Module):

    def __init__(self, num_labels=5):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        last_hidden_state = self.xlnet(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        logits = logits.view(-1, self.num_labels)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):

        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):

        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

def load_model():
    model = XLNetForMultiLabelSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=Config.NUM_LABELS)
    model.to(Config.DEVICE)
    return model



