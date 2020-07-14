import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel

class MonoBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(MonoBERT, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # load tf parameters
        self.weight = nn.Parameter(torch.zeros((config.num_labels, config.hidden_size)))
        self.bias = nn.Parameter(torch.zeros((config.num_labels)))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits = torch.matmul(pooled_output, self.weight.t())
        logits = logits + self.bias

        softmax_logits = nn.Softmax(dim=-1)(logits)
        outputs = (softmax_logits,) + outputs[2:]

        if labels is not None:
            one_hot_labels = torch.zeros_like(softmax_logits).scatter_(1, labels, 1)
            per_example_loss = -torch.sum(softmax_logits*one_hot_labels, dim=-1)
            loss = torch.mean(per_example_loss)
            outputs = (loss, per_example_loss) + outputs

        return outputs  # (loss, per_example_loss), softmax_logits, (hidden_states), (attentions)