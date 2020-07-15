import math
import torch
from torch import nn

class Fusion(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(Fusion, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # batch, hidden_size
        hidden_states = self.dropout(hidden_states)
        linear_out = self.dense(hidden_states)
        output = self.activation(linear_out)
        return output


class EmbeddingProb(nn.Module):
    def __init__(self, hidden_size, max_token_num, dropout_prob):
        super(EmbeddingProb, self).__init__()
        self.fusion = Fusion(hidden_size, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifcation = nn.Linear(hidden_size, 2, bias=True)
        self.apply(self._init_weights)

    def forward(self, hidden_states, labels=None):
        # batch, tokens, hidden_size
        fusion_output = self.fusion(hidden_states)
        fusion_output = self.dropout(fusion_output)
        logits = self.classifcation(fusion_output)
        logits = torch.max(logits, dim=1)[0]
        softmax_logits = torch.nn.Softmax(dim=-1)(logits)
        output = (softmax_logits, )
        if labels is not None:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            one_hot_labels = torch.zeros_like(log_probs).scatter_(1, labels, 1)
            per_example_loss = -torch.sum(log_probs*one_hot_labels, dim=-1)
            loss = torch.mean(per_example_loss)
            output = (loss, per_example_loss, softmax_logits, )
        return output  # softmax_logits, (loss, per_example_loss), 
    
    # from https://github.com/huggingface/transformers
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
