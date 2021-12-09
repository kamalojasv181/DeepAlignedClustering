from util import *
from transformers import AutoModel

        
class BertForModel(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertForModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained("google/bert_uncased_L-12_H-768_A-12")
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , return_representation = 0):

        output = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)[0][:, 0, :]
        if return_representation:
            return output

        output = self.activation(output)
        output = self.dropout(output)
        output = self.classifier(output)

        return output