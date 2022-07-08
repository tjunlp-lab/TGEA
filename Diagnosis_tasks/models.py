import torch
from torch import nn
from loss import focal_loss
from transformers import PreTrainedModel, BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from modeling_bert import BertModel


class BaseModel(PreTrainedModel):
    # copy from transformers/models/bert/modeling_bert.py
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class SequenceClassification(BaseModel):
    # used for benchamrk 1 and 4
    def __init__(self,
                 config,
                 num_labels,
                 use_focal=False,
                 focal_alpha=0.5,
                 ):
        super(SequenceClassification, self).__init__(config)
        self.config = config
        classifier_dropout = (self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.weights = torch.zeros(num_labels)
        self.weights[0] += focal_alpha
        self.weights[1:] += (1 - focal_alpha)  #
        self.init_weights()
        if use_focal:
            self.loss_fct = focal_loss(num_classes=num_labels, alpha=focal_alpha)
        else:
            self.loss_fct = nn.CrossEntropyLoss(self.weights)

    def forward(self, inputs, labels=None, num_choices=1):
        '''
        :param inputs: [batch_size, hidden_size]
        :param labels: [batch_size, 1]
        :return:
        '''
        pooled_output = self.dropout(inputs)
        logits = self.classifier(pooled_output)
        if num_choices != 1:
            logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            if num_choices != 1:
                loss = self.loss_fct(logits, labels)
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.squeeze())
        model_output =  SequenceClassifierOutput(
                                        loss=loss,
                                        logits=logits)

        return model_output

class TokenClassification(BaseModel):
    # used for benchamrk 2 and 3
    def __init__(self,
                 config,
                 num_labels,
                 use_focal=False,
                 focal_alpha=0.5,
                 ):
        super(TokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.config = config
        self.ignore_index= -100
        classifier_dropout = (self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.init_weights()
        self.weights = torch.zeros(num_labels)
        self.weights[0] += focal_alpha
        self.weights[1:] += (1 - focal_alpha)  #
        if use_focal:
            self.loss_fct = focal_loss(num_classes=num_labels, alpha=focal_alpha)
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, inputs, attention_mask=None, labels=None):
        """

        :param inputs: [batch_size, seq_len, hidden_state]
        :param attention_mask: [batch_size, seq_len]
        :param labels: [batch_size, 1]
        :return:
        """
        pooled_output = self.dropout(inputs)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if attention_mask != None:
                active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(self.ignore_index).type_as(labels))
            loss = self.loss_fct(active_logits, active_labels)
        model_output = TokenClassifierOutput(
                                    loss=loss,
                                    logits=logits)
        return model_output

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SeperateTaskForClassification(nn.Module):
    def __init__(self,
                 model_name,
                 input_type,
                 classification_type,
                 num_labels,
                 from_scratch=False,
                 use_focal=False,
                 use_erroneous_span=False,
                 focal_alpha=0.5):

        super(SeperateTaskForClassification, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        if from_scratch:
            from  transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.model = BertModel(config)
        else:
            self.model = BertModel.from_pretrained(model_name)

        self.use_erroneous_span = use_erroneous_span
        self.config = self.model.config
        self.input_type = input_type
        self.classification_type = classification_type
        self.pooler = Pooler(self.config)
        if classification_type == 'sequence':
            self.classifier = SequenceClassification( self.config,
                                                      num_labels,
                                                      use_focal,
                                                      focal_alpha,)
        elif classification_type == 'token':

            self.classifier = TokenClassification(self.config,
                                                  num_labels,
                                                  use_focal,
                                                  focal_alpha,
                                                  )
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                error_span_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,):
        if self.use_erroneous_span:

            token_type_ids = error_span_mask

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            error_type_ids=error_span_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.classification_type == 'sequence' and self.input_type == 'pooled_cls':
            # benchmark1
            #features = self.pooler(outputs[0][ :, 0])
            features = outputs[1]
            model_outputs = self.classifier(features, labels)
        if self.classification_type == 'sequence' and self.input_type == 'span':
            # benchmark4 error_type classification
            features = self.pooler(outputs[0])
            features = torch.mean(features * error_span_mask.unsqueeze(-1), 1)
            model_outputs = self.classifier(features, labels)
        if self.classification_type == 'token':
            features = self.pooler(outputs[0])
            model_outputs = self.classifier(features, attention_mask, labels)

        return model_outputs