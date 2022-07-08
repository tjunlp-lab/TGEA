import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput

class CompareForGPT(nn.Module):
    def __init__(self,
                 model_name_or_path,
                 config,
                 tokenizer,
                 cache_dir=None,
                 model_revision= 'main',
                 quality_token='[QUALITY]',
                 score_token='[SCORE]'):

        super(CompareForGPT, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
            revision=model_revision)
        print("loading from pre-trained gpt")
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.quality_token = quality_token
        self.score_token = score_token
        self.quality_token_id = self.tokenizer.convert_tokens_to_ids(self.quality_token)
        self.score_token_id = self.tokenizer.convert_tokens_to_ids(self.score_token)
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,):

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.logits
        ## For Discriminator loss
        mask_pos = (input_ids == self.quality_token_id).nonzero().reshape((input_ids.shape[0], -1, 2))[:,:,1]
        prediction_mask_score = sequence_output[torch.arange(sequence_output.size(0)), mask_pos.squeeze()]
        dis_logits = prediction_mask_score[:, self.score_token_id].unsqueeze(-1)
        dis_logits = dis_logits.view(-1, num_choices)

        # For generative loss
        # [t1, t2, t3,...tn, [QUALITY]]
        total_loss = None
        if labels is not None:
            dis_loss = self.loss_fct(dis_logits, labels)

            shift_logits = sequence_output[:, :-1,:].contiguous() # [t1,t2.....tn]
            shifit_labels = input_ids[:, 1:].contiguous()# [t2, t3.....[QUALITY]]
            label_mask = attention_mask[:,1:].contiguous()


            gen_loss = self.loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shifit_labels.view(-1))
            # get average loss of each token
            gen_loss = torch.mean(label_mask * gen_loss.view(input_ids.shape[0], -1), 1).view(-1, num_choices)
            # sum up two sentence loss

            gen_loss = torch.sum(gen_loss, -1)
            # sum up generative and discriminative loss
            total_loss = torch.mean(gen_loss + dis_loss)
        model_outputs = CausalLMOutput(loss=total_loss, logits=(sequence_output, dis_logits))

        return model_outputs