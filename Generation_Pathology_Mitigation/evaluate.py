from transformers import AutoTokenizer, AutoConfig, XLNetTokenizer
from models import CompareForGPT
import json
import torch
import tqdm
import jieba
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def get_data(file_name, tokenizer, max_len = 250):
    datasets = []
    with open(file_name, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            example = {}
            passes = json.loads(line)
            text = passes['masked_text'].split('<mask>')[0][-max_len:]
            #print(tokenizer.encode(text, return_tensors='tf'))
            example['text'] = text
            example['input_ids'] =  tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)
            example['target_len'] = len(example['input_ids'][0])+ len(passes['correct_word'])
            example['correct_word'] = passes['correct_word']
            example['input_len'] = len(text)
            datasets.append(example)
    f.close()
    return datasets

class XLNetTokenizer(XLNetTokenizer):
    translator = str.maketrans(" \n", "\u2582\u2583")

    def _tokenize(self, text, *args, **kwargs):
        text = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        text = " ".join(text)
        return super()._tokenize(text, *args, **kwargs)

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text


#tokenizer = XLNetTokenizer.from_pretrained('mymusise/CPM-GPT2-FP16')
#model = TFGPT2LMHeadModel.from_pretrained("mymusise/CPM-GPT2-FP16")
def generate(evaluate_file):
    pretrained_model_name = "uer/gpt2-chinese-cluecorpussmall"
    model_name = './tmp/comparision_correct'
    device = 'cuda:0'
    config = AutoConfig.from_pretrained(pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CompareForGPT(pretrained_model_name, config, tokenizer )
    model.load_state_dict(torch.load(os.path.join(model_name, 'pytorch_model.bin')))

    model.to(device)
    quality_token = '[QUALITY]'
    score_token = '[SCORE]'
    quality_token_id = tokenizer.convert_tokens_to_ids(quality_token)
    score_token_id = tokenizer.convert_tokens_to_ids(score_token)

    dataset = get_data(evaluate_file, tokenizer)
    top_1 = 0
    top_3 = 0
    total = 0
    res = []
    for d in tqdm.tqdm(dataset):
        input_ids = d['input_ids'].to(device)
        max_len = d['target_len']
        c_words = d['correct_word']
        i_len = d['input_len']
        #input_ids = input_ids[:,:4]

        beam_outputs = model.model.generate(
            input_ids,
            max_length=max_len+3,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=3,
        )
        guess_words = []
        dis_inputs = []

        for i, beam_output in enumerate(beam_outputs):
            raw_text = tokenizer.decode(beam_output, skip_special_tokens=True).replace(' ','').replace("[UNK]", " ")
            raw_text += quality_token
            text = raw_text[i_len: i_len + len(c_words)]
            guess_words.append(text)
            dis_inputs.append(raw_text)
        res.append(guess_words[0])
        dis_encode = tokenizer(dis_inputs, padding=True, return_tensors='pt', add_special_tokens=False).to(input_ids.device)
        dis_encode = {key:torch.unsqueeze(dis_encode[key], 0) for key in dis_encode.keys()}
        model_output = model(**dis_encode)
        dis_logits = model_output.logits[1]
        max_id = torch.argmax(dis_logits, -1)
        max_id = max_id.tolist()[0]

        if c_words == guess_words[max_id]:
            top_1 += 1
        if c_words in guess_words:
            top_3 += 1
            continue
        total += 1
    print(total)
    print("Top 1 accuracy is {:2}%".format(top_1 / total * 100))
    print("Top 3 accuracy is {:2}%".format(top_3 / total * 100))

if __name__ == '__main__':
    dev_file = '../data/word_prediction/dev.json'
    test_file = '../data/word_prediction/test.json'
    generate(dev_file)

