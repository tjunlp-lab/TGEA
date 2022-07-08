import torch
from itertools import chain
from typing import List, Dict, Any
from dataclasses import dataclass
@dataclass
class DataCollatorWithPadding:
    def __init__(self,
                 tokenizer=None,
                 padding=True,
                 max_length=None,
                 pad_to_multiple_of=None,
                 return_tensors='pt'):
        # transformers/data/data_collator.py
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        batch_size = len(features)
        num_choice = len(features[0]['input_ids'])
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        label = [feature.pop(label_name) for feature in features]
        flatten_choice = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choice)] for feature in features]
        flatten_choice = list(chain(*flatten_choice))

        batch = self.tokenizer.pad(
            flatten_choice,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {k: v.view(batch_size, num_choice, -1) for k, v in batch.items()}

        batch['labels'] = torch.tensor(label, dtype=torch.int64)

        return batch