import torch
from itertools import chain
from typing import List, Dict, Any
from dataclasses import dataclass
@dataclass
class CustomCollatorWithPadding:
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

        error_span_mask = []
        for feature in features:
            if "error_span_mask" in feature:
                error_span_mask.append(feature['error_span_mask'][:])
                del feature['error_span_mask']

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        sequence_length = batch['input_ids'].shape[1]
        if error_span_mask:
            batch["error_span_mask"] = torch.tensor([
                list(error_span) + [0] * (sequence_length - len(error_span)) for error_span in error_span_mask
            ], dtype=torch.int64)
        return batch
