from typing import Tuple
from typing import Dict, Union
import numpy as np
import hydra.utils
import torch
from torch.utils.data import Dataset

import os
import src.common.util as util

from torch.utils.data import Dataset
from datasets import  load_from_disk
from datasets import  Dataset as dt

from transformers import AutoTokenizer

def prepare_data(set):

    return set


class OntonotesDataset(Dataset):
    def __init__(self, name: str, path: str, max_doc_len, processed_dataset_path, tokenizer, mode, **kwargs):
        super().__init__()
        # cache, usefast, prefixspace, speakers, sentence_splitting(to extract spans)
        self.max_doc_len = max_doc_len
        self.mode = mode
        try:
            self.set = load_from_disk(hydra.utils.get_original_cwd() + "/" + processed_dataset_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True, add_prefix_space=True)
            self.set = dt.from_pandas(util.to_dataframe(path[0]))
            self.set = self.prepare_data(self.set)
            self.set = self.set.map(self.encode, batched=False, fn_kwargs={"tokenizer": self.tokenizer})
            self.set = self.set.remove_columns(column_names=["speakers", "clusters"])
            if not os.path.exists(hydra.utils.get_original_cwd() + "/data/cache"):
                os.makedirs(hydra.utils.get_original_cwd() + "/data/cache")
            self.set.save_to_disk(hydra.utils.get_original_cwd() + "/" + processed_dataset_path)

    def prepare_data(self, set):
        return set.filter(lambda x: len(self.tokenizer(x["tokens"])["input_ids"]) <= self.max_doc_len)
    
    def encode(self, example, tokenizer):
        if "clusters" not in example:
            example["clusters"] = []

        encoded = {"tokens": example["tokens"]}
        tokenized = tokenizer(example["tokens"], padding="max_length", truncation=True, add_special_tokens=True, max_length = self.max_doc_len,
                              is_split_into_words=True, return_offsets_mapping=True)

        encoded["input_ids"] = tokenized["input_ids"]
        encoded["offset_mapping"] = tokenized["offset_mapping"]
        encoded["attention_mask"] = tokenized["attention_mask"]
        encoded["gold_clusters"] = [[(tokenized.word_to_tokens(start).start,
                                 tokenized.word_to_tokens(end).end - 1)
                                         for start, end in cluster if tokenized.word_to_tokens(start) is not None and tokenized.word_to_tokens(end) is not None] for cluster in example["clusters"]]
        encoded["EOS_indices"] = [tokenized.word_to_tokens(eos - 1).start for eos in example["EOS"] if tokenized.word_to_tokens(eos - 1) is not None]       

        return encoded

    def __len__(self) -> int:
        return self.set.shape[0]

    def __getitem__(
            self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        elem = self.set[index]
        return {"input_ids": torch.tensor(elem["input_ids"]),
                "attention_mask": torch.tensor(elem["attention_mask"]),
                "gold_edges": torch.tensor(self.create_gold_matrix(len(elem["input_ids"]), elem["gold_clusters"])),
                "mask": torch.tensor(self.mask(elem["input_ids"], torch.tensor(elem["offset_mapping"]), elem["EOS_indices"]))
                }

    def mask(self, ids, offset_mapping, eos_indices):
        if self.mode == "s2e_sentence_level":
            mask = torch.zeros((len(ids), len(ids)))
            prec = 0
            for idx in eos_indices:
                for i in range(prec, idx + 1):
                    for j in range(prec,idx + 1):
                        mask[i][j] = 1
                prec = idx
            mask = np.triu(mask)
        if self.mode == "s2e":
            mask = np.zeros(len(ids))
            eoi = (torch.tensor(ids)==2).nonzero(as_tuple=False)
            mask[0:eoi+1] = 1
        if self.mode == "s2s":
            mask = np.zeros(len(ids))
            idxs_start_words = (offset_mapping[:,0] == 0) & (offset_mapping[:,1] != 0)
            mask[idxs_start_words] = 1
        return mask



        



    def create_gold_matrix(self, idslen, coreferences, eos = None):
        matrix = np.zeros((idslen, idslen))

        if self.mode == "s2e" or self.mode == "s2e_sentence_level":
            for cluster in coreferences:
                for start_bpe_idx, end_bpe_idx in cluster:
                    matrix[start_bpe_idx][end_bpe_idx] = 1
                    
        elif self.mode == "s2s":
            for cluster in coreferences:
                for idx, (start_bpe_idx, end_bpe_idx) in enumerate(cluster):
                    #starts_without_idx = [list(range(elem[0], elem[1] +1)) for elem in cluster]
                    #starts_without_idx = [item for sublist in starts_without_idx for item in sublist]
                    starts_without_idx = [elem[0] for elem in cluster]
                    starts_without_idx.pop(idx)
                    for target_bpe_idx in starts_without_idx:
                        matrix[start_bpe_idx][target_bpe_idx] = 1

        
        return matrix
        

