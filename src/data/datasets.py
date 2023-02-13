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
    def __init__(self, name: str, path: str, max_doc_len, processed_dataset_path, tokenizer, **kwargs):
        super().__init__()
        # cache, usefast, prefixspace, speakers, sentence_splitting(to extract spans)
        self.max_doc_len = max_doc_len
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
        tokenized = tokenizer(example["tokens"], padding="max_length", truncation=True, add_special_tokens=True, max_length = 4096,
                              is_split_into_words=True, return_offsets_mapping=True)
        encoded["input_ids"] = tokenized["input_ids"]
        encoded["offset_mapping"] = tokenized["offset_mapping"]
        encoded["attention_mask"] = tokenized["attention_mask"]

        encoded["gold_clusters"] = [[(tokenized.word_to_tokens(start).start,
                                 tokenized.word_to_tokens(end).end - 1)
                                         for start, end in cluster if tokenized.word_to_tokens(start) is not None and tokenized.word_to_tokens(end) is not None] for cluster in example["clusters"]]
        encoded["EOS_indices"] = [tokenized.word_to_tokens(eos - 1).start for eos in example["EOS"] if tokenized.word_to_tokens(eos - 1) is not None]        
        
        #encoded["s2s_indices"] = self.s2s(
        #encoded["input_ids"], encoded["gold_clusters"])
        return encoded


    def encode_with_eos(self, example, tokenizer):
        if "clusters" not in example:
            example["clusters"] = []

        encoded = {"tokens": example["tokens"]}
        tokenized = tokenizer(example["tokens"], padding="max_length", truncation=True, add_special_tokens=True, max_length = 4096,
                              is_split_into_words=True, return_offsets_mapping=True)

        encoded["input_ids"] = tokenized["input_ids"]
        encoded["offset_mapping"] = tokenized["offset_mapping"]
        encoded["attention_mask"] = tokenized["attention_mask"]

        encoded["gold_clusters"] = [[(tokenized.word_to_tokens(start).start,
                                 tokenized.word_to_tokens(end).end - 1)
                                         for start, end in cluster if tokenized.word_to_tokens(start) is not None and tokenized.word_to_tokens(end) is not None] for cluster in example["clusters"]]


        #encoded["s2s_indices"] = self.s2s(
        #encoded["input_ids"], encoded["gold_clusters"])
        return encoded

    def __len__(self) -> int:
        return len(self.set["input_ids"])

    def __getitem__(
            self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        elem = self.set[index]
        return {"input_ids": torch.tensor(elem["input_ids"]),
                "attention_mask": torch.tensor(elem["attention_mask"]),
                "offset_mapping": torch.tensor(elem["offset_mapping"]),
                "gold_edges": self.create_edge_matrix(elem["input_ids"], elem["gold_clusters"]),
                "eos": self.eos(elem["input_ids"], elem["EOS_indices"])
                }
                #"gold_edges": self.idxs(elem["input_ids"], elem["s2s_indices"])}

    def eos(self, ids, eos_indices):
        a = np.zeros(len(ids))
        a[eos_indices]=1
        return a

    def s2s(self, ids, coreferences):
        start_bpe_to_cluster = {}
        end_bpe_to_cluster = {}
        for cluster in coreferences:
            for idx, (start_bpe_idx, end_bpe_idx) in enumerate(cluster):
                starts_without_idx = [elem[0] for elem in cluster]
                ends_without_idx = [elem[1] for elem in cluster]
                ends_without_idx.pop(idx)
                starts_without_idx.pop(idx)
                if start_bpe_idx not in start_bpe_to_cluster.keys():
                    start_bpe_to_cluster[start_bpe_idx] = []
                if end_bpe_idx not in end_bpe_to_cluster.keys():
                    end_bpe_to_cluster[end_bpe_idx] = []
                end_bpe_to_cluster[end_bpe_idx].extend(ends_without_idx)
                start_bpe_to_cluster[start_bpe_idx].extend(starts_without_idx)
        return [(start_bpe_idx, end_bpe_idx) for start_bpe_idx, target_bpe_idxs in start_bpe_to_cluster.items() for end_bpe_idx in target_bpe_idxs]

    def idxs(self, ids, idxs):
        matrix = torch.zeros(len(ids), len(ids))
        for start_bpe_idx, target_bpe_idx in idxs:
            matrix[start_bpe_idx][target_bpe_idx] = 1
        return matrix

    def create_edge_matrix(self, ids, coreferences, type="s2s"):
        matrix = np.zeros((len(ids), len(ids)))
        start_bpe_to_cluster = {}
        end_bpe_to_cluster = {}
        for cluster in coreferences:
            for idx, (start_bpe_idx, end_bpe_idx) in enumerate(cluster):
                #starts_without_idx = [list(range(elem[0], elem[1] +1)) for elem in cluster]
                #starts_without_idx = [item for sublist in starts_without_idx for item in sublist]
                starts_without_idx = [elem[0] for elem in cluster]
                ends_without_idx = [elem[1] for elem in cluster]
                ends_without_idx.pop(idx)
                starts_without_idx.pop(idx)

                if start_bpe_idx not in start_bpe_to_cluster.keys():
                    start_bpe_to_cluster[start_bpe_idx] = []
                if end_bpe_idx not in end_bpe_to_cluster.keys():
                    end_bpe_to_cluster[end_bpe_idx] = []
                end_bpe_to_cluster[end_bpe_idx].extend(ends_without_idx)
                start_bpe_to_cluster[start_bpe_idx].append(end_bpe_idx)

        for start_bpe_idx, list_of_coreferring_idxs in start_bpe_to_cluster.items():
            for target_bpe_idx in list_of_coreferring_idxs:
                matrix[start_bpe_idx][target_bpe_idx] = 1
        return matrix
        

