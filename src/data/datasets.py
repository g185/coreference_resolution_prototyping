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

NULL_ID_FOR_COREF = -1

def prepare_data(set):

    return set


class OntonotesDataset(Dataset):
    def __init__(self, name: str, path: str, max_doc_len, processed_dataset_path, tokenizer, mode, **kwargs):
        super().__init__()
        # cache, usefast, prefixspace, speakers, sentence_splitting(to extract spans)
        self.max_doc_len = max_doc_len
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True, add_prefix_space=True)
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
        tokenized = tokenizer(example["tokens"], truncation=True, add_special_tokens=True, max_length = self.max_doc_len,
                              is_split_into_words=True, return_offsets_mapping=True)

        encoded["input_ids"] = tokenized["input_ids"]
        encoded["offset_mapping"] = tokenized["offset_mapping"]
        encoded["attention_mask"] = tokenized["attention_mask"]
        encoded["gold_clusters"] = [[(tokenized.word_to_tokens(start).start,
                                      tokenized.word_to_tokens(end).end - 1)
                                     for start, end in cluster if tokenized.word_to_tokens(start) is not None and tokenized.word_to_tokens(end) is not None] for cluster in example["clusters"]]
        encoded["EOS_indices"] = [tokenized.word_to_tokens(
            eos - 1).start for eos in example["EOS"] if tokenized.word_to_tokens(eos - 1) is not None]
        encoded["num_clusters"] = len(encoded["gold_clusters"])
        encoded["max_cluster_size"] = max(
            len(c) for c in encoded["gold_clusters"]) if encoded["gold_clusters"] else 0
        return encoded

    def __len__(self) -> int:
        return self.set.shape[0]

    def __getitem__(
            self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.set[index]

    def mask(self, ids_list, offset_mapping_list, attention_mask_list, eos_indices_list):
        result = []
        for ids, om, am, eos in zip(ids_list, offset_mapping_list, attention_mask_list, eos_indices_list):
            if self.mode == "s2e_sentence_level":
                mask = torch.zeros(len(ids), len(ids))
                prec = 0
                for idx in eos:
                    for i in range(prec, idx + 1):
                        for j in range(prec,idx + 1):
                            mask[i][j] = 1
                    prec = idx
                mask = mask.triu()
            if self.mode == "s2e":
                mask = torch.zeros(len(ids))
                eoi = (torch.tensor(am)==0).nonzero(as_tuple=False)[0]
                mask[:eoi + 1] = 1
                mask = mask.unsqueeze(0).T @ mask.unsqueeze(0)
            if self.mode == "s2s":
                mask = torch.zeros(len(ids))
                idxs_start_words = (om[:,0] == 0) & (om[:,1] != 0)
                mask[idxs_start_words] = 1
                mask = mask.unsqueeze(0).T @ mask.unsqueeze(0)
                mask = mask.fill_diagonal_(0)
            result.append(mask)
        return torch.stack(result)

    def create_gold_matrix(self, shape, coreferences, eos = None):
        result = []
        for batch_idx in range(0, shape[0]):
            matrix = torch.zeros((shape[1], shape[1]))

            if self.mode == "s2e" or self.mode == "s2e_sentence_level":
                for cluster in coreferences[batch_idx]:
                    for start_bpe_idx, end_bpe_idx in cluster:
                        matrix[start_bpe_idx][end_bpe_idx] = 1
                        
            elif self.mode == "s2s":
                for cluster in coreferences[batch_idx]:
                    for idx, (start_bpe_idx, end_bpe_idx) in enumerate(cluster):
                        #starts_without_idx = [list(range(elem[0], elem[1] +1)) for elem in cluster]
                        #starts_without_idx = [item for sublist in starts_without_idx for item in sublist]
                        starts_without_idx = [elem[0] for elem in cluster]
                        starts_without_idx.pop(idx)
                        for target_bpe_idx in starts_without_idx:
                            matrix[start_bpe_idx][target_bpe_idx] = 1
            result.append(matrix)
        return torch.stack(result)
    
    def collate_fn(self, batch):
        batch = self.tokenizer.pad(batch)
        
        max_num_clusters, max_max_cluster_size = max(batch["num_clusters"]), max(batch["max_cluster_size"])
        if max_num_clusters == 0:
            padded_clusters = None
        else:
            padded_clusters = [pad_clusters(cluster, max_num_clusters, max_max_cluster_size) for cluster in batch["gold_clusters"]]

        input_ids = torch.tensor(batch["input_ids"])
        output = {"input_ids": input_ids,
                "attention_mask": torch.tensor(batch["attention_mask"]),
                "gold_clusters": torch.tensor(padded_clusters),
                "gold_mentions": self.create_gold_matrix(input_ids.shape, batch["gold_clusters"]),
                "mask": self.mask(batch["input_ids"], batch["offset_mapping"], batch["attention_mask"], batch["EOS_indices"])
                }
        return output

def pad_clusters_inside(clusters, max_cluster_size):
    return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (max_cluster_size - len(cluster)) for cluster
            in clusters]


def pad_clusters_outside(clusters, max_num_clusters):
    return clusters + [[]] * (max_num_clusters - len(clusters))


def pad_clusters(clusters, max_num_clusters, max_cluster_size):
    clusters = pad_clusters_outside(clusters, max_num_clusters)
    clusters = pad_clusters_inside(clusters, max_cluster_size)
    return clusters