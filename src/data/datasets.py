from pathlib import Path
from typing import Any, Tuple
from typing import Dict, Iterator, List, Union

import hydra.utils
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

import numpy as np
import os
import logging
import hashlib
import src.common.util
from collections import defaultdict

import datasets
from torch.utils.data import Dataset
from datasets import  DatasetDict, load_from_disk
from datasets import  Dataset as dt
from tqdm import tqdm

from transformers import AutoTokenizer


def prepare_data(set):

    return set


class OntonotesDataset(Dataset):
    def __init__(self, name: str, path: str, max_doc_len, processed_dataset_path, **kwargs):
        super().__init__()
        # cache, usefast, prefixspace, speakers, sentence_splitting(to extract spans)
        self.max_doc_len = max_doc_len
        try:
            self.set = load_from_disk(hydra.utils.get_original_cwd() + "/" + processed_dataset_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True, add_prefix_space=True)
            self.set = dt.from_pandas(src.common.util.to_dataframe(path[0]))
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
        try:
            encoded["gold_clusters"] = [[(tokenized.word_to_tokens(start).start,
                                 tokenized.word_to_tokens(end).end - 1)
                                         for start, end in cluster if tokenized.word_to_tokens(start) is not None and tokenized.word_to_tokens(end) is not None] for cluster in example["clusters"]]
        except:
            a = 1
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
                "gold_edges": self.create_edge_matrix(elem["input_ids"], elem["gold_clusters"])}

    def create_edge_matrix(self, ids, coreferences, type="s2s"):
        matrix = torch.zeros(len(ids), len(ids))
        start_bpe_to_cluster = {}
        end_bpe_to_cluster = {}
        for cluster in coreferences:
            for idx, (start_bpe_idx, end_bpe_idx) in enumerate(cluster):
                starts_without_idx = [list(range(elem[0], elem[1] +1)) for elem in cluster]
                starts_without_idx = [item for sublist in starts_without_idx for item in sublist]
                #starts_without_idx = [elem[0] for elem in cluster]
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


logger = logging.getLogger(__name__)


def _tokenize(tokenizer, tokens, clusters, speakers):
    token_to_new_token_map = []
    new_token_map = []
    new_tokens = []
    last_speaker = None

    for idx, (token, speaker) in enumerate(zip(tokens, speakers)):
        if last_speaker != speaker:
            new_token_map += [None, None, None]
            last_speaker = speaker
        token_to_new_token_map.append(len(new_tokens))
        new_token_map.append(idx)
        new_tokens.append(token)

    for cluster in clusters:
        for start, end in cluster:
            assert tokens[start:end + 1] == new_tokens[token_to_new_token_map[start]:token_to_new_token_map[end] + 1]

    encoded_text = tokenizer(new_tokens, add_special_tokens=True, is_split_into_words=True)

    new_clusters = [[(encoded_text.word_to_tokens(token_to_new_token_map[start]).start,
                      encoded_text.word_to_tokens(token_to_new_token_map[end]).end - 1)
                     for start, end in cluster] for cluster in clusters]

    return {'tokens': tokens,
            'input_ids': encoded_text['input_ids'],
            'gold_clusters': new_clusters,
            'subtoken_map': encoded_text.word_ids(),
            'new_token_map': new_token_map
            }


def encode(example, tokenizer):
    if 'clusters' not in example:
        example['clusters'] = []
    encoded_example = _tokenize(tokenizer, example['tokens'], example['clusters'], example['speakers'])

    gold_clusters = encoded_example['gold_clusters']
    encoded_example['num_clusters'] = len(gold_clusters) if gold_clusters else 0
    encoded_example['max_cluster_size'] = max(len(c) for c in gold_clusters) if gold_clusters else 0
    encoded_example['length'] = len(encoded_example['input_ids'])

    return encoded_example


def create(tokenizer, train_file=None, dev_file=None, test_file=None, cache_dir='cache'):
    if train_file is None and dev_file is None and test_file is None:
        raise Exception(f'Provide at least train/dev/test file to create the dataset')

    dataset_files = {'train': train_file, 'dev': dev_file, 'test': test_file}

    cache_key = hashlib.md5(str.encode(str(tuple((k, v) for k, v in dataset_files.items())))).hexdigest()
    dataset_path = os.path.join(cache_dir, cache_key)

    try:
        dataset = datasets.load_from_disk(dataset_path)
        logger.info(f'Dataset restored from: {dataset_path}')
    except FileNotFoundError:
        logger.info(f'Creating dataset for {dataset_files}')

        dataset_dict = {}
        for split, path in dataset_files.items():
            if path is not None:
                df = src.common.util.to_dataframe(path)
                dataset_dict[split] = Dataset.from_pandas(df)

        dataset = DatasetDict(dataset_dict)
        logger.info(f'Tokenize documents...')
        dataset = dataset.map(encode, batched=False, fn_kwargs={'tokenizer': tokenizer})
        dataset = dataset.remove_columns(column_names=['speakers', 'clusters'])

        logger.info(f'Saving dataset to {dataset_path}')
        dataset.save_to_disk(dataset_path)

    return dataset, dataset_files


def create_batches(sampler, path_to_save=None):
    logger.info(f'Creating batches for {len(sampler.dataset)} examples...')

    # huggingface dataset cannot save tensors. so we will save lists and on train loop transform to tensors.
    batches_dict = defaultdict(lambda: [])

    for i, batch in enumerate(tqdm(sampler)):
        for k, v in batch.items():
            batches_dict[k].append(v)

    batches = Dataset.from_dict(batches_dict)
    logger.info(f'{len(batches)} batches created.')

    if path_to_save is not None:
        batches.save_to_disk(path_to_save)
        logger.info(f'Saving batches to {path_to_save}')

    return batches


class GenerativeDataset(IterableDataset):
    def __init__(
            self,
            name: str,
            path: Union[str, Path, List[str], List[Path]],
            max_tokens_per_batch: int = 800,
            drop_last_batch: bool = False,
            shuffle: bool = False,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.max_tokens_per_batch = max_tokens_per_batch
        self.drop_last_batch = drop_last_batch
        self.shuffle = shuffle
        self.data = self.load(path)
        self.n_batches = sum([1 for _ in self])

    def __repr__(self) -> str:
        return f"Dataset({self.name=}, {self.path=})"

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        batch = []
        ct = 0
        for sample in self.data:
            # number of tokens in the sample
            sample_tokens = len(sample)
            if (
                    max(ct, sample_tokens) * (len(batch) + 1) > self.max_tokens_per_batch
                    and len(batch) > 0
            ):
                yield self.prepare_output_batch(batch)
                batch = []
                ct = 0
            batch.append(sample)
            ct = max(ct, sample_tokens)
        # drop last cause might be too short and result in issues (nan if we are using amp)
        if not self.drop_last_batch and len(batch) > 0:
            yield self.prepare_output_batch(batch)

    def prepare_output_batch(self, batch: Any) -> Any:
        # Use this as `collate_fn`
        raise NotImplementedError

    def load(self, paths: Union[str, Path, List[str], List[Path]]) -> Any:
        # load data from single or multiple paths in one single dataset
        # it may be useful to shuffle the dataset here if this is a train dataset:
        # if self.shuffle:
        #   random.shuffle(data)
        raise NotImplementedError
