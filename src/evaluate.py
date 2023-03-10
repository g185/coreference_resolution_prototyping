import json
import hydra
import torch
from omegaconf import omegaconf

import tqdm
from data.pl_data_modules import BasePLDataModule
from models.pl_modules import BasePLModule
from utils.logging import get_console_logger
from transformers import AutoTokenizer
import argparse

logger = get_console_logger()


@torch.no_grad()
def evaluate(conf: omegaconf.DictConfig):
    device = conf.evaluation.device
    hydra.utils.log.info("Using {} as device".format(device))

    pl_data_module: BasePLDataModule = hydra.utils.instantiate(
        conf.data.datamodule, _recursive_=False
    )

    pl_data_module.prepare_data()
    pl_data_module.setup("test")

    logger.log(f"Instantiating the Model from {conf.evaluation.checkpoint}")
    model = BasePLModule.load_from_checkpoint(
        conf.evaluation.checkpoint,
        _recursive_=False,
    )

    #if conf.model.module.model.coreference_mode == "t2c":
        #evaluate_t2c_model(model, pl_data_module.test_dataloader()[0], device)
    #else:
    evaluate_topk_model(model, pl_data_module.test_dataloader()[0], device)
    return

def evaluate_topk_model(model, test_dataloader, device):
    tokenizer = test_dataloader.dataset.tokenizer
    model.to(device)
    model.eval()
    results = []
    for batch in test_dataloader:
        tokenized = tokenizer(batch["tokens"], truncation=True, add_special_tokens=True, max_length = test_dataloader.dataset.max_doc_len,
                              is_split_into_words=True, return_offsets_mapping=True)
        gold = model.unpad_gold_clusters(batch["gold_clusters"])
        gold = [[(tokenized.token_to_word(start),
                                      tokenized.token_to_word(end))
                                     for start, end in cluster] for cluster in gold]
        results.append({"doc_key": batch["doc_key"][0], "sentences": batch["tokens"], "clusters": gold})
    with open("/root/coreference_resolution_evaluation/out.jsonlines", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    return

    
def evaluate_t2c_model(model, test_dataloader, device):
    tokenizer = test_dataloader.dataset.tokenizer
    model.to(device)
    model.eval()
    output = []
    for batch in test_dataloader:
        output = model(batch)
        s2e = output["pred_dict"]["mentions"]
        t2c = output["pred_dict"]["coreferences_matrix_form"]
        sentence_clusters = (batch["tokens"], s2e_t2c_to_clusters(s2e,t2c, batch["tokens"], tokenizer))
                             
def s2e_t2c_to_clusters(s2e, t2c, tokens, tokenizer):
    #carlos
    result=[[[]]]
    tokenized = tokenizer(tokens, truncation=True, add_special_tokens=True, max_length = 4096,
                              is_split_into_words=True, return_offsets_mapping=True)
    
    result = [[(tokenized.token_to_word(start).start,
                                      tokenized.word_to_tokens(end).end - 1)
                                     for start, end in cluster if tokenized.word_to_tokens(start) is not None and tokenized.word_to_tokens(end) is not None] for cluster in example["clusters"]]
    
    encoded["EOS_indices"] = [tokenized.word_to_tokens(eos - 1).start for eos in example["EOS"] if tokenized.word_to_tokens(eos - 1) is not None]




        



@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    evaluate(conf)


if __name__ == "__main__":
    main()
