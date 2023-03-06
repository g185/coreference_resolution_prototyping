from typing import Dict
from transformers import AutoModel, AutoConfig
import torch
from transformers.activations import ACT2FN
import numpy as np

from src.common.util import FullyConnectedLayer
from src.common.latent.models import IndependentLatentModel


class CorefModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.hf_model_name = kwargs["huggingface_model_name"]
        self.model = AutoModel.from_pretrained(self.hf_model_name)
        self.config = AutoConfig.from_pretrained(self.hf_model_name)
        self.linear = kwargs["linear_layer_hidden_size"]
        self.mention_mode = kwargs["mention_mode"]
        self.coreference_mode = kwargs["coreference_mode"]
        self.pos_weight = kwargs["pos_weight"]

        if kwargs["transformer_freeze"] == "freezed":
            for param in self.model.parameters():
                param.requires_grad = False

        if self.coreference_mode == "latent":
            self.z = IndependentLatentModel()
        
        if self.mention_mode != "gold":
            self.representation_s2e_start = FullyConnectedLayer(
                input_dim=self.config.hidden_size, hidden_size=self.linear, output_dim=self.config.hidden_size, dropout_prob=0.3)
            self.representation_s2e_end = FullyConnectedLayer(
                input_dim=self.config.hidden_size, hidden_size=self.linear, output_dim=self.config.hidden_size, dropout_prob=0.3)
        
        if self.coreference_mode == "t2c":
            self.representation_t2c_start = FullyConnectedLayer(
                input_dim=self.config.hidden_size, hidden_size=self.linear, output_dim=self.config.hidden_size, dropout_prob=0.3)
            self.representation_t2c_end = FullyConnectedLayer(
                input_dim=self.config.hidden_size, hidden_size=self.linear, output_dim=self.config.hidden_size, dropout_prob=0.3)
        
        if self.coreference_mode in ["topk", "latent"]:
            self.representation_ment_start = FullyConnectedLayer(
                input_dim=2*self.config.hidden_size, hidden_size=1536, output_dim = self.config.hidden_size, dropout_prob=0.3)
            self.repr_ment_end = FullyConnectedLayer(
                input_dim=2*self.config.hidden_size, hidden_size=1536, output_dim = self.config.hidden_size, dropout_prob=0.3)
        
        


    def forward(
            self,
            batch: torch.Tensor,
            step: int,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(batch)
    
    def forward(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        lhs = last_hidden_states

        preds = {}
        golds = {}

        loss = torch.tensor([0.0], requires_grad=True).cuda()
        loss_dict={}

        if self.mention_mode != "gold":
            gold_mentions = batch["gold_mentions"]

            mention_logits = self.representation_s2e_start(lhs) @ self.representation_s2e_end(lhs).permute(0,2,1) 
            mask = batch["mentions_mask"].detach()
            
            mention_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    mention_logits[mask == 1], gold_mentions[mask == 1], pos_weight=torch.tensor(self.pos_weight)) 
            
            loss = loss + mention_loss
            loss_dict["mention_loss"] = mention_loss

            preds["mentions"] = torch.sigmoid(mention_logits[mask == 1].detach())
            golds["mentions"] = gold_mentions[mask == 1].detach()
        else:
            mentions_gold = batch["gold_mentions"]

            mention_start_idxs = ((mentions_gold==1).nonzero(as_tuple=False)[:,1]).unsqueeze(0)
            mention_end_idxs = ((mentions_gold==1).nonzero(as_tuple=False)[:,2]).unsqueeze(0)

            topk_start_coref_reps = torch.index_select(lhs, 1, mention_start_idxs.squeeze(0))
            topk_end_coref_reps = torch.index_select(lhs, 1, mention_end_idxs.squeeze(0))

        if self.coreference_mode != "gold":
            if self.coreference_mode == "t2c":
                coref_logits = self.representation_t2c_start(lhs) @ self.representation_t2c_end(lhs).permute(0,2,1) 
                labels = batch["gold_clusters"]
                coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    coref_logits, batch["gold_clusters"])
            else:

                if self.mention_mode != "gold":
                    mention_start_idxs, mention_end_idxs, span_mask, topk_mention_logits = self._prune_topk_mentions(mention_logits, batch["attention_mask"])
                    topk_start_coref_reps = torch.index_select(lhs, 1, mention_start_idxs.squeeze())
                    topk_end_coref_reps = torch.index_select(lhs, 1, mention_end_idxs.squeeze())

                b = torch.cat((topk_start_coref_reps, topk_end_coref_reps), dim=2)

                coref_logits = self.representation_ment_start(b) @ self.repr_ment_end(b).permute([0,2,1]) 
        
                coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits])
                labels = self._get_cluster_labels_after_pruning(mention_start_idxs, mention_end_idxs, batch["gold_clusters"])

                #span_starts = (torch.round(torch.sigmoid(coref_logits)) == 1).nonzero(as_tuple=False)[:,1]
                doc, m2a = create_mention_to_antecedent(mention_start_idxs, mention_end_idxs, coref_logits)
                preds["coreferences"] = create_clusters(m2a)
                golds["coreferences"] = batch["gold_clusters"]

            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits, labels)
            
            #coreference_loss = (coreference_loss / (2000 / (step + 1))) if step < 2000 else coreference_loss 
            loss = loss + coreference_loss
            loss_dict["coreference_loss"] = coreference_loss

            preds["coreferences_matrix_form"] = torch.sigmoid(coref_logits.flatten().detach())
            golds["coreferences_matrix_form"] = labels.flatten().detach()

        loss_dict["full_loss"] = loss 
    
        output = {  "pred_dict": preds,
                    "gold_dict": golds,
                    "loss_dict": loss_dict,
                    "loss": loss
                    }
        return output
    
    def forward_as_BCE_classification_partial(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        mask = batch["mask"]
        lhs = last_hidden_states
        gold = batch["gold_mentions"]

        coref_logits = self.representation_start(
                    lhs) @ self.representation_end(lhs).permute(0,2,1) 

        coref_logits = coref_logits[mask==1]
        gold = gold[mask == 1]
        pred = torch.sigmoid(coref_logits.detach()) 
        gold = gold.detach()
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits, gold, pos_weight=torch.tensor(self.pos_weight))
        
        output = {"pred": pred,
                    "gold": gold,
                    "references": ((mask.detach()==1).nonzero(as_tuple=False)).detach() if self.mode!="s2s" else None,
                    "loss": loss}
        return output

    def forward_as_BCE_classification(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        lhs = last_hidden_states
        mentions_gold = batch["gold_mentions"]

        start_coref_reps = self.representation_start(lhs)
        end_coref_reps = self.representation_end(lhs)
        
        mention_logits =  start_coref_reps @ end_coref_reps.permute([0,2,1]) 

        mention_logits = mention_logits 
        mentions_gold = mentions_gold 
        
        mention_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                mention_logits, mentions_gold, pos_weight=torch.tensor(self.pos_weight))
        
        mention_start_idxs, mention_end_idxs, span_mask, topk_mention_logits = self._prune_topk_mentions(mention_logits, batch["attention_mask"])

        topk_start_coref_reps = torch.index_select(lhs, 1, mention_start_idxs.squeeze())
        topk_end_coref_reps = torch.index_select(lhs, 1, mention_end_idxs.squeeze())

        b = torch.cat((topk_start_coref_reps, topk_end_coref_reps), dim=2)

        coref_logits = self.representation_ment_start(b) @ self.repr_ment_end(b).permute([0,2,1]) 
        
        coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits])
        labels = self._get_cluster_labels_after_pruning(mention_start_idxs, mention_end_idxs, batch["gold_clusters"])

        coref_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            coref_logits, labels)
        output = {  "pred": torch.sigmoid(coref_logits.flatten().detach()),
                    "gold": labels.flatten().detach(),
                    "loss":  coref_loss + mention_loss
                    }
        return output
    
    def coref(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        lhs = last_hidden_states
        
        mentions_gold = batch["gold_mentions"]

        mention_start_idxs=((mentions_gold==1).nonzero(as_tuple=False)[:,1])
        mention_end_idxs=((mentions_gold==1).nonzero(as_tuple=False)[:,2])

        start_coref_reps = torch.index_select(lhs, 1, mention_start_idxs)
        end_coref_reps = torch.index_select(lhs, 1, mention_end_idxs)

        b = torch.cat((start_coref_reps, end_coref_reps), dim=2)

        coref_logits = self.representation_ment_start(b) @ self.repr_ment_end(b).permute([0,2,1])
        
        coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits])
        labels = self._get_cluster_labels_after_pruning(mention_start_idxs.unsqueeze(0), mention_end_idxs.unsqueeze(0), batch["gold_clusters"])

        coref_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            coref_logits, labels)
        
        output = {  "pred": torch.sigmoid(coref_logits.flatten().detach()),
                    "gold": labels.flatten().detach(),
                    "loss":  coref_loss
                    }
        return output


    def forward_as_BCE_classification_s2s(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        mask = batch["mask"]
        lhs = last_hidden_states
        gold = batch["gold_edges"]

        coref_logits = self.representation_start(
                    lhs) @ self.representation_end(lhs).permute(0,2,1) 

        coref_logits = coref_logits[mask==1]
        gold = gold[mask == 1]
        pred = torch.sigmoid(coref_logits.flatten().detach()) 
        gold = gold.flatten().detach()
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits, gold, pos_weight=torch.tensor(self.pos_weight))
        
        output = {"pred": pred,
                    "gold": gold,
                    "loss": loss}
        
        return output


    def forward_as_BCE_classification_s2e(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        mask = batch["mask"]
        lhs = last_hidden_states
        gold = batch["gold_edges"]

        coref_logits = self.representation_start(
                    lhs) @ self.representation_end(lhs).permute(0,2,1) 

        coref_logits = coref_logits[mask==1]
        gold = gold[mask == 1]
        pred = torch.sigmoid(coref_logits.flatten().detach()) 
        gold = gold.flatten().detach()
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits, gold, pos_weight=torch.tensor(self.pos_weight))

        output = {"pred": pred,
                    "gold": gold,
                    "loss": loss}
        return output


    def forward_as_BCE_classification_s2e_sentence_level(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        loss = []
        preds = []
        golds = []
        references = []
        for  lhs, ids, mask, gold in zip(last_hidden_states, batch["input_ids"], batch["mask"], batch["gold_edges"]):

            eoi = (ids == 2).nonzero(as_tuple=False)
            lhs = lhs[:eoi+1]
            gold = gold[:eoi+1, :eoi+1]

            
            mask = mask[:eoi+1, :eoi+1]
            coref_logits = self.representation_start(
                    lhs) @ self.representation_end(lhs).T
            coref_logits = coref_logits[mask==1]
            gold = gold[mask==1]
            references.append((mask==1).nonzero(as_tuple=False).detach())
            
            coref_logits = coref_logits.flatten()
            preds.append(torch.sigmoid(coref_logits.detach()))  # S*S
            golds.append(gold.flatten().detach())  # S*S

            loss.append(torch.nn.functional.binary_cross_entropy_with_logits(
                    coref_logits, gold.flatten(), pos_weight=torch.tensor(self.pos_weight)))
        loss = torch.stack(loss).sum()
        output = {"pred": torch.cat(preds, 0) if len(preds) > 1 else preds[0],
                  "gold": torch.cat(golds, 0) if len(golds) > 1 else golds[0],
                  "references": torch.cat(references, 0)  if len(references) > 1 else references[0],
                  "loss": loss}
        
        return output

    def forward_as_BCE_classification_s2s_iterative(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        loss = []
        preds = []
        golds = []
        for lhs, mask, gold in zip(last_hidden_states, batch["mask"], batch["gold_edges"]):
            lhs = lhs[mask == 1] #MS * HS
            gold = gold[mask == 1][:, mask == 1] #MSx MS

            coref_logits = self.representation_start(
                    lhs) @ self.representation_end(lhs).T #MS x MS 

            coref_logits = coref_logits.fill_diagonal_(0)
            pred = torch.sigmoid(coref_logits.flatten().detach()) #MS * MS
            gold = gold.flatten().detach()

            preds.append(pred)  # S*S
            golds.append(gold.flatten().detach())  # S*S

            loss.append(torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits.flatten(), gold.flatten(), pos_weight=torch.tensor(self.pos_weight)))
        loss = torch.stack(loss).sum()
        output = {"pred": torch.cat(preds, 0) if len(preds) > 1 else preds[0],
                    "gold": torch.cat(golds, 0) if len(golds) > 1 else golds[0],
                    "loss": loss}

        return output
    
    def forward_as_BCE_classification_s2e_iterative(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        loss = []
        preds = []
        golds = []
        references = []
        for  lhs, ids, mask,  gold in zip(last_hidden_states, batch["input_ids"],  batch["mask"], batch["gold_edges"]):

            eoi = (ids == 2).nonzero(as_tuple=False)
            lhs = lhs[:eoi+1]
            gold = gold[:eoi+1, :eoi+1]

            coref_logits = self.representation_start(
                lhs) @ self.representation_end(lhs).T
            
            coref_logits = coref_logits.flatten()
            preds.append(torch.sigmoid(coref_logits.detach()))  # S*S
            golds.append(gold.flatten().detach())  # S*S
            loss.append(torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits, gold.flatten(), pos_weight=torch.tensor(self.pos_weight)))
        loss = torch.stack(loss).sum()

        output = {"pred": torch.cat(preds, 0) if len(preds) > 1 else preds[0],
                  "gold": torch.cat(golds, 0) if len(golds) > 1 else golds[0],
                  "loss": loss}
        return output
    
    def forward_as_BCE_classification_s2e_sentence_level_iterative(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        loss = []
        preds = []
        golds = []
        references = []
        for  lhs, ids, mask, gold in zip(last_hidden_states, batch["input_ids"], batch["mask"], batch["gold_edges"]):

            eoi = (ids == 2).nonzero(as_tuple=False)
            lhs = lhs[:eoi+1]
            gold = gold[:eoi+1, :eoi+1]

            
            mask = mask[:eoi+1, :eoi+1]
            coref_logits = self.representation_start(
                    lhs) @ self.representation_end(lhs).T
            coref_logits = coref_logits[mask==1]
            gold = gold[mask==1]
            references.append((mask==1).nonzero(as_tuple=False).detach())
            
            coref_logits = coref_logits.flatten()
            preds.append(torch.sigmoid(coref_logits.detach()))  # S*S
            golds.append(gold.flatten().detach())  # S*S

            loss.append(torch.nn.functional.binary_cross_entropy_with_logits(
                    coref_logits, gold.flatten(), pos_weight=torch.tensor(self.pos_weight)))
        loss = torch.stack(loss).sum()
        output = {"pred": torch.cat(preds, 0) if len(preds) > 1 else preds[0],
                  "gold": torch.cat(golds, 0) if len(golds) > 1 else golds[0],
                  "references": torch.cat(references, 0)  if len(references) > 1 else references[0],
                  "loss": loss}
        
        return output
    

    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        idx = torch.arange(max_k, device=self.model.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return (idx < len_expanded).int()

    def _prune_topk_mentions(self, mention_logits, attention_mask):
        """
        :param mention_logits: Shape [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        :param top_lambda:
        :return:
        """
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * 0.4).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]

        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]

        # drop the invalid indices and set them to the last index
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * ((seq_length ** 2) - 1)  # We take different k for each example

        # sorting for coref mention order
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        # gives the row index in 2D matrix
        topk_mention_start_ids = torch.div(sorted_topk_1d_indices, seq_length, rounding_mode='floor') # [batch_size, max_k]
        topk_mention_end_ids = sorted_topk_1d_indices % seq_length  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                             topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]

        # this is antecedents scores - rows mentions, cols coref mentions
        topk_mention_logits = topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]

        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_logits
    
    def _mask_antecedent_logits(self, antecedent_logits, span_mask):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)  # [batch_size, k, k]
        antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [batch_size, k, k]
        antecedent_logits = mask_tensor(antecedent_logits, antecedents_mask)
        return antecedent_logits
    
    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k), device='cpu')
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)):
            gold_clusters = extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)] :
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.model.device)
        return new_cluster_labels

def extract_clusters(gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in cluster if (-1) not in m) for cluster in gold_clusters]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters

def mask_tensor(t, mask):
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch.clamp(t, min=-10000.0, max=10000.0)
    return t

def extract_mentions_to_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    return mention_to_gold

def create_mention_to_antecedent(span_starts, span_ends, coref_logits):
    bs, n_spans, _ = coref_logits.shape

    no_ant =  1 - torch.sum(torch.round(torch.sigmoid(coref_logits)), dim=-1).bool().float()
    coref_logits = torch.cat((coref_logits, no_ant.unsqueeze(-1)), dim=-1)  # [batch_size, max_k, max_k + 1]

    span_starts = span_starts.detach().cpu()
    span_ends = span_ends.detach().cpu()
    max_antecedents = coref_logits.argmax(axis=-1).detach().cpu()
    doc_indices = np.nonzero(max_antecedents < n_spans)[:,0]
    mention_indices = np.nonzero(max_antecedents < n_spans)[:,1]       # indices where antecedent is not null.
    antecedent_indices = max_antecedents[max_antecedents < n_spans]
    span_indices = np.stack([span_starts.detach().cpu(), span_ends.detach().cpu()], axis=-1)

    mentions = span_indices[doc_indices, mention_indices]
    antecedents = span_indices[doc_indices, antecedent_indices]
    mention_to_antecedent = np.stack([mentions, antecedents], axis=1)
    if len(mentions.shape) == 1:
        mention_to_antecedent = [mention_to_antecedent]

    return doc_indices, mention_to_antecedent


def create_clusters(m2a):
    # Note: mention_to_antecedent is a numpy array

    clusters, mention_to_cluster = [], {}
    for mention, antecedent in m2a:
        mention, antecedent = tuple(mention), tuple(antecedent)
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            if mention not in clusters[cluster_idx]:
                clusters[cluster_idx].append(mention)
                mention_to_cluster[mention] = cluster_idx
        elif mention in mention_to_cluster:
            cluster_idx = mention_to_cluster[mention]
            if antecedent not in clusters[cluster_idx]:
                clusters[cluster_idx].append(antecedent)
                mention_to_cluster[antecedent] = cluster_idx
        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])

    clusters = [tuple(cluster) for cluster in clusters]
    return clusters