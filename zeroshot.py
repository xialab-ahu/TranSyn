"""
@description: Calculate mutation KL divergence scores via forward logits (unified bn/pt evaluation)
"""

import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_kl(ref_logits, alt_logits, reduction="sum"):
    """Calculate KL divergence"""
    ref_logprob = F.log_softmax(ref_logits, dim=-1)
    alt_logprob = F.log_softmax(alt_logits, dim=-1)

    ref_prob = ref_logprob.exp()
    kl = ref_prob * (ref_logprob - alt_logprob)
    if reduction == "sum":
        return kl.sum().item()
    elif reduction == "mean":
        return kl.mean().item()
    else:
        return kl

def get_loglikelihood(logits, token_id):
    """Calculate log likelihood"""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[token_id].item()

def load_vocab(vocab_path):
    """Load vocabulary"""
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    token_to_id = {k: int(v) for k, v in vocab.items()}
    return token_to_id

def extract_token_sequence(meta, model_name):
    """Extract token sequences before and after mutation"""
    if model_name == "CaLM":
        return meta["codon_ref"][63:66].replace('T', 'U'), meta["codon_alt"][63:66].replace('T', 'U')
    elif model_name == "ERNIE-RNA":
        return meta["Ref_Sequence"][64].replace('T', 'U'), meta["Alt_Sequence"][64].replace('T', 'U')
    elif model_name == "SpliceBERT":
        return meta["Ref_Sequence"][64], meta["Alt_Sequence"][64]
    elif model_name == "NT":
        return meta["Ref_Sequence"][60:66], meta["Alt_Sequence"][60:66]
    elif model_name == "RiNALMo":
        return meta["codon_ref"][64], meta["codon_alt"][64]
    else:
        raise ValueError(model_name)

def collect_ll_scores(
    data,
    df_meta,
    ref_feature,
    alt_feature,
    token_to_id,
    model_name
):
    """Collect log likelihood scores"""
    meta_map = df_meta.set_index("Variant38").to_dict("index")
    results = []
    for vid in data:
        if vid not in meta_map:
            print(f"[Warning] {vid} not in df_meta")
            continue
        meta = meta_map[vid]
        ref_emb = data[vid][ref_feature]   # [T, V]
        alt_emb = data[vid][alt_feature]   # [T, V]
        center = ref_emb.shape[0] // 2
        ref_logits = ref_emb[center]
        alt_logits = alt_emb[center]
        # Extract sequence information from meta
        ref_token, alt_token = extract_token_sequence(meta, model_name)
        ref_id = token_to_id.get(ref_token, None)
        alt_id = token_to_id.get(alt_token, None)
        if ref_id is None or alt_id is None:
            print(f"[Warning] missing token: {ref_token} / {alt_token}")
            continue
        ll_ref = get_loglikelihood(ref_logits, ref_id)
        ll_alt = get_loglikelihood(alt_logits, alt_id)
        score = ll_alt - ll_ref
        # if model_name == 'NT':
        #     print(vid)
        #     print(f"Ref token: {ref_token} | Alt token: {alt_token}")
        #     print(f"Ref id: {ref_id} | Alt id: {alt_id}")
        #     print(f"ll_ref: {ll_ref} | ll_alt: {ll_alt} | score: {score}")
        results.append((vid, score))
    return results

def calculate_ll_metrics_all(bn_data, pt_data, ref_feature, alt_feature,
                             token_to_id, model_name, save_dir):
    """Calculate metrics based on log likelihood"""
    all_vids = []
    all_scores = []
    all_labels = []

    # BN = 0
    bn_results = collect_ll_scores(bn_data, df_meta, ref_feature, alt_feature, token_to_id, model_name)
    for vid, score in bn_results:
        all_vids.append(vid)
        all_scores.append(score)
        all_labels.append(0)

    # PT = 1
    pt_results = collect_ll_scores(pt_data,  df_meta, ref_feature, alt_feature, token_to_id, model_name)
    for vid, score in pt_results:
        all_vids.append(vid)
        all_scores.append(score)
        all_labels.append(1)

    auc = roc_auc_score(all_labels, all_scores)
    aupr = average_precision_score(all_labels, all_scores)

    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame({
        "Variant38": all_vids,
        "label": all_labels,
        "score": all_scores
    }).to_csv(f"{save_dir}/ll_pred.txt", sep="\t", index=False)

    pd.DataFrame({
        "AUC": [auc],
        "AUPR": [aupr]
    }).to_csv(f"{save_dir}/ll_metrics.txt", sep="\t", index=False)

    return auc, aupr

# ++++++++++++++++++++++++++ KL Divergence Calculation ++++++++++++++++++++++++++
def collect_scores(data, ref_feature, alt_feature):
    """Batch calculate KL divergence"""
    results = []
    for vid in data:

        ref_emb = data[vid][ref_feature]   # [T, V]
        alt_emb = data[vid][alt_feature]   # [T, V]

        seq_len = ref_emb.shape[0]
        center = seq_len // 2

        ref_logits = ref_emb[center]
        alt_logits = alt_emb[center]

        score = calculate_kl(ref_logits, alt_logits)
        results.append((vid, score))
    return results


def calculate_metrics_all(bn_data, pt_data, ref_feature, alt_feature, save_dir):
    """Calculate evaluation metrics"""

    all_vids = []
    all_scores = []
    all_labels = []

    # ===== BN =====
    bn_results = collect_scores(bn_data, ref_feature, alt_feature)
    for vid, score in bn_results:
        all_vids.append(vid)
        all_scores.append(score)
        all_labels.append(0)

    # ===== PT =====
    pt_results = collect_scores(pt_data, ref_feature, alt_feature)
    for vid, score in pt_results:
        all_vids.append(vid)
        all_scores.append(score)
        all_labels.append(1)

    # ===== metrics =====
    auc = roc_auc_score(all_labels, all_scores)
    aupr = average_precision_score(all_labels, all_scores)

    df_pred = pd.DataFrame({
        'Variant38': all_vids,
        'label': all_labels,
        'score': all_scores
    })

    df_metrics = pd.DataFrame({
        'AUC': [auc],
        'AUPR': [aupr]
    })

    os.makedirs(save_dir, exist_ok=True)
    df_pred.to_csv(f"{save_dir}/pred.txt", sep='\t', index=False)
    df_metrics.to_csv(f"{save_dir}/metrics.txt", sep='\t', index=False)

    return auc, aupr


# main
ROOT_PATH = "./benchmark/LLM_logits/synonymous-test/"
feature_names = ['CaLM', 'ERNIE-RNA', 'SpliceBERT', 'NT', 'RiNALMo']
df_meta = pd.read_csv("./benchmarks/LLM_logits/meta.info.txt", sep='\t')


for feature_name in feature_names:
    bn_path = f"{ROOT_PATH}-bn/{feature_name}_logits.pt"
    pt_path = f"{ROOT_PATH}-pt/{feature_name}_logits.pt"

    bn_data = torch.load(bn_path)
    pt_data = torch.load(pt_path)

    vocab_path = f"./benchmark/LLM_logits/token_to_ids/{feature_name}_vocab.json"
    token_to_id = load_vocab(vocab_path)

    save_dir_kl = f"./benchmark/LLM_logits/result/{feature_name}/kl"
    save_dir_ll = f"./benchmark/LLM_logits/result/{feature_name}/ll"

    # KL
    auc_kl, aupr_kl = calculate_metrics_all(
        bn_data, pt_data,
        ref_feature='ref_logits',
        alt_feature='alt_logits',
        save_dir=save_dir_kl
    )

    # LL
    auc_ll, aupr_ll = calculate_ll_metrics_all(
        bn_data, pt_data,
        ref_feature='ref_logits',
        alt_feature='alt_logits',
        token_to_id=token_to_id,
        model_name=feature_name,
        save_dir=save_dir_ll
    )

    print(f"[{feature_name}] KL  AUC={auc_kl:.4f}, AUPR={aupr_kl:.4f}")
    print(f"[{feature_name}] LL  AUC={auc_ll:.4f}, AUPR={aupr_ll:.4f}")