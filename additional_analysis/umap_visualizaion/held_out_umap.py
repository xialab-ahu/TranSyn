"""
@description: UMAP visualization and quantitative held-out calculation
@author: chen ye
@email: q23101020@ahu.stu.edu.cn
"""

# path settings
from pathlib import Path
import os, sys, json, platform
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

# common pkgs
import torch
import numpy as np
import pandas as pd
import yaml
import joblib
from torch.utils.data import DataLoader, ConcatDataset

# visual and stats pkgs
import umap
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, pairwise_distances

# CUDA and other settings
SEED = 44
PCT = 0.02
model_combo = "GPN_CaLM_NewBio"
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

# customized functions
from functions.Dataset import MyDataset, concat_and_loader
from functions.Utils import set_full_deterministic_seed
from functions.TranSynAttention import TranSynAttention
from functions.ModelScaffold import model_fn, build_backbone
from functions.DyPositionTransformer import DyPositionTransformer

# config
with open('./config/config.yaml', 'r', encoding='utf-8') as file:
	config = yaml.safe_load(file)
	spc_config = config[model_combo]
     
# =========== Model construction ===========
tsa_dna_module = build_backbone(name="ResNet1D", seq_len=129, embed_dim=768)
tsa_rna_module = DyPositionTransformer(
									input_dim=768,
									num_layers=4,
									d_model=256,
									num_heads=4,
									dff=512,
									dropout_rate=0.1,
									head_hidden=128,
							).to(device) 
tsa_bio_module = build_backbone(name="MLP", embed_dim=49, output_dim=spc_config['BIO_HIDDEN']).to(device) 
my_model = TranSynAttention(
			dna_module = tsa_dna_module,
			rna_module = tsa_rna_module,
			bio_module = tsa_bio_module,
			hidden_layer = spc_config['HEAD_HIDDEN'],
			dropout = spc_config['DROPOUT_RATE'],
		).to(device) 

# =========== Train and test data ===========
TARGET_ROOT_PATH = "/data1/yechen/11_transfer_learing_predictor/C-transfer-stategy/MetaSyn/feature/bio_merge/"
train_data_bn = MyDataset(data_path=f"{TARGET_ROOT_PATH}/train_neg_diff_with_new_bio.pth", label=0)
train_data_pt = MyDataset(data_path=f"{TARGET_ROOT_PATH}/train_pos_diff_with_new_bio.pth", label=1)
train_data = ConcatDataset([train_data_bn, train_data_pt])
test_data_bn = MyDataset(data_path=f"{TARGET_ROOT_PATH}/test_neg_diff_with_new_bio.pth", label=0)
test_data_pt = MyDataset(data_path=f"{TARGET_ROOT_PATH}/test_pos_diff_with_new_bio.pth", label=1)
test_data = ConcatDataset([test_data_bn, test_data_pt])


@torch.no_grad()
def extract_features(model, loader, device):
	"""
    Desc:
        extract features
    Args:
        model, model
        loader, feature
        device, GPU/CPU
	return:
		dna_feat: [N, D]
		rna_feat: [N, D]
		labels:   [N]
	"""
	model.eval()

	dna_list, rna_list, label_list = [], [], []
	for batch in loader:
		dna_x = batch["gpn_msa"].to(device)
		rna_x = batch["calm_diff"].to(device)
		labels = batch["label"].cpu().numpy()

		# test mode: only outputs, so re-run feature branch
		dna_feat = model.dna_module(dna_x, use_tsrs=False)
		rna_feat = model.rna_module(rna_x, backbone=True)

		dna_feat = model.norm_dna(model.proj_dna(dna_feat))
		rna_feat = model.norm_rna(model.proj_rna(rna_feat))

		dna_list.append(dna_feat.detach().cpu().numpy())
		rna_list.append(rna_feat.detach().cpu().numpy())
		label_list.append(labels)

	return (
		np.concatenate(dna_list, axis=0),
		np.concatenate(rna_list, axis=0),
		np.concatenate(label_list, axis=0),
	)


def separation(x, y, metric):
    """
    Desc:
        calculate the separation metrics, using the 
    Args:
        x, the baseline feature
        y, the transfer feature
        metric, how to measture the distance: 'cityblock'|'cosine'|'euclidean'|'l1'|'l2'|'manhattan'|'nan_euclidean'
    """
    # 🔸 Fixed held-out distances: labels enter evaluation only, never UMAP fitting.
    distances = pairwise_distances(x, metric=metric)
    np.fill_diagonal(distances, 0) # a safety, the distance to oneself should be zero
    scores = silhouette_samples(distances, y, metric="precomputed") # precomputed means original matrix is distance matrix, no need for re-cal. scores = (b-a)/max(a,b)
    observed = float(scores.mean()) # mean silhouette score for every sample

    # 🔸 Permutation Test, permutate the y 999 times and re-cal the silhouette score under random label
    rng = np.random.default_rng(42)
    null = np.array([silhouette_samples(distances, rng.permutation(y),
                     metric="precomputed").mean() for _ in range(999)])
    return observed, float((1 + (null >= observed).sum()) / 1000), scores


def main():
    rows, sample_rows = [], []
    train_ids_ref = test_ids_ref = None
    # 🔸 Under Baseline / Transfer
    for _, group in enumerate([True, False]):
        set_full_deterministic_seed(SEED)
        checkpoint = ROOT / f"result/Section3/GPN_CaLM_NewBio/0.02_{group}_best.model"
        my_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        my_model.to(device).eval()
        original_loader, _ = concat_and_loader(train_data, split_rate=PCT,
                                               batch_size=spc_config['BATCH_SIZE'], seed=SEED)
        train_loader = DataLoader(original_loader.dataset, batch_size=32,
                                  shuffle=False, drop_last=False, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False,
                                 drop_last=False, num_workers=0)
        def ids(loader):
            return [str(loader.dataset[i]['vid']) for i in range(len(loader.dataset))]
        train_ids, test_ids = ids(train_loader), ids(test_loader)
        assert len(set(train_ids)) == len(train_ids)
        assert len(set(test_ids)) == len(test_ids)
        assert not set(train_ids).intersection(test_ids), "Train/test sample overlap"
        if train_ids_ref is not None:
            assert train_ids == train_ids_ref and test_ids == test_ids_ref
        train_ids_ref, test_ids_ref = train_ids, test_ids
        print(f"Transfer={group}: extracting {len(train_ids)} train / {len(test_ids)} test", flush=True)
        # Extract features
        train_dna, train_rna, train_y = extract_features(my_model, train_loader, device)
        test_dna, test_rna, test_y = extract_features(my_model, test_loader, device)

        # 🔸 Under DNA/RNA modal
        for _, (modality, train_x, test_x) in enumerate([
                ("DNA", train_dna, test_dna), ("RNA", train_rna, test_rna)]):
            assert np.isfinite(train_x).all() and np.isfinite(test_x).all()
            reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, metric="cosine",
                                random_state=42, transform_seed=42, n_jobs=1)
            reducer.fit(train_x)  # Train features only; no labels.
            train_emb = reducer.embedding_
            test_emb = reducer.transform(test_x)  # Never fit on held-out samples.
            prefix = f"transfer_{group}_{modality}"
            joblib.dump(reducer, OUT / f"{prefix}_reducer.joblib")
            np.savez_compressed(OUT / f"{prefix}_features.npz", train=train_x,
                                test=test_x, train_y=train_y, test_y=test_y,
                                train_ids=train_ids, test_ids=test_ids)
            for split, emb, y, vids in [("train", train_emb, train_y, train_ids),
                                       ("test", test_emb, test_y, test_ids)]:
                pd.DataFrame(dict(vid=vids, label=y.astype(int), UMAP1=emb[:,0],
                                  UMAP2=emb[:,1])).to_csv(OUT / f"{prefix}_{split}_embedding.csv", index=False)
            for space, x, metric in [("original_features", test_x, "cosine"),
                                      ("umap_2d", test_emb, "euclidean")]:
                score, p, per_sample = separation(x, test_y, metric)
                rows.append(dict(transfer=group, modality=modality, space=space,
                    metric=metric, n_train=len(train_y), n_test=len(test_y),
                    n_test_negative=int((test_y==0).sum()), n_test_positive=int((test_y==1).sum()),
                    silhouette=score, permutation_p_greater=p))
                sample_rows.extend(dict(transfer=group, modality=modality, space=space,
                    vid=vid, label=int(label), silhouette=float(s))
                    for vid,label,s in zip(test_ids,test_y,per_sample))
                print(rows[-1], flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "held_out_separation.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(OUT / "held_out_sample_silhouette.csv", index=False)
    comparison = results.pivot(index=['modality','space'], columns='transfer', values='silhouette')
    comparison.columns = ['baseline', 'transfer']
    comparison['transfer_minus_baseline'] = comparison['transfer'] - comparison['baseline']
    comparison.to_csv(OUT / "separation_comparison.csv")
    (OUT / "run_metadata.json").write_text(json.dumps(dict(seed=SEED, pct=PCT,
        umap_random_state=42, transform_seed=42, n_neighbors=50, min_dist=.5,
        device=str(device), torch=torch.__version__, umap=umap.__version__,
        sklearn=sklearn.__version__, python=platform.python_version(), permutations=999,
        source_notebook=str(ROOT/'Result-3.3.1-few-shot.ipynb')), indent=2))
    print(comparison.to_string(), flush=True)
    import runpy
    runpy.run_path(str(OUT / "plot_individual_umap.py"), run_name="__main__")

if __name__ == "__main__":
    main()
