"""
@description: extract CaLM representations for reference and alternate codon contexts
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""


import os
import io
import gc
import lmdb
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from calm import CaLM
from datetime import datetime
import argparse


def retrieve_embeddings_pth(input_path, model, output_path, average=False):
    """
    Desc:
        Extract representations and save them as a PyTorch dictionary.
    Args:
        input_path, Input table path.
        model, Initialized feature-extraction model.
        output_path, Output path.
        average, Whether to average token representations.
    """
    df = pd.read_csv(input_path, sep="\t")
    # df = df.rename(columns={'Variant38': 'variant_hg38'})
    result_dict = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CaLM features"):
        vid = row["Variant38"]
        ref_seq = row["codon_ref"]
        alt_seq = row["codon_alt"]

        with torch.no_grad():
            ref_embed = model.embed_sequence(ref_seq, average=average)[0].squeeze().to("cpu")
            alt_embed = model.embed_sequence(alt_seq, average=average)[0].squeeze().to("cpu")

        result_dict[vid] = {"calm_diff": alt_embed - ref_embed}

    torch.save(result_dict, output_path)
    print(f"[INFO] Feature extraction and saving complete: {output_path}")


def retrieve_embeddings_lmdb(input_file, output_path, model, chunksize=10000, average=False):
    """
    Desc:
        Extract representations in chunks and save them to LMDB.
    Args:
        input_file, Input variant table path.
        output_path, Output path.
        model, Initialized feature-extraction model.
        chunksize, Rows per processing chunk.
        average, Whether to average token representations.
    """
    print(
        f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Loading codon data by chunks (size={chunksize})..."
    )
    os.makedirs(output_path, exist_ok=True)

    reader = pd.read_csv(input_file, sep="\t", chunksize=chunksize)
    device = (
        next(model.parameters()).device
        if hasattr(model, "parameters")
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    for chunk_idx, chunk in enumerate(reader):
        print(f"\n[INFO] Processing chunk {chunk_idx} ({len(chunk)} records) ...")

        chunk = chunk.rename(
            columns={
                "Variant38": "variant_hg38",
                "variant38": "variant_hg38",
                "codon_ref": "context_codon_original",
                "codon_alt": "context_codon_mutated",
            }
        )

        required_cols = ["variant_hg38", "context_codon_original", "context_codon_mutated"]
        for col in required_cols:
            if col not in chunk.columns:
                raise ValueError(f"Missing required column: {col}")

        chunk_dir = os.path.join(output_path, f"chunk_{chunk_idx:04d}")
        os.makedirs(chunk_dir, exist_ok=True)
        lmdb_path = os.path.join(chunk_dir, "CaLM")

        map_size = 1024 * 1024 * 1024 * 5
        env = lmdb.open(
            lmdb_path,
            map_size=map_size,
            subdir=True,
            readonly=False,
            lock=True,
            readahead=True,
            meminit=False,
        )

        all_keys = []
        variant_list = chunk["variant_hg38"].tolist()
        ref_seq_list = chunk["context_codon_original"].tolist()
        alt_seq_list = chunk["context_codon_mutated"].tolist()

        print(f"[INFO] Writing embeddings to {lmdb_path} ...")

        with torch.no_grad():
            with env.begin(write=True) as txn:
                for i in trange(len(chunk)):
                    variant = str(variant_list[i])
                    ref_seq = ref_seq_list[i]
                    alt_seq = alt_seq_list[i]

                    try:
                        ref_embed = (
                            model.embed_sequence(ref_seq, average=average)[0]
                            .squeeze()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        alt_embed = (
                            model.embed_sequence(alt_seq, average=average)[0]
                            .squeeze()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    except Exception as e:
                        print(f"[WARN] Failed on {variant}: {e}")
                        continue

                    key = variant.encode("utf-8")
                    buffer = io.BytesIO()
                    np.savez_compressed(
                        buffer, calm_ref_feature=ref_embed, calm_alt_feature=alt_embed
                    )
                    txn.put(key, buffer.getvalue())
                    all_keys.append(key)

                    torch.cuda.empty_cache()
                    gc.collect()

                txn.put(b"__keys__", pickle.dumps(all_keys))

        env.sync()
        env.close()
        print(f"[INFO] ✅ Chunk {chunk_idx} finished, saved to {lmdb_path}")

        del chunk
        gc.collect()

    print(f"\n[INFO] All chunks processed and saved under {output_path}")


def main(args):
    """
    Desc:
        Run the command-line workflow.
    Args:
        args, Parsed command-line arguments.
    """
    model = CaLM()
    task_dir = args.task_dir
    input_file = args.input_file
    os.makedirs(task_dir, exist_ok=True)
    retrieve_embeddings_pth(
        input_path=input_file, model=model, output_path=f"{task_dir}/CaLM.pth", average=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from CaLM.")
    parser.add_argument("--input_file", required=True, help="Input codon-context TSV.")
    parser.add_argument("--task_dir", required=True, help="Directory for CaLM.pth.")
    args = parser.parse_args()
    main(args)
