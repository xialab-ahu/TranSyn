"""
@description: extract GPN-MSA representations around genomic variants
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import os
import io
import gc

import lmdb
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import datetime

import torch
import gpn.model
from transformers import AutoModel
from gpn.data import GenomeMSA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_embeddings_lmdb(
    input_file,
    vid_column,
    chr_column,
    pos_column,
    model,
    genome_msa,
    flank_len,
    output_path,
    chunksize=10000,
):
    """
    Desc:
        Extract representations in chunks and save them to LMDB.
    Args:
        input_file, Input variant table path.
        vid_column, Value supplied for vid_column.
        chr_column, Value supplied for chr_column.
        pos_column, Value supplied for pos_column.
        model, Initialized feature-extraction model.
        genome_msa, Loaded GenomeMSA resource.
        flank_len, Number of flanking positions to extract.
        output_path, Output path.
        chunksize, Rows per processing chunk.
    """

    print(
        f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Loading VCF file by chunks (size={chunksize})..."
    )
    os.makedirs(output_path, exist_ok=True)

    reader = pd.read_csv(input_file, sep="\t", chunksize=chunksize)
    device = next(model.parameters()).device

    for chunk_idx, chunk in enumerate(reader):
        print(f"\n[INFO] Processing chunk {chunk_idx} ({len(chunk)} records) ...")

        chunk_dir = os.path.join(output_path, f"chunk_{chunk_idx:04d}")
        os.makedirs(chunk_dir, exist_ok=True)
        lmdb_path = os.path.join(chunk_dir, "GPN-MSA")

        map_size = 1024 * 1024 * 1024 * 20
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
        variant_list = chunk[vid_column].tolist()
        chrom_list = chunk[chr_column].tolist()
        pos_list = chunk[pos_column].tolist()

        print(f"[INFO] Writing embeddings to {lmdb_path} ...")

        with torch.no_grad():
            with env.begin(write=True) as txn:
                for i in trange(len(chunk)):
                    chrom = str(chrom_list[i])
                    start_pos = int(pos_list[i])
                    variant = variant_list[i]
                    end_pos = start_pos

                    # Retrieve MSA context
                    msa = genome_msa.get_msa(
                        chrom,
                        start_pos - flank_len,
                        end_pos + flank_len + 1,
                        strand="+",
                        tokenize=True,
                    )
                    msa = torch.tensor(np.expand_dims(msa, 0).astype(np.int64)).to(device)
                    input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]

                    # Model inference
                    embedding = model(
                        input_ids=input_ids, aux_features=aux_features
                    ).last_hidden_state
                    embedding_np = embedding[0].cpu().numpy().astype(np.float32)

                    key = str(variant).encode("utf-8")
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, gpn_msa=embedding_np)
                    txn.put(key, buffer.getvalue())
                    all_keys.append(key)

                    del msa, embedding
                    torch.cuda.empty_cache()
                    gc.collect()

                txn.put(b"__keys__", pickle.dumps(all_keys))

        env.sync()
        env.close()

        print(f"[INFO] ✅ Chunk {chunk_idx} finished, saved to {lmdb_path}")

        del chunk
        gc.collect()

    print(f"\n[INFO] All chunks processed and saved under {output_path}")


def retrieve_embeddings_pth(
    input_file, vid_column, chr_column, pos_column, model, genome_msa, flank_len, output_path
):
    """
    Desc:
        Extract representations and save them as a PyTorch dictionary.
    Args:
        input_file, Input variant table path.
        vid_column, Value supplied for vid_column.
        chr_column, Value supplied for chr_column.
        pos_column, Value supplied for pos_column.
        model, Initialized feature-extraction model.
        genome_msa, Loaded GenomeMSA resource.
        flank_len, Number of flanking positions to extract.
        output_path, Output path.
    """

    print("[INFO]Loading VCF file...")
    vcf_data = pd.read_csv(input_file, sep="\t")
    if not "Variant38" in vcf_data.columns:
        vcf_data["Variant38"] = retrieve_variant(vcf_data, "#CHROM", "POS", "REF", "ALT")
    # retrieve variant info
    variant_list = vcf_data[vid_column].tolist()
    chrom_list = vcf_data[chr_column].tolist()
    pos_list = vcf_data[pos_column].tolist()
    amount = len(chrom_list)

    # init feature matrix
    result_dict = {}

    # execute feature extracting
    print("[INFO]Extracting features...")
    with torch.no_grad():
        for i in trange(amount):
            # mutation set start pos == end pos
            chrom = str(chrom_list[i])
            start_pos = int(pos_list[i])
            variant = variant_list[i]
            end_pos = start_pos
            # retrieve context embedding
            msa = genome_msa.get_msa(
                chrom,
                start_pos - flank_len,
                # end_pos+flank_len + 1,
                end_pos + flank_len,
                strand="+",
                tokenize=True,
            )
            # turn to torch tensor
            msa = torch.tensor(np.expand_dims(msa, 0).astype(np.int64)).to(device)
            # retrieve last hidden layer output
            input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]
            embedding = model(input_ids=input_ids, aux_features=aux_features).last_hidden_state
            # Save embedding under 'gpn_msa' key
            result_dict[variant] = {"gpn_msa": embedding[0].cpu()}
    torch.save(result_dict, output_path)
    print("[INFO]Feature extraction and saving complete.")


def retrieve_variant(df, chrom_col, pos_col, ref_col, alt_col, prefix=""):
    """
    Desc:
        Construct canonical variant identifiers from genomic columns.
    Args:
        df, Input data frame.
        chrom_col, Value supplied for chrom_col.
        pos_col, Value supplied for pos_col.
        ref_col, Value supplied for ref_col.
        alt_col, Value supplied for alt_col.
        prefix, Value supplied for prefix.
    """
    variant = [
        f"{prefix}{chro}_{pos}_{ref}/{alt}"
        for chro, pos, ref, alt in zip(df[chrom_col], df[pos_col], df[ref_col], df[alt_col])
    ]
    return variant


def main(args):
    """
    Desc:
        Run the command-line workflow.
    Args:
        args, Parsed command-line arguments.
    """
    task_dir = args.task_dir
    input_file = args.input_file
    os.makedirs(task_dir, exist_ok=True)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading MSA resource: {args.msa_path}")
    genome_msa = GenomeMSA(args.msa_path)
    print(f"[INFO] Loading GPN-MSA model: {args.model_path}")
    model = AutoModel.from_pretrained(args.model_path).to(device)
    model.eval()
    retrieve_embeddings_pth(
        input_file=input_file,
        vid_column="Variant38",
        chr_column="#CHROM",
        pos_column="POS",
        model=model,
        genome_msa=genome_msa,
        flank_len=args.flank_length,
        output_path=f"{task_dir}/GPN_MSA.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from GPN-MSA.")
    parser.add_argument("--task_dir", required=True, help="Directory for GPN_MSA.pth.")
    parser.add_argument("--input_file", required=True, help="Input GRCh38 TSV.")
    parser.add_argument("--model_path", required=True, help="Downloaded GPN-MSA model directory.")
    parser.add_argument("--msa_path", required=True, help="Downloaded GenomeMSA Zarr archive.")
    parser.add_argument("--flank_length", type=int, default=64, help="Positions on each side.")
    args = parser.parse_args()
    main(args)
