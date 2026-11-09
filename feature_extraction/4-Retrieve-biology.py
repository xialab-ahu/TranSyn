"""
@description: extract and merge biological annotations for TranSyn
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import warnings

warnings.filterwarnings("ignore")

import os
import joblib
import argparse
import subprocess
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


SYNMALL_TABIX_SCRIPT = None
SILVA_SCRIPT = None
GDI_PATH = None
GENE_META_PATH = None
GENE_ANNO_PATH = None
liftover_tool = None
chain_file_38_to_19 = None
selected_header = []


def create_bed_from_vcf(task_dir, input_file):
    """
    Desc:
        Convert GRCh38 variant positions to BED intervals.
    Args:
        task_dir, Directory containing intermediate and output files.
        input_file, Input variant table path.
    """
    print("[INFO] Creating BED file from VCF file...")
    df = pd.read_csv(input_file, sep="\t")
    chrom = df["#CHROM"].tolist()
    new_chrom = ["chr" + str(i) for i in chrom]  # liftover needs a 'chr' prefix
    df_bed = pd.DataFrame(
        {"#CHR": new_chrom, "START": df["POS"], "END": df["POS"], "OLD_POS": df["POS"]}
    )
    df_bed.to_csv(f"{task_dir}/liftover_input.bed", sep="\t", index=False, header=None)


def run_liftover(task_dir):
    """
    Desc:
        Lift variant intervals from GRCh38 to GRCh37.
    Args:
        task_dir, Directory containing intermediate and output files.
    """
    print("[INFO] Run LiftOver...")
    input_bed = f"{task_dir}/liftover_input.bed"
    output_bed = f"{task_dir}/liftover_output.bed"
    unmap_bed = f"{task_dir}/liftover_unmapped.bed"
    liftOver_cmd = f"{liftover_tool} {input_bed} {chain_file_38_to_19} {output_bed} {unmap_bed}"
    print(f"[INFO] Exec cmd: {liftOver_cmd}")
    subprocess.run(liftOver_cmd, shell=True)


def retrieve_position(df, chrom_col, pos_col, prefix=""):
    """
    Desc:
        Construct chromosome-position identifiers.
    Args:
        df, Input data frame.
        chrom_col, Value supplied for chrom_col.
        pos_col, Value supplied for pos_col.
        prefix, Value supplied for prefix.
    """
    link = [f"{prefix}{chro}_{pos}" for chro, pos in zip(df[chrom_col], df[pos_col])]
    return link


def create_vcf_from_bed(task_dir, input_file):
    """
    Desc:
        Join lifted positions back to the original variant table.
    Args:
        task_dir, Directory containing intermediate and output files.
        input_file, Input variant table path.
    """
    print("[INFO] Creating VCF file from BED file...")
    vcf = pd.read_csv(input_file, sep="\t")
    bed = pd.read_csv(f"{task_dir}/liftover_output.bed", sep="\t", header=None)
    bed.drop_duplicates(inplace=True)
    bed.columns = ["#CHR", "POS_hg38", "END", "OLD_POS"]
    bed["POS_hg38"] = bed["POS_hg38"].astype("Int64")
    bed["LINK"] = retrieve_position(bed, "#CHR", "OLD_POS")
    vcf["LINK"] = retrieve_position(vcf, "#CHROM", "POS", prefix="chr")

    result_df = vcf.merge(bed, how="left", left_on="LINK", right_on="LINK", suffixes=("", "_lift"))
    result_df.drop(["#CHR", "END", "OLD_POS", "LINK"], axis=1, inplace=True)
    result_df["Variant38"] = retrieve_variant(result_df, "#CHROM", "POS_hg38", "REF", "ALT")
    print(f"[Nested INFO] Result dataframe:\n{result_df.head()}")
    result_df.to_csv(f"{task_dir}/liftover_output_hg38.vcf", sep="\t", index=False)


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


def retrieve_raw_synmall(input_file, output_file):
    """
    Desc:
        Run the configured SynMall annotation script.
    Args:
        input_file, Input variant table path.
        output_file, Value supplied for output_file.
    """
    cmd = f"python {SYNMALL_TABIX_SCRIPT} -i {input_file} -o {output_file}"
    # cmd = ["python", SYNMALL_TABIX_SCRIPT, "-i", input_file, "-o", output_file]
    print(f"[INFO] Exec synmall cmd: {cmd}")
    subprocess.run(cmd, shell=True)


def retrieve_raw_silva(input_file, output_dir):
    """
    Desc:
        Run the configured SilVA annotation script.
    Args:
        input_file, Input variant table path.
        output_dir, Output directory.
    """
    cmd = f"python {SILVA_SCRIPT} --genome hg38 --task_name {output_dir} --input_file {input_file}"
    # cmd = ["python", SILVA_SCRIPT,
    #        "--genome", "hg38",
    #        "--task_name", output_dir,
    #        "--input_file", input_file]
    print(f"[INFO] Exec silva cmd: {cmd}")
    subprocess.run(cmd, shell=True)


def retrieve_raw_gdi(input_path, output_path):
    """
    Desc:
        Join gene-damage and GC-content annotations.
    Args:
        input_path, Input table path.
        output_path, Output path.
    """
    print("[INFO] Annotate GDI + GC...")
    df_gdi = pd.read_csv(GDI_PATH, sep="\t")
    df_meta = pd.read_csv(input_path, sep="\t")
    if not "Variant38" in df_meta.columns:
        df_meta["Variant38"] = retrieve_variant(df_meta, "#CHROM", "POS", "REF", "ALT")
    df_meta = df_gdi[df_gdi["Variant38"].isin(df_meta["Variant38"])]
    df_meta.to_csv(output_path, sep="\t", index=False)


def retrieve_raw_gene_anno(input_file, output_file):
    """
    Desc:
        Join gene-level annotation resources.
    Args:
        input_file, Input variant table path.
        output_file, Value supplied for output_file.
    """
    print("[INFO] Loading gene information and annotation...")
    # Read gene meta info
    gene_info = pd.read_csv(
        GENE_META_PATH, sep="\t", usecols=["gene_id", "transcript_id", "refseq_transcript_id_mane"]
    )
    transcript2gene = dict(zip(gene_info["transcript_id"], gene_info["gene_id"]))
    # refseq2gene = dict(zip(gene_info['refseq_transcript_id_mane'], gene_info['gene_id']))
    usecols = [
        "gene_id",
        "denovo_dis_all",
        "denovo_dis_syn",
        "gevir_percentile",
        "loeuf_percentile",
        "virlof_percentile",
        "gevir_ad_enrichment",
        "loeuf_ad_enrichment",
        "virlof_ad_enrichment",
        "gevir_ar_enrichment",
        "loeuf_ar_enrichment",
        "virlof_ar_enrichment",
        "aggarwala_gene_score",
        "petrovski_score",
        "rvis_percentile",
        "go_term_accession_num",
        "hpo_id_num",
    ]
    # Read gene anno info
    gene_centric_anno = pd.read_csv(GENE_ANNO_PATH, sep="\t", usecols=usecols)

    def map_transcript_to_gene(transcript_str):
        """
        Desc:
            Map the first recognized transcript identifier to its gene.
        Args:
            transcript_str, Semicolon-delimited transcript identifiers.
        """
        for enst in transcript_str.split(";"):
            if enst in transcript2gene:
                return transcript2gene[enst]  # return the first gene_id
        return None  # if not match

    # def map_refseq_to_gene(transcript_str):
    #     """ for every transcript_id (multiple ENST possible, split by semicolon) """
    #     for enst in transcript_str.split(';'):
    #         if enst in refseq2gene:
    #             return refseq2gene[enst]  # return the first gene_id
    #     return None # if not match
    df = pd.read_csv(
        input_file, sep="\t", usecols=["transcript_id", "Variant38"]
    )  # NOTE transcript id is needed HERE!
    df["transcript_id"] = df["transcript_id"].apply(
        lambda x: ";".join([item for item in x.split("|") if item.startswith("ENST")])
    )
    # for multiple transcript id, return unique gene id
    df["gene_id"] = df["transcript_id"].apply(map_transcript_to_gene)
    # merge gene-centric annotation
    result_df = df.merge(
        gene_centric_anno, how="left", left_on="gene_id", right_on="gene_id", suffixes=("", "_m")
    )
    result_df.to_csv(output_file, sep="\t", index=False)


def process_synmall_output(
    df_path, original_header, selected_header, canonical_transcript_set, save_dir
):
    """
    Desc:
        Transform raw SynMall annotations into model features.
    Args:
        df_path, Value supplied for df_path.
        original_header, Value supplied for original_header.
        selected_header, Value supplied for selected_header.
        canonical_transcript_set, Value supplied for canonical_transcript_set.
        save_dir, Value supplied for save_dir.
    """
    # read data
    # df = pd.read_csv(df_path, sep='\t', comment='#', header=None)
    # df.drop_duplicates(inplace=True)
    df = pd.read_csv(df_path, sep="\t", header=4)
    df.drop_duplicates(inplace=True)
    df.columns = original_header

    # the transcript meta info
    transcript_meta_list = [
        "transcript_id",
        "cds_pos",
        "chr",
        "pos",
        "ref",
        "alt",
        "mutation_source",
        "codon_alternate",
        "delta_ess",
        "delta_ese",
        "delta_esr",
        "splice_site_acceptor",
        "splice_site_donor",
    ]

    def select_transcript_info(info_str, canonical_set, meta_list):
        """
        Desc:
            Select the canonical or best-supported transcript annotation.
        Args:
            info_str, Semicolon-delimited transcript annotation records.
            canonical_set, Canonical transcript identifiers.
            meta_list, Field names in each annotation record.
        """
        if pd.isna(info_str):
            return {col: np.nan for col in meta_list}
        # Split all record
        records = [dict(zip(meta_list, record.split("|"))) for record in info_str.split(";")]
        # filter canonical
        canonical_records = [r for r in records if r["transcript_id"] in canonical_set]
        selected_record = None
        if len(canonical_records) == 1:
            selected_record = canonical_records[0]
        elif len(canonical_records) > 1:
            # multiple record select longest mutation_source (most confident)
            selected_record = max(
                canonical_records,
                key=lambda x: len(x.get("mutation_source", "")),
                default=canonical_records[0],
            )
        else:  # len(canonical_records) == 0
            # multiple record select longest mutation_source (most confident)
            if records:
                selected_record = max(
                    records, key=lambda x: len(x.get("mutation_source", "")), default=records[0]
                )
            else:
                return {col: np.nan for col in meta_list}
        # retreive necessary field
        return {
            col: selected_record.get(col, np.nan)
            for col in [
                "transcript_id",
                "cds_pos",
                "codon_alternate",
                "d_ess",
                "d_ese",
                "d_esr",
                "splice_site_acceptor",
                "splice_site_donor",
            ]
        }

    # expand
    extracted_cols = df["transcript_wise_info"].apply(
        lambda x: pd.Series(
            select_transcript_info(x, canonical_transcript_set, transcript_meta_list)
        )
    )
    # concat
    df = pd.concat([df, extracted_cols], axis=1)
    # generate variant id
    df["Variant38"] = retrieve_variant(df, "#CHROM", "POS", "REF", "ALT")
    df.drop_duplicates(inplace=True)
    # meta info
    df_meta = df[["Variant38", "transcript_id"]]
    df_meta.drop_duplicates(inplace=True)
    # cds info
    df_cds = df[
        ["#CHROM", "POS", "REF", "ALT", "Variant38", "transcript_id", "cds_pos", "codon_alternate"]
    ]
    df_cds.drop_duplicates(inplace=True)
    df.drop(["transcript_id", "cds_pos"], axis=1, inplace=True)

    # select columns
    available_header = [col for col in selected_header if col in df.columns]
    df = df[available_header]

    # change bool to numeric
    for col in ["splice_site_acceptor", "splice_site_donor"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("True", "1")
            .str.replace("False", "0")
            .replace("nan", np.nan)
        )
    # reformat to numeric
    numeric_cols = [col for col in df.columns if col != "Variant38"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # save result
    if save_dir:
        df.to_csv(f"{save_dir}/SynMall.output.txt", sep="\t", index=False)
        df_meta.to_csv(f"{save_dir}/input.meta.txt", sep="\t", index=False)
        df_cds.to_csv(f"{save_dir}/input.cds.txt", sep="\t", index=False)
    return df


def merge_features(task_dir):
    """
    Desc:
        Merge all biological feature tables by variant identifier.
    Args:
        task_dir, Directory containing intermediate and output files.
    """
    print("[INFO] Start merge features...")
    # Meta info
    df_meta = pd.read_csv(f"{task_dir}/input.meta.txt", sep="\t", usecols=["Variant38"])
    print(f"[INFO] Before merge: {len(df_meta)}")
    # GDI
    df_gdi = pd.read_csv(f"{task_dir}/gdi.gc.output.txt", sep="\t")
    df_result = df_meta.merge(
        df_gdi, how="left", left_on="Variant38", right_on="Variant38", suffixes=("", "_m")
    )
    print(f"[INFO] After merge gdi: {len(df_result)}")
    # SILVA
    sliva_columns = [
        "Variant38",
        "TFBS",
        "TE",
        "dPSIZ",
        "DSP",
        "CpG",
        "CpGExon",
        "fPre",
        "fPost",
        "dRSCU",
        "aRSCU",
        "MES",
        "adMES",
        "dMES+",
        "dMES-",
        "MES-MC",
        "MES-CS",
        "MES-KM",
        "SR-",
        "SR+",
        "FAS6-",
        "FAS6+",
        "PESE-",
        "PESE+",
        "PESS-",
        "PESS+",
    ]
    df_silva = pd.read_csv(f"{task_dir}/SilVA.output.txt", sep="\t", usecols=sliva_columns)
    df_result = df_result.merge(
        df_silva, how="left", left_on="Variant38", right_on="Variant38", suffixes=("", "_m")
    )
    print(f"[INFO] After merge sliva: {len(df_result)}")
    # Gene-centric
    df_gene_centric = pd.read_csv(f"{task_dir}/Gene.centric.output.txt", sep="\t")
    df_gene_centric.rename(
        columns={"variant38": "Variant38", "Transcript_stable_ID": "transcript_id"}, inplace=True
    )
    df_result = df_result.merge(
        df_gene_centric, how="left", left_on="Variant38", right_on="Variant38", suffixes=("", "_m")
    )
    print(f"[INFO] After merge gene centric: {len(df_result)}")
    # SynMall
    df_synmall = pd.read_csv(f"{task_dir}/SynMall.output.txt", sep="\t")
    df_result = df_result.merge(
        df_synmall, how="left", left_on="Variant38", right_on="Variant38", suffixes=("", "_m")
    )
    print(f"[INFO] After merge synMall: {len(df_result)}")
    # rearrange
    df_result = df_result[selected_header]
    df_result.to_csv(f"{task_dir}/biological.raw.txt", sep="\t", index=False)


def load_and_process_new_data(task_dir, pkl_root):
    """
    Desc:
        Apply released biological-feature preprocessing objects.
    Args:
        task_dir, Directory containing intermediate and output files.
        pkl_root, Value supplied for pkl_root.
    """
    df_new = pd.read_csv(f"{task_dir}/biological.raw.txt", sep="\t")
    # 1. Load objs
    empty_cols = joblib.load(f"{pkl_root}/empty_cols.pkl")
    imputer = joblib.load(f"{pkl_root}/imputer.pkl")
    scaler = joblib.load(f"{pkl_root}/scaler.pkl")
    # 2. Split meta info and feature
    df_new_info = df_new.iloc[:, :3]
    df_new_fea = df_new.iloc[:, 3:].copy()
    # 3. replace NaN
    df_new_fea.replace(["na", "NA", "nan", ""], np.nan, inplace=True)
    # 4. delete empty columns
    df_new_fea.drop(columns=empty_cols, errors="ignore", inplace=True)
    # 5. impute missing values
    df_new_fea_imputed = pd.DataFrame(
        imputer.transform(df_new_fea), columns=df_new_fea.columns, index=df_new_fea.index
    )
    # 6. scale
    df_new_scaled = pd.DataFrame(
        scaler.transform(df_new_fea_imputed),
        columns=df_new_fea_imputed.columns,
        index=df_new_fea_imputed.index,
    )
    # 7. concat original meta columns
    df_final = pd.concat([df_new_info, df_new_scaled], axis=1)
    df_final.to_csv(f"{task_dir}/biological.scaler.imputed.txt", sep="\t", index=False)


def main(args):
    """
    Desc:
        Run the command-line workflow.
    Args:
        args, Parsed command-line arguments.
    """
    global SYNMALL_TABIX_SCRIPT, SILVA_SCRIPT, GDI_PATH
    global GENE_META_PATH, GENE_ANNO_PATH, liftover_tool, chain_file_38_to_19
    global selected_header

    resources = yaml.safe_load(open(args.resource_config, encoding="utf-8"))
    SYNMALL_TABIX_SCRIPT = resources["synmall_tabix_script"]
    SILVA_SCRIPT = resources["silva_script"]
    GDI_PATH = resources["gdi_path"]
    GENE_META_PATH = resources["gene_meta_path"]
    GENE_ANNO_PATH = resources["gene_annotation_path"]
    liftover_tool = resources["liftover_executable"]
    chain_file_38_to_19 = resources["hg38_to_hg19_chain"]
    original_header = pd.read_csv(resources["original_header"], sep="\t", header=None)[0].tolist()
    selected_header = pd.read_csv(resources["selected_header"], sep="\t", header=None)[0].tolist()
    canonical_transcript_set = set(
        pd.read_csv(
            resources["canonical_transcripts"],
            sep="\t",
            usecols=["Transcript_stable_ID"],
        )["Transcript_stable_ID"].tolist()
    )
    pkl_root = resources["preprocessing_objects"]
    task_dir = args.task_dir
    input_file = args.input_file  # #CHROM   POS   REF   ALT
    genome = args.genome

    if genome == "hg19":
        create_bed_from_vcf(task_dir, input_file)
        run_liftover(task_dir)
        create_vcf_from_bed(task_dir, input_file)
        input_file = "liftover_output_hg38.vcf"

    # make directory
    print(f"[INFO] Task starts at {task_dir}...")
    os.makedirs(task_dir, exist_ok=True)
    # 1 Annotate SynMall infos
    retrieve_raw_synmall(input_file=input_file, output_file=f"{task_dir}/SynMall.raw.txt")
    process_synmall_output(
        df_path=f"{task_dir}/SynMall.raw.txt",
        original_header=original_header,
        selected_header=selected_header,
        canonical_transcript_set=canonical_transcript_set,
        save_dir=task_dir,
    )
    # 2 Annotate SilVA infos
    retrieve_raw_silva(input_file=input_file, output_dir=task_dir)
    # 3 Annotate GDI infos
    retrieve_raw_gdi(
        input_path=f"{task_dir}/input.meta.txt", output_path=f"{task_dir}/gdi.gc.output.txt"
    )
    # 4 Annotate Gene centric infos
    retrieve_raw_gene_anno(
        input_file=f"{task_dir}/input.meta.txt", output_file=f"{task_dir}/Gene.centric.output.txt"
    )
    # 5 Merge features
    merge_features(task_dir)
    # 6 Scale and impute features
    load_and_process_new_data(task_dir=task_dir, pkl_root=pkl_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from biological data.")
    parser.add_argument(
        "--task_dir", type=str, required=True, help="Path to save the output files."
    )
    parser.add_argument("--input_file", type=str, default=None, help="input VCF file")
    parser.add_argument("--genome", type=str, default="hg38", help="input alternate fasta file")
    parser.add_argument(
        "--resource_config",
        required=True,
        help="YAML file containing user-installed annotation resources.",
    )
    args = parser.parse_args()
    main(args)
