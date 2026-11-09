#!/usr/bin/env python3
"""
@description: run the complete TranSyn feature-extraction workflow
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer  # noqa: F401


def run_stage(name: str, specification: dict, variables: dict[str, str]) -> Path:
    """
    Desc:
        Execute one configured feature-extraction stage.
    Args:
        name, Output or task name.
        specification, Value supplied for specification.
        variables, Value supplied for variables.
    """
    command = [str(token).format(**variables) for token in specification["command"]]
    logging.info("Running %s: %s", name, shlex.join(command))
    subprocess.run(command, cwd=specification.get("working_directory"), check=True)
    output = Path(str(specification["output"]).format(**variables)).resolve()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"{name} did not create a non-empty output: {output}")
    return output


def normalize_biological(raw_path: Path, config: dict, output_path: Path) -> pd.DataFrame:
    """
    Desc:
        Apply the released biological-feature normalization objects.
    Args:
        raw_path, Value supplied for raw_path.
        config, Workflow configuration mapping.
        output_path, Output path.
    """
    frame = pd.read_csv(raw_path, sep="\t")
    identifier = config.get("identifier_column", "Variant38")
    selected = pd.read_csv(config["selected_columns"], sep="\t", header=None)[0].tolist()
    missing = set(selected).difference(frame.columns)
    if missing:
        raise ValueError(f"Biological feature table is missing columns: {sorted(missing)}")
    metadata_count = int(config.get("metadata_columns", 3))
    selected_frame = frame[selected].copy()
    metadata = selected_frame.iloc[:, :metadata_count]
    values = selected_frame.iloc[:, metadata_count:].replace(["na", "NA", "nan", ""], np.nan)
    empty_columns = joblib.load(config["empty_columns"])
    values = values.drop(columns=empty_columns, errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )
    imputer = joblib.load(config["imputer"])
    imputed = pd.DataFrame(
        imputer.transform(values),
        columns=values.columns,
        index=values.index,
    )
    scaler = joblib.load(config["scaler"])
    scaled = pd.DataFrame(
        scaler.transform(imputed),
        columns=imputed.columns,
        index=imputed.index,
    )
    normalized = pd.concat([metadata, scaled], axis=1)
    if identifier not in normalized:
        raise ValueError(f"Normalized table does not contain identifier column {identifier!r}")
    normalized.to_csv(output_path, sep="\t", index=False)
    return normalized


def feature_value(record: object, field: str) -> torch.Tensor | None:
    """
    Desc:
        Read a tensor from a feature record.
    Args:
        record, Value supplied for record.
        field, Value supplied for field.
    """
    value = record.get(field) if isinstance(record, dict) else record
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.float32).squeeze(0)


def merge_features(
    gpn_path: Path, calm_path: Path, biological: pd.DataFrame, config: dict, output_path: Path
) -> None:
    """
    Desc:
        Merge all biological feature tables by variant identifier.
    Args:
        gpn_path, Value supplied for gpn_path.
        calm_path, Value supplied for calm_path.
        biological, Value supplied for biological.
        config, Workflow configuration mapping.
        output_path, Output path.
    """
    gpn = torch.load(gpn_path, map_location="cpu")
    calm = torch.load(calm_path, map_location="cpu")
    identifier = config.get("identifier_column", "Variant38")
    metadata_count = int(config.get("metadata_columns", 3))
    feature_columns = biological.columns[metadata_count:]
    merged, skipped = {}, []
    for _, row in biological.iterrows():
        key = row[identifier]
        gpn_value = feature_value(gpn.get(key), config.get("gpn_field", "gpn_msa"))
        calm_value = feature_value(calm.get(key), config.get("calm_field", "calm_diff"))
        missing = [
            name for name, value in (("GPN-MSA", gpn_value), ("CaLM", calm_value)) if value is None
        ]
        if missing:
            skipped.append({"Variant38": key, "missing_features": ",".join(missing)})
            continue
        merged[key] = {
            "gpn_msa": gpn_value,
            "calm_diff": calm_value,
            "biological_feature": torch.tensor(
                row[feature_columns].to_numpy(dtype=float), dtype=torch.float32
            ),
        }
    torch.save(merged, output_path)
    pd.DataFrame(skipped, columns=["Variant38", "missing_features"]).to_csv(
        output_path.with_suffix(".skipped.tsv"), sep="\t", index=False
    )
    logging.info("Retained %d variants; skipped %d", len(merged), len(skipped))


def main() -> None:
    """
    Desc:
        Run the command-line workflow.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variables = {"input": str(args.input.resolve()), "output_dir": str(args.output_dir.resolve())}
    outputs = {}
    for name in ("biological", "gpn_msa", "codon_context", "calm"):
        outputs[name] = run_stage(name, configuration["extractors"][name], variables)
    normalized = normalize_biological(
        outputs["biological"],
        configuration["normalization"],
        args.output_dir / "biological.normalized.tsv",
    )
    merge_features(
        outputs["gpn_msa"],
        outputs["calm"],
        normalized,
        configuration["normalization"],
        args.output_dir / "features.pth",
    )


if __name__ == "__main__":
    main()
