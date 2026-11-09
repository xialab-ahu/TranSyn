#!/usr/bin/env python3
"""
@description: run the complete TranSyn feature-extraction workflow
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
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


INSTALL_HINTS = {
    "bio_feature": "conda env create -f config/bio_feature.yml",
    "GPNMSA": "install GPN-MSA from https://gpn.readthedocs.io and download its model/MSA assets",
    "calm": "install CaLM from https://github.com/oxpig/CaLM",
}


def formatted_command(specification: dict, variables: dict[str, str]) -> list[str]:
    """
    Desc:
        Resolve placeholders in a configured stage command.
    Args:
        specification, Stage configuration mapping.
        variables, Placeholder values for the current run.
    """
    return [str(token).format(**variables) for token in specification["command"]]


def command_option(command: list[str], option: str) -> str | None:
    """
    Desc:
        Return the value following a command-line option.
    Args:
        command, Resolved command tokens.
        option, Option name to locate.
    """
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def configured_path(value: str, config_dir: Path) -> Path:
    """
    Desc:
        Resolve a configured filesystem path relative to the pipeline YAML.
    Args:
        value, Configured path string.
        config_dir, Directory containing the pipeline YAML.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def check_prerequisites(
    configuration: dict,
    variables: dict[str, str],
    config_path: Path,
    input_path: Path,
) -> None:
    """
    Desc:
        Validate Conda environments, Python modules, weights, and resource files.
    Args:
        configuration, Complete pipeline configuration mapping.
        variables, Placeholder values for the current run.
        config_path, Pipeline YAML path.
        input_path, Input variant table path.
    """
    issues: list[str] = []
    config_dir = config_path.resolve().parent
    if not input_path.is_file():
        issues.append(f"input file does not exist: {input_path}")

    conda = shutil.which("conda")
    available_environments: set[str] = set()
    if conda is None:
        issues.append("Conda executable was not found on PATH")
    else:
        result = subprocess.run(
            [conda, "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            issues.append(f"cannot query Conda environments: {result.stderr.strip()}")
        else:
            available_environments = {
                Path(path).name for path in json.loads(result.stdout).get("envs", [])
            }

    checked_modules: set[tuple[str, tuple[str, ...]]] = set()
    reported_environments: set[str] = set()
    for stage_name, specification in configuration["extractors"].items():
        command = formatted_command(specification, variables)
        environment = command_option(command, "-n") or command_option(command, "--name")
        if (
            environment
            and environment not in available_environments
            and environment not in reported_environments
        ):
            reported_environments.add(environment)
            hint = INSTALL_HINTS.get(environment, f"create the Conda environment {environment!r}")
            issues.append(f"missing Conda environment {environment!r}; {hint}")
        modules = tuple(specification.get("required_modules", []))
        module_check = (environment or "", modules)
        if (
            conda
            and environment in available_environments
            and modules
            and module_check not in checked_modules
        ):
            checked_modules.add(module_check)
            code = (
                "import importlib.util,sys; missing=[m for m in sys.argv[1:] "
                "if importlib.util.find_spec(m) is None]; "
                "print(','.join(missing)); raise SystemExit(bool(missing))"
            )
            result = subprocess.run(
                [conda, "run", "-n", environment, "python", "-c", code, *modules],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                missing = result.stdout.strip() or result.stderr.strip()
                issues.append(f"environment {environment!r} is missing Python modules: {missing}")

        working_directory = configured_path(
            str(specification["working_directory"]).format(**variables), config_dir
        )
        if not working_directory.is_dir():
            issues.append(f"{stage_name} working directory does not exist: {working_directory}")
        elif "python" in command:
            script_index = command.index("python") + 1
            if script_index < len(command):
                script = configured_path(command[script_index], working_directory)
                if not script.is_file():
                    issues.append(f"{stage_name} script does not exist: {script}")

        for option, expected_type in (
            ("--resource_config", "file"),
            ("--model_path", "directory"),
            ("--msa_path", "path"),
            ("--transcript_info", "file"),
        ):
            value = command_option(command, option)
            if value is None:
                continue
            path = configured_path(value, config_dir)
            valid = path.is_dir() if expected_type == "directory" else path.exists()
            if not valid:
                issues.append(f"{stage_name} {option} does not exist: {path}")
            if option == "--resource_config" and path.is_file():
                resources = yaml.safe_load(path.read_text(encoding="utf-8"))
                for resource_name, resource_value in resources.items():
                    resource_path = configured_path(str(resource_value), path.parent)
                    if not resource_path.exists():
                        issues.append(
                            f"biological resource {resource_name!r} does not exist: {resource_path}"
                        )

    for name in ("selected_columns", "empty_columns", "imputer", "scaler"):
        path = configured_path(configuration["normalization"][name], config_dir)
        if not path.is_file():
            issues.append(f"normalization file {name!r} does not exist: {path}")

    if issues:
        details = "\n".join(f"  - {issue}" for issue in issues)
        raise RuntimeError(f"Feature-extraction prerequisite check failed:\n{details}")


def run_stage(
    name: str,
    specification: dict,
    variables: dict[str, str],
    dry_run: bool = False,
) -> Path:
    """
    Desc:
        Execute one configured feature-extraction stage.
    Args:
        name, Output or task name.
        specification, Value supplied for specification.
        variables, Value supplied for variables.
        dry_run, Print the resolved command without executing it.
    """
    command = formatted_command(specification, variables)
    output = Path(str(specification["output"]).format(**variables)).resolve()
    logging.info("Stage %s command: %s", name, shlex.join(command))
    logging.info("Stage %s expected output: %s", name, output)
    if dry_run:
        return output
    working_directory = str(specification["working_directory"]).format(**variables)
    subprocess.run(command, cwd=working_directory, check=True)
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved workflow without running feature extractors.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for name in ("selected_columns", "empty_columns", "imputer", "scaler"):
        configuration["normalization"][name] = str(
            configured_path(configuration["normalization"][name], args.config.resolve().parent)
        )
    variables = {
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "config_dir": str(args.config.resolve().parent),
    }
    if args.dry_run:
        logging.info("Dry run: prerequisite existence checks are skipped")
    else:
        check_prerequisites(configuration, variables, args.config, args.input.resolve())
        args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in ("biological", "gpn_msa", "codon_context", "calm"):
        outputs[name] = run_stage(
            name,
            configuration["extractors"][name],
            variables,
            dry_run=args.dry_run,
        )
    if args.dry_run:
        normalization = configuration["normalization"]
        logging.info("Normalization header: %s", normalization["selected_columns"])
        logging.info("Training-empty columns: %s", normalization["empty_columns"])
        logging.info("Fitted IterativeImputer: %s", normalization["imputer"])
        logging.info("Fitted scaler: %s", normalization["scaler"])
        logging.info(
            "Normalization order: select header -> remove empty columns -> "
            "IterativeImputer.transform -> scaler.transform"
        )
        logging.info("Merged feature output: %s", args.output_dir.resolve() / "features.pth")
        logging.info(
            "Skipped-feature report: %s",
            args.output_dir.resolve() / "features.skipped.tsv",
        )
        return
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
