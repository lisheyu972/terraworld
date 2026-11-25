#!/usr/bin/env python3
"""Materialize an absolute-path config from one of the bundled templates."""

import argparse
import os
from pathlib import Path

import yaml


def absolute(value: str, base: Path) -> str:
    path = Path(os.path.expanduser(value))
    return str((path if path.is_absolute() else base / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("frame", "nir", "sar", "inpaint", "sr"))
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    with args.template.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["flux_path"] = absolute(os.environ["FLUX_MODEL_DIR"], root)
    transformer_variable = {
        "frame": "FRAME_TRANSFORMER_DIR",
        "sar": "SAR_TRANSFORMER_DIR",
        "inpaint": "INPAINT_TRANSFORMER_DIR",
        "sr": "SR_TRANSFORMER_DIR",
    }.get(args.task)
    if transformer_variable:
        config["transformer_path"] = absolute(os.environ[transformer_variable], root)
    else:
        config.pop("transformer_path", None)

    training = config["train"]
    training["save_path"] = str(root / "runs" / Path(training["save_path"]).name)
    dataset = training["dataset"]
    root_variable = {
        "frame": "FMOW_TRAIN_ROOT",
        "nir": "NIR_DATA_ROOT",
        "sar": "SAR_DATA_ROOT",
        "inpaint": "FMOW_TRAIN_ROOT",
        "sr": "AID_DATA_ROOT",
    }[args.task]
    dataset["root"] = absolute(os.environ[root_variable], root)
    if args.task in ("frame", "inpaint"):
        dataset["index_path"] = absolute(os.environ["FMOW_TRAIN_INDEX"], root)
    if args.task == "inpaint":
        dataset["test_root"] = absolute(os.environ["FMOW_TEST_ROOT"], root)
        dataset["test_index_path"] = absolute(os.environ["FMOW_TEST_INDEX"], root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
