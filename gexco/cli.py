"""One CLI for every GeXCo pipeline stage."""

import sys
import importlib

STAGES = {
    "preprocess-ct": "gexco.data.preprocess_ct",
    "prepare-xray": "gexco.data.radiographs",
    "train-classifier": "gexco.evaluation.train_classifier",
    "make-drr": "gexco.data.drr_rendering",
    "export-hdf5": "gexco.data.export_hdf5",
    "pretrain-wm": "gexco.pretraining.train",
    "extract-latents": "gexco.data.extract_latents",
    "train-gen": "gexco.reconstruction.train",
    "infer": "gexco.reconstruction.inference",
    "evaluate": "gexco.evaluation.run_suite",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("Usage: gexco {" + ",".join(STAGES) + "} [options]")
        return
    command = sys.argv.pop(1)
    if command not in STAGES:
        raise SystemExit(f"Unknown stage: {command}")
    importlib.import_module(STAGES[command]).main()
