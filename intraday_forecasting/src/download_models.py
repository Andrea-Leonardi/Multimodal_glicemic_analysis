"""Download only public model weights/configs. No patient data are read/sent."""

import os

from src.data import ROOT, read_config


def main():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/hf-xet"))
    from huggingface_hub import snapshot_download
    config = read_config()
    for name, revision in config["model_revisions"].items():
        repo = "mistralai/" + {"timestral-28m": "Timestral-28M", "timestral-209m": "Timestral-209M"}[name]
        path = snapshot_download(repo, revision=revision, allow_patterns=["config.json", "model.safetensors", "LICENSE*", "README.md"],
                                 cache_dir=ROOT / ".cache/huggingface")
        print(repo, revision, path, flush=True)


if __name__ == "__main__":
    main()
