"""Download the surface/concrete crack dataset (train/validation/test) as parquet.

The dataset is a Hugging Face mirror of Ozgenel's "Concrete Crack Images for
Classification" (Mendeley Data, DOI 10.17632/5y9wdsg2zt.2) - the same 40,000
image, 2-class (Positive/Negative) dataset described in the project report,
also referenced by the arthurflor23/surface-crack-detection GitHub repo.
"""

import os

from huggingface_hub import hf_hub_download

from config import DATA_RAW_DIR, HF_DATASET_REPO, HF_REVISION


def download_all_splits():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    paths = {}
    for split in ["train", "validation", "test"]:
        path = hf_hub_download(
            HF_DATASET_REPO,
            filename=f"default/{split}/0000.parquet",
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=DATA_RAW_DIR,
        )
        paths[split] = path
        print(f"[download_data] {split}: {path}")
    return paths


if __name__ == "__main__":
    download_all_splits()
