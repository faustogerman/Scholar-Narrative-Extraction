"""Regenerate the pre-computed document embeddings used by the demo.

The embedding pickles are large and are *not* committed to the repository, so
run this script once before launching the app to (re)generate them:

    python extract_embeddings.py            # all datasets
    python extract_embeddings.py --dataset vispub
    python extract_embeddings.py --force    # overwrite existing pickles

For each dataset it embeds ``"<title>; <abstract>"`` for every document with
OpenAI ``text-embedding-3-small`` and writes an ``embed_data-gpt4.pickle`` into
that dataset's embeddings folder, row-aligned with its ``text_data.csv``. This
alignment is what lets the pipeline match embedding row i to document i.

Requires OPENAI_API_KEY (see .env.example).
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import torch
from openai import OpenAI
from tqdm import tqdm

from demo_datasets import DATASETS, DatasetConfig

EMBEDDING_MODEL = "text-embedding-3-small"
PICKLE_NAME = "embed_data-gpt4.pickle"


def _load_dotenv() -> None:
    """Load OPENAI_API_KEY from a .env file next to this script, if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_openai_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in response.data]


def batched_embeddings(client: OpenAI, texts: list[str],
                       batch_size: int = 16) -> torch.Tensor:
    embeddings: list[list[float]] = []
    for i in tqdm(range(0, len(texts), batch_size)):
        embeddings.extend(get_openai_embeddings(client, texts[i:i + batch_size]))
        time.sleep(0.75)  # Avoid making too many requests too fast.
    return torch.tensor(embeddings)


def extract_for_dataset(client: OpenAI, cfg: DatasetConfig,
                        batch_size: int, force: bool) -> None:
    out_path = Path(cfg.embed_folder) / PICKLE_NAME
    if out_path.exists() and not force:
        print(f"[{cfg.key}] {out_path} already exists — skipping (use --force to overwrite).")
        return

    df = cfg.loader()
    texts = (df[cfg.title_col].astype(str) + "; " + df[cfg.abstract_col].astype(str)).tolist()
    print(f"[{cfg.key}] embedding {len(texts)} documents → {out_path}")

    embeddings = batched_embeddings(client, texts, batch_size=batch_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{cfg.key}] wrote {tuple(embeddings.shape)} embeddings to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=[*DATASETS.keys(), "all"], default="all",
        help="Which dataset to (re)generate embeddings for (default: all).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Documents per OpenAI request (default: 16).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing embedding pickles.")
    args = parser.parse_args()

    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Add it to a .env file (see .env.example).")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for key in keys:
        extract_for_dataset(client, DATASETS[key], args.batch_size, args.force)


if __name__ == "__main__":
    main()
