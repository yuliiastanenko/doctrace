import time
from functools import lru_cache

import numpy as np

from doctrace.config import settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model(name: str):
    from fastembed import TextEmbedding

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(name, cache_dir=str(settings.models_dir))


def _prefixes(name: str) -> tuple[str, str]:
    lower = name.lower()
    if "e5" in lower:
        return "query: ", "passage: "
    if "bge" in lower and "-en" in lower:
        return BGE_QUERY_PREFIX, ""
    return "", ""


def embed_passages(texts: list[str], name: str, batch_size: int = 16) -> np.ndarray:
    _, prefix = _prefixes(name)
    model = _model(name)
    vectors = []
    start = time.time()
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i : i + batch_size]]
        vectors.extend(model.embed(batch, batch_size=batch_size))
        done = min(i + batch_size, len(texts))
        elapsed = time.time() - start
        eta = elapsed / done * (len(texts) - done)
        print(f"\r  embedded {done}/{len(texts)}  ({elapsed:.0f}s, ~{eta:.0f}s left)", end="", flush=True)
    print()
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str, name: str) -> np.ndarray:
    prefix, _ = _prefixes(name)
    return np.array(next(iter(_model(name).embed([prefix + text]))), dtype=np.float32)
