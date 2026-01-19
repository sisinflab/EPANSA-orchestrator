import os
import numpy as np
from typing import List
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .io_utils import log, save_parquet
from .features import canonical_text  # ensure canonical_text is present and imported

# Configuration via environment variables (sane defaults)
MODEL_NAME = os.getenv("EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMB_BATCH = int(os.getenv("EMB_BATCH", "64"))
MAX_LEN = int(os.getenv("EMB_MAX_LEN", "256"))
HF_TOKEN = os.getenv("HF_TOKEN", None)  # optional HF token for private models


def cosine_sim(vec_a, vec_b) -> float:
    """
    Computes cosine similarity between two vectors.
    Handles None inputs and zero vectors gracefully.
    """
    if vec_a is None or vec_b is None:
        return 0.0

    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)

    if a.shape != b.shape:
        return 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class TransformersEmbeddingEncoder:
    """
    Simple encoder based on HF AutoModel:

    - Tokenize (padding/truncation)
    - Forward pass -> last_hidden_state
    - Mean pool using attention_mask
    - L2 normalize embeddings

    Parameters
    ----------
    model_name : str
      HF model id (or local path).
    batch_size : int
      number of texts per forward pass.
    device : str
      "cuda" or "cpu" (defaults to cuda if available).
    max_length : int
      tokenizer max length.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        max_length: int | None = None,
    ):
        self.model_name = model_name or MODEL_NAME
        self.batch_size = int(batch_size or EMB_BATCH)
        self.max_length = int(max_length or MAX_LEN)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # HF kwargs — trust_remote_code for some community models, optional auth token
        hf_kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            hf_kwargs["use_auth_token"] = HF_TOKEN

        # When CUDA available try to use float16 to reduce memory footprint
        load_kwargs = {}
        if "cuda" in self.device:
            try:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["low_cpu_mem_usage"] = True
            except Exception:
                pass

        log.info(
            f"Loading tokenizer & model '{self.model_name}' on device={self.device} (batch={self.batch_size})"
        )
        self.tok = AutoTokenizer.from_pretrained(self.model_name, **hf_kwargs)
        # Model loading: for very large models you may want device_map="auto" and accelerate/transformers 4.30+
        try:
            self.mdl = AutoModel.from_pretrained(
                self.model_name, **load_kwargs, **hf_kwargs
            ).to(self.device)
        except Exception as e:
            # fallback: try without some load kwargs
            log.warning(
                f"Primary model load failed: {e}. Retrying with fallback settings."
            )
            self.mdl = AutoModel.from_pretrained(self.model_name, **hf_kwargs).to(
                self.device
            )
        self.mdl.eval()

    def _encode_batch(self, batch_texts: List[str]) -> np.ndarray:
        """Tokenize and encode one batch, returning numpy array [B, D]."""
        enc = self.tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.mdl(**enc)
            # Many encoder models expose last_hidden_state
            if hasattr(out, "last_hidden_state"):
                hs = out.last_hidden_state  # [B, T, H]
            else:
                # fallback for unusual model outputs
                hs = out[0]
            mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts and return an array of shape [N, D]."""
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        out_parts = []
        bs = self.batch_size
        for i in tqdm(range(0, len(texts), bs), desc="Embedding"):
            batch = texts[i : i + bs]
            try:
                out_parts.append(self._encode_batch(batch))
            except RuntimeError as e:
                # OOM or other runtime error: try a smaller batch size recursively
                log.warning(
                    f"RuntimeError while encoding batch size {len(batch)}: {e}. Retrying with smaller batch."
                )
                if bs <= 1:
                    raise
                # reduce batch (half) and reattempt encoding remaining items
                self.batch_size = max(1, self.batch_size // 2)
                return self.encode_texts(texts)  # re-run with smaller batch
        return np.vstack(out_parts)


QwenEmbeddingEncoder = TransformersEmbeddingEncoder


def save_venue_embeddings(
    venue_features_df: pd.DataFrame,
    out_path: str,
    model: TransformersEmbeddingEncoder | None = None,
    batch_size: int | None = None,
    embedding_col: str = "embedding",
) -> pd.DataFrame:
    """
    High-level helper:
     - Builds canonical texts from venue_features_df using canonical_text()
     - Encodes them with the provided model (or default TransformersEmbeddingEncoder)
     - Writes a parquet file with columns: venue_id, embedding (list[float])

    Returns the DataFrame that was written.
    """
    encoder = model or TransformersEmbeddingEncoder(batch_size=batch_size)
    log.info(
        f"Encoding {len(venue_features_df)} venues with model={encoder.model_name}"
    )

    # Build texts (order must match venue_id array)
    texts = venue_features_df.apply(canonical_text, axis=1).tolist()

    # Encode
    embs = encoder.encode_texts(texts)  # numpy [N, D]

    # Prepare output DataFrame with embedding column as list[float]
    emb_list = [row.tolist() for row in embs]
    out_df = pd.DataFrame(
        {"venue_id": venue_features_df["venue_id"].values, embedding_col: emb_list}
    )

    # Save parquet (using src.io_utils.save_parquet to keep behaviour consistent)
    try:
        save_parquet(out_df, out_path)
    except Exception as e:
        # fallback: pandas native write
        log.warning(f"save_parquet failed ({e}), falling back to pandas.to_parquet")
        out_df.to_parquet(out_path, index=False)

    log.info(f"Saved embeddings for {len(out_df)} venues -> {out_path}")
    return out_df
