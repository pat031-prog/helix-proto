from __future__ import annotations

import ctypes
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from helix_proto.format import (
    create_store,
    load_full_tensor,
    load_meta,
    load_tensor_rows,
    store_stats,
    streaming_matvec,
)


MANIFEST_FILE = "manifest.json"
_SESSION_CACHES: dict[str, _TensorRuntimeCache] = {}


class _TensorRuntimeCache:
    def __init__(self, *, max_tensor_bytes: int = 256 * 1024) -> None:
        self.max_tensor_bytes = max_tensor_bytes
        self._tensor_cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def tensor(self, store: Path) -> np.ndarray:
        key = str(store)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        tensor = load_full_tensor(store)
        if tensor.nbytes <= self.max_tensor_bytes:
            self._tensor_cache[key] = tensor
        return tensor

    def rows(self, store: Path, indices: list[int]) -> np.ndarray:
        meta = load_meta(store)
        estimated_bytes = int(np.prod(meta.shape)) * np.dtype(meta.dtype).itemsize
        if estimated_bytes <= self.max_tensor_bytes:
            tensor = self.tensor(store)
            return tensor[indices]
        return load_tensor_rows(store, indices)

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._tensor_cache),
            "hits": self.hits,
            "misses": self.misses,
        }


def _cache_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        "entries": after["entries"],
        "hits": after["hits"] - before["hits"],
        "misses": after["misses"] - before["misses"],
    }


def clear_session_runtime_cache(export_dir: str | Path | None = None) -> None:
    if export_dir is None:
        _SESSION_CACHES.clear()
        return
    _SESSION_CACHES.pop(str(Path(export_dir).resolve()), None)


def _resolve_runtime_cache(
    export_dir: str | Path,
    *,
    cache_mode: str,
    max_tensor_bytes: int = 256 * 1024,
) -> _TensorRuntimeCache:
    if cache_mode == "none":
        return _TensorRuntimeCache(max_tensor_bytes=0)
    if cache_mode == "fresh":
        return _TensorRuntimeCache(max_tensor_bytes=max_tensor_bytes)
    if cache_mode == "session":
        key = str(Path(export_dir).resolve())
        cache = _SESSION_CACHES.get(key)
        if cache is None:
            cache = _TensorRuntimeCache(max_tensor_bytes=max_tensor_bytes)
            _SESSION_CACHES[key] = cache
        return cache
    raise ValueError(f"unsupported cache_mode: {cache_mode}")


def _process_rss_mb() -> float:
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if counters.WorkingSetSize <= 0:
            return float("nan")
        return float(counters.WorkingSetSize) / (1024 * 1024)
    except Exception:
        return float("nan")


def _safe_tensor_dir(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "tensor"


def _is_supported_array(array: np.ndarray) -> bool:
    return array.ndim in (1, 2) and np.issubdtype(array.dtype, np.number)


def _normalize_array(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float32)
    return np.asarray(array)


def export_tensor_map(
    tensor_map: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    block_rows: int = 256,
    compression_level: int = 6,
    model_ref: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for name, array in tensor_map.items():
        arr = np.asarray(array)
        if not _is_supported_array(arr):
            skipped.append(
                {
                    "name": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "reason": "only 1D and 2D numeric tensors are exported in this prototype",
                }
            )
            continue

        normalized = _normalize_array(arr)
        tensor_dir = output_dir / "tensors" / _safe_tensor_dir(name)
        create_store(
            normalized,
            tensor_dir,
            block_rows=block_rows,
            compression_level=compression_level,
            extra={
                "tensor_name": name,
                "source": "huggingface",
                "original_dtype": str(arr.dtype),
            },
        )
        transpose_path: str | None = None
        if normalized.ndim == 2:
            transpose_dir = tensor_dir / "transpose"
            create_store(
                normalized.T,
                transpose_dir,
                block_rows=block_rows,
                compression_level=compression_level,
                extra={
                    "tensor_name": name,
                    "source": "huggingface",
                    "transpose_of": name,
                    "original_dtype": str(arr.dtype),
                },
            )
            transpose_path = str(transpose_dir.relative_to(output_dir))

        stats = store_stats(tensor_dir)
        exported.append(
            {
                "name": name,
                "path": str(tensor_dir.relative_to(output_dir)),
                "transpose_path": transpose_path,
                "shape": list(normalized.shape),
                "dtype": str(normalized.dtype),
                "compression_ratio": stats["compression_ratio"],
                "raw_bytes": stats["raw_bytes"],
                "compressed_bytes": stats["compressed_bytes"],
            }
        )

    manifest = {
        "format": "helix-proto-hf-manifest",
        "model_ref": model_ref,
        "config": config or {},
        "block_rows": block_rows,
        "compression_level": compression_level,
        "exported": exported,
        "skipped": skipped,
    }
    (output_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_local_npz(
    npz_path: str | Path,
    output_dir: str | Path,
    *,
    block_rows: int = 256,
) -> dict[str, Any]:
    with np.load(npz_path) as data:
        tensor_map = {key: data[key] for key in data.files}
    return export_tensor_map(
        tensor_map,
        output_dir,
        block_rows=block_rows,
        model_ref=str(npz_path),
        config={"source": "npz"},
    )


def _export_torch_model(
    model: Any,
    config: Any,
    output_dir: str | Path,
    *,
    model_ref: str | None,
    block_rows: int,
    compression_level: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    tensor_map: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            tensor_map[name] = tensor.detach().cpu().numpy()
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return export_tensor_map(
        tensor_map,
        output_dir,
        block_rows=block_rows,
        compression_level=compression_level,
        model_ref=model_ref,
        config=config_dict,
    )


def export_huggingface_model(
    model_ref: str,
    output_dir: str | Path,
    *,
    block_rows: int = 256,
    compression_level: int = 6,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM
    except ImportError as exc:
        raise RuntimeError(
            "convert-hf needs optional dependencies: pip install transformers torch"
        ) from exc

    config = AutoConfig.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    last_error: Exception | None = None
    for model_cls in (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel):
        try:
            model = model_cls.from_pretrained(
                model_ref,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            return _export_torch_model(
                model,
                config,
                output_dir,
                model_ref=model_ref,
                block_rows=block_rows,
                compression_level=compression_level,
            )
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            continue

    raise RuntimeError(f"could not load model {model_ref!r} with available HF loaders") from last_error


def load_manifest(output_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(output_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))


def tensor_store_map(output_dir: str | Path) -> dict[str, Path]:
    manifest = load_manifest(output_dir)
    return {item["name"]: Path(output_dir) / item["path"] for item in manifest["exported"]}


def _layer_norm_last_dim(array: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = array.mean(axis=-1, keepdims=True)
    var = ((array - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (array - mean) / np.sqrt(var + eps)
    return normalized * weight + bias


def _gelu(array: np.ndarray) -> np.ndarray:
    return 0.5 * array * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (array + 0.044715 * (array**3))))


def _softmax(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _streaming_linear_rows(store: Path, hidden_states: np.ndarray, bias: np.ndarray) -> np.ndarray:
    rows = [streaming_matvec(store, row).astype(np.float32) + bias for row in hidden_states]
    return np.stack(rows, axis=0)


def _streaming_right_linear_rows(store: Path, hidden_states: np.ndarray, bias: np.ndarray) -> np.ndarray:
    transpose_store = store / "transpose"
    rows = [streaming_matvec(transpose_store, row).astype(np.float32) + bias for row in hidden_states]
    return np.stack(rows, axis=0)


def _streaming_right_linear_vector(store: Path, hidden_state: np.ndarray, bias: np.ndarray) -> np.ndarray:
    transpose_store = store / "transpose"
    return streaming_matvec(transpose_store, hidden_state).astype(np.float32) + bias


def _bert_embeddings(
    stores: dict[str, Path],
    *,
    token_ids: list[int],
    token_type_ids: list[int],
    eps: float,
) -> np.ndarray:
    word_embeddings = load_tensor_rows(stores["bert.embeddings.word_embeddings.weight"], token_ids)
    pos_embeddings = load_tensor_rows(
        stores["bert.embeddings.position_embeddings.weight"],
        list(range(len(token_ids))),
    )
    type_embeddings = load_tensor_rows(stores["bert.embeddings.token_type_embeddings.weight"], token_type_ids)
    hidden_states = np.stack(
        [
            word_embeddings[position_id] + pos_embeddings[position_id] + type_embeddings[position_id]
            for position_id, _ in enumerate(token_ids)
        ],
        axis=0,
    ).astype(np.float32)
    return _layer_norm_last_dim(
        hidden_states,
        load_full_tensor(stores["bert.embeddings.LayerNorm.weight"]),
        load_full_tensor(stores["bert.embeddings.LayerNorm.bias"]),
        eps,
    ).astype(np.float32)


def _bert_encoder_layer(
    hidden_states: np.ndarray,
    *,
    stores: dict[str, Path],
    layer_index: int,
    num_heads: int,
    eps: float,
) -> np.ndarray:
    layer_prefix = f"bert.encoder.layer.{layer_index}"
    seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads

    q = _streaming_linear_rows(
        stores[f"{layer_prefix}.attention.self.query.weight"],
        hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.attention.self.query.bias"]),
    )
    k = _streaming_linear_rows(
        stores[f"{layer_prefix}.attention.self.key.weight"],
        hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.attention.self.key.bias"]),
    )
    v = _streaming_linear_rows(
        stores[f"{layer_prefix}.attention.self.value.weight"],
        hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.attention.self.value.bias"]),
    )

    q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(float(head_dim))
    probs = _softmax(scores, axis=-1)
    context = np.matmul(probs, v).transpose(1, 0, 2).reshape(seq_len, hidden_size)

    attn_output = _streaming_linear_rows(
        stores[f"{layer_prefix}.attention.output.dense.weight"],
        context,
        load_full_tensor(stores[f"{layer_prefix}.attention.output.dense.bias"]),
    )
    hidden_states = _layer_norm_last_dim(
        attn_output + hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.attention.output.LayerNorm.weight"]),
        load_full_tensor(stores[f"{layer_prefix}.attention.output.LayerNorm.bias"]),
        eps,
    ).astype(np.float32)

    intermediate = _streaming_linear_rows(
        stores[f"{layer_prefix}.intermediate.dense.weight"],
        hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.intermediate.dense.bias"]),
    )
    intermediate = _gelu(intermediate)

    output = _streaming_linear_rows(
        stores[f"{layer_prefix}.output.dense.weight"],
        intermediate,
        load_full_tensor(stores[f"{layer_prefix}.output.dense.bias"]),
    )
    return _layer_norm_last_dim(
        output + hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.output.LayerNorm.weight"]),
        load_full_tensor(stores[f"{layer_prefix}.output.LayerNorm.bias"]),
        eps,
    ).astype(np.float32)


def infer_bert_mlm_logits(
    export_dir: str | Path,
    *,
    token_ids: list[int],
    token_type_ids: list[int] | None = None,
) -> np.ndarray:
    stores = tensor_store_map(export_dir)
    config = load_manifest(export_dir).get("config", {})
    eps = float(config.get("layer_norm_eps", 1e-12))
    num_heads = int(config["num_attention_heads"])
    num_layers = int(config["num_hidden_layers"])

    token_type_ids = token_type_ids or [0] * len(token_ids)
    hidden_states = _bert_embeddings(
        stores,
        token_ids=token_ids,
        token_type_ids=token_type_ids,
        eps=eps,
    )

    for layer_index in range(num_layers):
        hidden_states = _bert_encoder_layer(
            hidden_states,
            stores=stores,
            layer_index=layer_index,
            num_heads=num_heads,
            eps=eps,
        )

    transformed = _streaming_linear_rows(
        stores["cls.predictions.transform.dense.weight"],
        hidden_states,
        load_full_tensor(stores["cls.predictions.transform.dense.bias"]),
    )
    transformed = _gelu(transformed)
    transformed = _layer_norm_last_dim(
        transformed,
        load_full_tensor(stores["cls.predictions.transform.LayerNorm.weight"]),
        load_full_tensor(stores["cls.predictions.transform.LayerNorm.bias"]),
        eps,
    ).astype(np.float32)
    return _streaming_linear_rows(
        stores["cls.predictions.decoder.weight"],
        transformed,
        load_full_tensor(stores["cls.predictions.bias"]),
    )


def infer_zero_layer_bert_mlm_logits(
    export_dir: str | Path,
    *,
    token_id: int,
    position_id: int = 0,
    token_type_id: int = 0,
) -> np.ndarray:
    logits = infer_bert_mlm_logits(
        export_dir,
        token_ids=[token_id],
        token_type_ids=[token_type_id],
    )
    return logits[position_id]


def infer_zero_layer_bert_mlm(
    export_dir: str | Path,
    *,
    token_id: int,
    position_id: int = 0,
    token_type_id: int = 0,
    top_k: int = 5,
) -> dict[str, Any]:
    logits = infer_zero_layer_bert_mlm_logits(
        export_dir,
        token_id=token_id,
        position_id=position_id,
        token_type_id=token_type_id,
    )
    top_indices = np.argsort(logits)[-top_k:][::-1]
    return {
        "token_id": token_id,
        "top_indices": top_indices.tolist(),
        "top_logits": [float(logits[idx]) for idx in top_indices],
        "vocab_size": int(logits.shape[0]),
    }


def infer_one_layer_bert_mlm_logits(
    export_dir: str | Path,
    *,
    token_ids: list[int],
    token_type_ids: list[int] | None = None,
) -> np.ndarray:
    return infer_bert_mlm_logits(export_dir, token_ids=token_ids, token_type_ids=token_type_ids)


def _gpt2_block(
    hidden_states: np.ndarray,
    *,
    stores: dict[str, Path],
    layer_index: int,
    num_heads: int,
    eps: float,
    runtime_cache: _TensorRuntimeCache | None = None,
) -> np.ndarray:
    runtime_cache = runtime_cache or _TensorRuntimeCache()
    prefix = f"transformer.h.{layer_index}"
    seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads

    ln1 = _layer_norm_last_dim(
        hidden_states,
        runtime_cache.tensor(stores[f"{prefix}.ln_1.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_1.bias"]),
        eps,
    ).astype(np.float32)

    attn_proj = _streaming_right_linear_rows(
        stores[f"{prefix}.attn.c_attn.weight"],
        ln1,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_attn.bias"]),
    )
    q, k, v = np.split(attn_proj, 3, axis=-1)
    q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(float(head_dim))
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(causal_mask[None, :, :], -1e9, scores)
    probs = _softmax(scores, axis=-1)
    context = np.matmul(probs, v).transpose(1, 0, 2).reshape(seq_len, hidden_size)

    attn_output = _streaming_right_linear_rows(
        stores[f"{prefix}.attn.c_proj.weight"],
        context,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_proj.bias"]),
    )
    hidden_states = hidden_states + attn_output

    ln2 = _layer_norm_last_dim(
        hidden_states,
        runtime_cache.tensor(stores[f"{prefix}.ln_2.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_2.bias"]),
        eps,
    ).astype(np.float32)
    mlp = _streaming_right_linear_rows(
        stores[f"{prefix}.mlp.c_fc.weight"],
        ln2,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_fc.bias"]),
    )
    mlp = _gelu(mlp)
    mlp = _streaming_right_linear_rows(
        stores[f"{prefix}.mlp.c_proj.weight"],
        mlp,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_proj.bias"]),
    )
    return hidden_states + mlp


def infer_gpt2_causal_lm_logits(
    export_dir: str | Path,
    *,
    token_ids: list[int],
) -> np.ndarray:
    stores = tensor_store_map(export_dir)
    config = load_manifest(export_dir).get("config", {})
    runtime_cache = _TensorRuntimeCache()
    eps = float(config.get("layer_norm_epsilon", 1e-5))
    num_heads = int(config["n_head"])
    num_layers = int(config["n_layer"])

    wte = runtime_cache.rows(stores["transformer.wte.weight"], token_ids)
    wpe = runtime_cache.rows(stores["transformer.wpe.weight"], list(range(len(token_ids))))
    hidden_states = np.stack(
        [wte[position_id] + wpe[position_id] for position_id, _ in enumerate(token_ids)],
        axis=0,
    ).astype(np.float32)

    for layer_index in range(num_layers):
        hidden_states = _gpt2_block(
            hidden_states,
            stores=stores,
            layer_index=layer_index,
            num_heads=num_heads,
            eps=eps,
            runtime_cache=runtime_cache,
        )

    hidden_states = _layer_norm_last_dim(
        hidden_states,
        runtime_cache.tensor(stores["transformer.ln_f.weight"]),
        runtime_cache.tensor(stores["transformer.ln_f.bias"]),
        eps,
    ).astype(np.float32)

    lm_head_store = stores.get("lm_head.weight", stores["transformer.wte.weight"])
    zeros = np.zeros(load_meta(lm_head_store).shape[0], dtype=np.float32)
    return _streaming_linear_rows(lm_head_store, hidden_states, zeros)


def _gpt2_step_with_kv(
    token_embedding: np.ndarray,
    *,
    stores: dict[str, Path],
    layer_index: int,
    num_heads: int,
    eps: float,
    past_k: np.ndarray | None,
    past_v: np.ndarray | None,
    runtime_cache: _TensorRuntimeCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runtime_cache = runtime_cache or _TensorRuntimeCache()
    prefix = f"transformer.h.{layer_index}"
    hidden_size = token_embedding.shape[0]
    head_dim = hidden_size // num_heads

    ln1 = _layer_norm_last_dim(
        token_embedding,
        runtime_cache.tensor(stores[f"{prefix}.ln_1.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_1.bias"]),
        eps,
    ).astype(np.float32)
    attn_proj = _streaming_right_linear_vector(
        stores[f"{prefix}.attn.c_attn.weight"],
        ln1,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_attn.bias"]),
    )
    q, k_new, v_new = np.split(attn_proj, 3)
    q = q.reshape(num_heads, head_dim)
    k_new = k_new.reshape(num_heads, 1, head_dim)
    v_new = v_new.reshape(num_heads, 1, head_dim)

    if past_k is None:
        k_all = k_new
        v_all = v_new
    else:
        k_all = np.concatenate([past_k, k_new], axis=1)
        v_all = np.concatenate([past_v, v_new], axis=1)

    scores = np.einsum("hd,hnd->hn", q, k_all) / np.sqrt(float(head_dim))
    probs = _softmax(scores, axis=-1)
    context = np.einsum("hn,hnd->hd", probs, v_all).reshape(hidden_size)
    attn_output = _streaming_right_linear_vector(
        stores[f"{prefix}.attn.c_proj.weight"],
        context,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_proj.bias"]),
    )
    hidden = token_embedding + attn_output

    ln2 = _layer_norm_last_dim(
        hidden,
        runtime_cache.tensor(stores[f"{prefix}.ln_2.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_2.bias"]),
        eps,
    ).astype(np.float32)
    mlp = _streaming_right_linear_vector(
        stores[f"{prefix}.mlp.c_fc.weight"],
        ln2,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_fc.bias"]),
    )
    mlp = _gelu(mlp)
    mlp = _streaming_right_linear_vector(
        stores[f"{prefix}.mlp.c_proj.weight"],
        mlp,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_proj.bias"]),
    )
    return hidden + mlp, k_all.astype(np.float32), v_all.astype(np.float32)


class GPT2StreamingEngine:
    def __init__(
        self,
        export_dir: str | Path,
        *,
        cache_mode: str = "fresh",
        max_tensor_bytes: int = 256 * 1024,
    ) -> None:
        self.export_dir = Path(export_dir)
        self.stores = tensor_store_map(export_dir)
        self.config = load_manifest(export_dir).get("config", {})
        self.eps = float(self.config.get("layer_norm_epsilon", 1e-5))
        self.num_heads = int(self.config["n_head"])
        self.num_layers = int(self.config["n_layer"])
        self.cache_mode = cache_mode
        self.runtime_cache = _resolve_runtime_cache(
            export_dir,
            cache_mode=cache_mode,
            max_tensor_bytes=max_tensor_bytes,
        )
        self.lm_head_store = self.stores.get("lm_head.weight", self.stores["transformer.wte.weight"])
        self.lm_bias = np.zeros(load_meta(self.lm_head_store).shape[0], dtype=np.float32)
        self.reset_sequence()

    def reset_sequence(self) -> None:
        self.caches: list[dict[str, np.ndarray | None]] = [
            {"k": None, "v": None} for _ in range(self.num_layers)
        ]

    def save_session(
        self,
        session_dir: str | Path,
        *,
        generated_ids: list[int],
        last_logits: np.ndarray | None = None,
    ) -> Path:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "export_dir": str(self.export_dir),
            "cache_mode": self.cache_mode,
            "generated_ids": list(generated_ids),
            "num_layers": self.num_layers,
        }
        (session_dir / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        arrays: dict[str, np.ndarray] = {}
        for layer_index, cache in enumerate(self.caches):
            if cache["k"] is not None:
                arrays[f"layer_{layer_index}_k"] = cache["k"]
            if cache["v"] is not None:
                arrays[f"layer_{layer_index}_v"] = cache["v"]
        if last_logits is not None:
            arrays["last_logits"] = last_logits
        np.savez(session_dir / "kv_cache.npz", **arrays)
        return session_dir

    def load_session(self, session_dir: str | Path) -> dict[str, Any]:
        session_dir = Path(session_dir)
        meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        if Path(meta["export_dir"]).resolve() != self.export_dir.resolve():
            raise ValueError("session export_dir does not match this engine")
        self.reset_sequence()
        cache_path = session_dir / "kv_cache.npz"
        last_logits: np.ndarray | None = None
        if cache_path.exists():
            with np.load(cache_path) as data:
                for layer_index in range(self.num_layers):
                    key_name = f"layer_{layer_index}_k"
                    value_name = f"layer_{layer_index}_v"
                    if key_name in data:
                        self.caches[layer_index]["k"] = data[key_name]
                    if value_name in data:
                        self.caches[layer_index]["v"] = data[value_name]
                if "last_logits" in data:
                    last_logits = data["last_logits"]
        meta["last_logits"] = last_logits
        return meta

    def _run_step(self, token_id: int, position_id: int) -> np.ndarray:
        token_embed = self.runtime_cache.rows(self.stores["transformer.wte.weight"], [token_id])[0]
        pos_embed = self.runtime_cache.rows(self.stores["transformer.wpe.weight"], [position_id])[0]
        hidden = (token_embed + pos_embed).astype(np.float32)
        for layer_index in range(self.num_layers):
            hidden, next_k, next_v = _gpt2_step_with_kv(
                hidden,
                stores=self.stores,
                layer_index=layer_index,
                num_heads=self.num_heads,
                eps=self.eps,
                past_k=self.caches[layer_index]["k"],
                past_v=self.caches[layer_index]["v"],
                runtime_cache=self.runtime_cache,
            )
            self.caches[layer_index]["k"] = next_k
            self.caches[layer_index]["v"] = next_v
        hidden = _layer_norm_last_dim(
            hidden,
            self.runtime_cache.tensor(self.stores["transformer.ln_f.weight"]),
            self.runtime_cache.tensor(self.stores["transformer.ln_f.bias"]),
            self.eps,
        ).astype(np.float32)
        return streaming_matvec(self.lm_head_store, hidden).astype(np.float32) + self.lm_bias

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
        return self.generate_advanced(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            seed=None,
        )

    def stream_generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
        reset_sequence: bool = True,
    ):
        if reset_sequence:
            self.reset_sequence()
        generated = list(prompt_ids)
        step_logits: list[np.ndarray] = []
        rng = np.random.default_rng(seed) if seed is not None else None

        for position_id, token_id in enumerate(prompt_ids):
            logits = self._run_step(token_id, position_id)
            step_logits.append(logits)
            yield {
                "phase": "prompt",
                "position_id": position_id,
                "token_id": int(token_id),
                "generated_ids": list(generated),
                "last_logits": logits,
            }

        for step_index in range(max_new_tokens):
            next_token = self._sample_next_token(
                step_logits[-1],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
            generated.append(next_token)
            logits = self._run_step(next_token, len(generated) - 1)
            step_logits.append(logits)
            yield {
                "phase": "generated",
                "step_index": step_index,
                "position_id": len(generated) - 1,
                "token_id": int(next_token),
                "generated_ids": list(generated),
                "last_logits": logits,
            }

    def _sample_next_token(
        self,
        logits: np.ndarray,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        rng: np.random.Generator | None,
    ) -> int:
        if not do_sample:
            return int(np.argmax(logits))

        scaled = logits.astype(np.float64)
        temp = max(float(temperature), 1e-6)
        scaled = scaled / temp

        if top_k > 0 and top_k < scaled.shape[0]:
            cutoff = np.partition(scaled, -top_k)[-top_k]
            scaled = np.where(scaled >= cutoff, scaled, -1e9)

        probs = _softmax(scaled, axis=-1)

        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            ordered = probs[order]
            cumulative = np.cumsum(ordered)
            keep_mask = cumulative <= top_p
            if not np.any(keep_mask):
                keep_mask[0] = True
            first_exceed = np.argmax(cumulative > top_p)
            if cumulative[first_exceed] > top_p:
                keep_mask[first_exceed] = True
            filtered = np.zeros_like(probs)
            filtered[order[keep_mask]] = probs[order[keep_mask]]
            probs = filtered / filtered.sum()

        rng = rng or np.random.default_rng()
        return int(rng.choice(np.arange(probs.shape[0]), p=probs))

    def generate_advanced(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
        reset_sequence: bool = True,
    ) -> dict[str, Any]:
        if reset_sequence:
            self.reset_sequence()
        cache_before = self.runtime_cache.stats()
        generated = list(prompt_ids)
        step_logits: list[np.ndarray] = []
        step_times_ms: list[float] = []
        step_rss_mb: list[float] = []
        rss_before_mb = _process_rss_mb()
        total_start = time.perf_counter()
        rng = np.random.default_rng(seed) if seed is not None else None

        for position_id, token_id in enumerate(prompt_ids):
            step_start = time.perf_counter()
            step_logits.append(self._run_step(token_id, position_id))
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
            step_rss_mb.append(_process_rss_mb())

        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(
                step_logits[-1],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
            generated.append(next_token)
            step_start = time.perf_counter()
            step_logits.append(self._run_step(next_token, len(generated) - 1))
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
            step_rss_mb.append(_process_rss_mb())

        total_time_s = time.perf_counter() - total_start
        return {
            "prompt_ids": list(prompt_ids),
            "generated_ids": generated,
            "new_ids": generated[len(prompt_ids) :],
            "num_layers": self.num_layers,
            "cache_mode": self.cache_mode,
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "seed": seed,
            "cache_lengths": [
                int(c["k"].shape[1]) if c["k"] is not None else 0 for c in self.caches
            ],
            "runtime_cache": _cache_delta(cache_before, self.runtime_cache.stats()),
            "step_times_ms": [round(value, 3) for value in step_times_ms],
            "avg_step_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
            "total_time_s": total_time_s,
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": _process_rss_mb(),
            "rss_peak_mb": max(step_rss_mb) if step_rss_mb else rss_before_mb,
            "last_logits": step_logits[-1] if step_logits else None,
        }

    def resume_advanced(
        self,
        session_dir: str | Path,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
    ) -> dict[str, Any]:
        meta = self.load_session(session_dir)
        generated_ids = list(meta["generated_ids"])
        if not generated_ids:
            raise ValueError("session contains no generated_ids")
        last_logits = meta.get("last_logits")
        if last_logits is None:
            raise ValueError("session is missing last_logits, cannot resume correctly")

        rng = np.random.default_rng(seed) if seed is not None else None
        next_token = self._sample_next_token(
            np.asarray(last_logits),
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            rng=rng,
        )

        cache_before = self.runtime_cache.stats()
        step_times_ms: list[float] = []
        step_rss_mb: list[float] = []
        rss_before_mb = _process_rss_mb()
        total_start = time.perf_counter()
        generated = list(generated_ids)
        generated.append(next_token)

        step_start = time.perf_counter()
        step_logits = [self._run_step(next_token, len(generated_ids))]
        step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
        step_rss_mb.append(_process_rss_mb())

        for _ in range(max_new_tokens - 1):
            next_token = self._sample_next_token(
                step_logits[-1],
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
                rng=rng,
            )
            generated.append(next_token)
            step_start = time.perf_counter()
            step_logits.append(self._run_step(next_token, len(generated) - 1))
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
            step_rss_mb.append(_process_rss_mb())

        return {
            "resumed_from": generated_ids,
            "generated_ids": generated,
            "new_ids": generated[len(generated_ids) :],
            "cache_mode": self.cache_mode,
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "seed": seed,
            "cache_lengths": [
                int(c["k"].shape[1]) if c["k"] is not None else 0 for c in self.caches
            ],
            "runtime_cache": _cache_delta(cache_before, self.runtime_cache.stats()),
            "step_times_ms": [round(value, 3) for value in step_times_ms],
            "avg_step_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
            "total_time_s": time.perf_counter() - total_start,
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": _process_rss_mb(),
            "rss_peak_mb": max(step_rss_mb) if step_rss_mb else rss_before_mb,
            "last_logits": step_logits[-1] if step_logits else np.asarray(last_logits),
        }


def gpt2_generate_greedy(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    cache_mode: str = "fresh",
) -> dict[str, Any]:
    engine = GPT2StreamingEngine(export_dir, cache_mode=cache_mode)
    return engine.generate(prompt_ids, max_new_tokens)


def gpt2_generate_sample(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    cache_mode: str = "fresh",
) -> dict[str, Any]:
    engine = GPT2StreamingEngine(export_dir, cache_mode=cache_mode)
    return engine.generate_advanced(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


def gpt2_resume_generation(
    export_dir: str | Path,
    session_dir: str | Path,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    cache_mode: str = "session",
) -> dict[str, Any]:
    engine = GPT2StreamingEngine(export_dir, cache_mode=cache_mode)
    return engine.resume_advanced(
        session_dir,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


def benchmark_gpt2_generation_cache(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    clear_session_runtime_cache(export_dir)
    runs = {
        "no_cache": gpt2_generate_greedy(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache_mode="none",
        ),
        "fresh_cache": gpt2_generate_greedy(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache_mode="fresh",
        ),
        "session_cold": gpt2_generate_greedy(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache_mode="session",
        ),
        "session_warm": gpt2_generate_greedy(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache_mode="session",
        ),
    }
    baseline = runs["no_cache"]["total_time_s"]
    summary: dict[str, Any] = {}
    for name, result in runs.items():
        total_time = float(result["total_time_s"])
        summary[name] = {
            "generated_ids": result["generated_ids"],
            "total_time_s": total_time,
            "avg_step_ms": float(result["avg_step_ms"]),
            "runtime_cache": result["runtime_cache"],
            "speedup_vs_no_cache": (baseline / total_time) if total_time else float("inf"),
        }
    return {
        "prompt_ids": list(prompt_ids),
        "max_new_tokens": max_new_tokens,
        "runs": summary,
    }


def benchmark_gpt2_generation_suite(
    export_dir: str | Path,
    *,
    prompt_lengths: list[int],
    max_new_tokens: int,
    warm_repeats: int = 2,
) -> dict[str, Any]:
    config = load_manifest(export_dir).get("config", {})
    vocab_size = int(config["vocab_size"])
    suite: dict[str, Any] = {}

    for prompt_length in prompt_lengths:
        prompt_ids = [((idx * 7) + 3) % vocab_size for idx in range(prompt_length)]
        result = benchmark_gpt2_generation_cache(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
        )

        clear_session_runtime_cache(export_dir)
        session_engine = GPT2StreamingEngine(export_dir, cache_mode="session")
        warm_runs = [session_engine.generate(prompt_ids, max_new_tokens) for _ in range(warm_repeats)]
        result["session_warm_repeats"] = [
            {
                "total_time_s": run["total_time_s"],
                "avg_step_ms": run["avg_step_ms"],
                "runtime_cache": run["runtime_cache"],
                "generated_ids": run["generated_ids"],
            }
            for run in warm_runs
        ]
        result["session_warm_avg"] = {
            "total_time_s": float(np.mean([run["total_time_s"] for run in warm_runs])),
            "avg_step_ms": float(np.mean([run["avg_step_ms"] for run in warm_runs])),
            "generated_ids": warm_runs[-1]["generated_ids"],
        }
        suite[str(prompt_length)] = result

    return {
        "prompt_lengths": list(prompt_lengths),
        "max_new_tokens": max_new_tokens,
        "warm_repeats": warm_repeats,
        "suite": suite,
    }
