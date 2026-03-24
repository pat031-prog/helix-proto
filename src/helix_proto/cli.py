from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from helix_proto.api import HelixRuntime, serve_api
from helix_proto.format import create_store, store_stats, streaming_matvec, verify_store
from helix_proto.hf import (
    benchmark_gpt2_generation_suite,
    benchmark_gpt2_generation_cache,
    export_huggingface_model,
    export_local_npz,
    GPT2StreamingEngine,
    gpt2_generate_greedy,
    gpt2_generate_sample,
    gpt2_resume_generation,
    infer_bert_mlm_logits,
    infer_gpt2_causal_lm_logits,
    infer_one_layer_bert_mlm_logits,
    infer_zero_layer_bert_mlm,
    infer_zero_layer_bert_mlm_logits,
)
from helix_proto.text import save_toy_tokenizer
from helix_proto.workspace import slugify, workspace_root


def _json_ready(value):
    if isinstance(value, dict):
        cleaned = dict(value)
        last_logits = cleaned.pop("last_logits", None)
        if isinstance(last_logits, np.ndarray):
            cleaned["last_logits_shape"] = list(last_logits.shape)
        return {key: _json_ready(item) for key, item in cleaned.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_json(payload) -> None:
    print(json.dumps(_json_ready(payload), indent=2))


def _default_session_id(alias: str) -> str:
    return f"{slugify(alias)}-{int(time.time())}"


def _cmd_convert(args: argparse.Namespace) -> int:
    matrix = np.load(args.input)
    create_store(matrix, args.output, block_rows=args.block_rows, compression_level=args.level)
    stats = store_stats(args.output)
    print(f"created store at {args.output}")
    print(
        f"shape={stats['rows']}x{stats['cols']} blocks={stats['blocks']} "
        f"ratio={stats['compression_ratio']:.2f}x"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    results = verify_store(args.store)
    print(f"verified {len(results)} blocks")
    return 0


def _cmd_matvec(args: argparse.Namespace) -> int:
    vector = np.load(args.vector)
    result = streaming_matvec(args.store, vector)
    if args.output:
        np.save(args.output, result)
        print(f"saved output to {args.output}")
    else:
        print(result)
    return 0


def _make_demo_matrix(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.15, size=(rows, cols)).astype(np.float32)
    mask = rng.random((rows, cols)) > 0.78
    base[mask] = 0.0
    return base


def _run_benchmark(rows: int, cols: int, block_rows: int, seed: int) -> dict[str, float]:
    matrix = _make_demo_matrix(rows, cols, seed)
    vector = np.random.default_rng(seed + 1).normal(0.0, 0.15, size=(cols,)).astype(np.float32)

    workdir = Path(tempfile.mkdtemp(prefix="helix-proto-"))
    try:
        store_dir = workdir / "matrix.cdna"
        dense_start = time.perf_counter()
        dense = matrix @ vector
        dense_time = time.perf_counter() - dense_start

        create_store(matrix, store_dir, block_rows=block_rows)
        stream_start = time.perf_counter()
        streamed = streaming_matvec(store_dir, vector)
        stream_time = time.perf_counter() - stream_start

        stats = store_stats(store_dir)
        max_abs_err = float(np.max(np.abs(dense - streamed)))
        return {
            "dense_time_s": dense_time,
            "stream_time_s": stream_time,
            "compression_ratio": stats["compression_ratio"],
            "raw_mb": stats["raw_bytes"] / (1024 * 1024),
            "compressed_mb": stats["compressed_bytes"] / (1024 * 1024),
            "max_abs_err": max_abs_err,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    result = _run_benchmark(args.rows, args.cols, args.block_rows, args.seed)
    for key, value in result.items():
        if key.endswith("_s"):
            print(f"{key}={value:.6f}")
        elif key.endswith("_mb"):
            print(f"{key}={value:.2f}")
        else:
            print(f"{key}={value:.6f}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("demo-output").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    matrix = _make_demo_matrix(args.rows, args.cols, args.seed)
    vector = np.random.default_rng(args.seed + 1).normal(0.0, 0.15, size=(args.cols,)).astype(
        np.float32
    )
    np.save(workdir / "matrix.npy", matrix)
    np.save(workdir / "vector.npy", vector)

    create_store(matrix, workdir / "matrix.cdna", block_rows=args.block_rows)
    verify_store(workdir / "matrix.cdna")

    dense = matrix @ vector
    streamed = streaming_matvec(workdir / "matrix.cdna", vector)
    max_abs_err = float(np.max(np.abs(dense - streamed)))
    stats = store_stats(workdir / "matrix.cdna")

    print(f"demo written to {workdir}")
    print(f"compression_ratio={stats['compression_ratio']:.2f}x")
    print(f"max_abs_err={max_abs_err:.8f}")
    return 0


def _cmd_convert_hf(args: argparse.Namespace) -> int:
    manifest = export_huggingface_model(
        args.model_ref,
        args.output,
        block_rows=args.block_rows,
        compression_level=args.level,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"exported {len(manifest['exported'])} tensors to {args.output}")
    print(f"skipped {len(manifest['skipped'])} tensors")
    print(json.dumps({"manifest": str(Path(args.output) / 'manifest.json')}, indent=2))
    return 0


def _cmd_convert_npz(args: argparse.Namespace) -> int:
    manifest = export_local_npz(args.input, args.output, block_rows=args.block_rows)
    print(f"exported {len(manifest['exported'])} tensors to {args.output}")
    print(f"skipped {len(manifest['skipped'])} tensors")
    return 0


def _cmd_build_tiny_bert(args: argparse.Namespace) -> int:
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=max(1, args.hidden_size // 8),
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
    )
    model = BertForMaskedLM(config)
    model.save_pretrained(args.output)
    print(f"saved tiny bert with {args.num_hidden_layers} layers to {args.output}")
    return 0


def _cmd_build_tiny_gpt2(args: argparse.Namespace) -> int:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        n_positions=args.max_position_embeddings,
        n_ctx=args.max_position_embeddings,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(args.output)
    tokenizer_info = save_toy_tokenizer(args.output, vocab_size=args.vocab_size)
    print(f"saved tiny gpt2 with {args.num_layers} layers to {args.output}")
    print(f"saved toy tokenizer to {tokenizer_info['tokenizer_dir']}")
    return 0


def _cmd_demo_bert_block(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForMaskedLM

    workdir = Path(args.output).resolve() if args.output else Path("bert-block-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-bert-1layer"
    export_dir = workdir / "export"
    _cmd_build_tiny_bert(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            type_vocab_size=args.type_vocab_size,
            num_hidden_layers=1,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    model = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    input_ids = torch.tensor([args.token_ids], dtype=torch.long)
    token_type_ids = torch.tensor([[0] * len(args.token_ids)], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
    hf_logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_one_layer_bert_mlm_logits(export_dir, token_ids=args.token_ids)
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))
    print(f"demo workspace written to {workdir}")
    print(f"seq_len={len(args.token_ids)}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_top_indices_pos0={np.argsort(our_logits[0])[-args.top_k:][::-1].tolist()}")
    print(f"hf_top_indices_pos0={np.argsort(hf_logits[0])[-args.top_k:][::-1].tolist()}")
    return 0


def _cmd_demo_bert_stack(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForMaskedLM

    workdir = Path(args.output).resolve() if args.output else Path("bert-stack-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-bert-stack"
    export_dir = workdir / "export"
    _cmd_build_tiny_bert(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            type_vocab_size=args.type_vocab_size,
            num_hidden_layers=args.num_hidden_layers,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    model = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    input_ids = torch.tensor([args.token_ids], dtype=torch.long)
    token_type_ids = torch.tensor([[0] * len(args.token_ids)], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
    hf_logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_bert_mlm_logits(export_dir, token_ids=args.token_ids)
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))
    print(f"demo workspace written to {workdir}")
    print(f"layers={args.num_hidden_layers}")
    print(f"seq_len={len(args.token_ids)}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_top_indices_last={np.argsort(our_logits[-1])[-args.top_k:][::-1].tolist()}")
    print(f"hf_top_indices_last={np.argsort(hf_logits[-1])[-args.top_k:][::-1].tolist()}")
    return 0


def _cmd_demo_hf_infer(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForMaskedLM

    workdir = Path(args.output).resolve() if args.output else Path("hf-infer-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-bert"
    export_dir = workdir / "export"
    _cmd_build_tiny_bert(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            type_vocab_size=args.type_vocab_size,
            num_hidden_layers=0,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    model = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor([[args.token_id]], dtype=torch.long),
            token_type_ids=torch.tensor([[args.token_type_id]], dtype=torch.long),
        )
    hf_logits = outputs.logits[0, 0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_zero_layer_bert_mlm_logits(
        export_dir,
        token_id=args.token_id,
        token_type_id=args.token_type_id,
    )
    top = infer_zero_layer_bert_mlm(
        export_dir,
        token_id=args.token_id,
        token_type_id=args.token_type_id,
        top_k=args.top_k,
    )
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))
    print(f"demo workspace written to {workdir}")
    print(f"exported_tensors={len(json.loads((export_dir / 'manifest.json').read_text(encoding='utf-8'))['exported'])}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_top_indices={top['top_indices']}")
    print(f"hf_top_indices={np.argsort(hf_logits)[-args.top_k:][::-1].tolist()}")
    return 0


def _cmd_demo_gpt_causal(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    workdir = Path(args.output).resolve() if args.output else Path("gpt-causal-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    input_ids = torch.tensor([args.token_ids], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    hf_logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_gpt2_causal_lm_logits(export_dir, token_ids=args.token_ids)
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))
    print(f"demo workspace written to {workdir}")
    print(f"layers={args.num_layers}")
    print(f"seq_len={len(args.token_ids)}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_top_indices_last={np.argsort(our_logits[-1])[-args.top_k:][::-1].tolist()}")
    print(f"hf_top_indices_last={np.argsort(hf_logits[-1])[-args.top_k:][::-1].tolist()}")
    return 0


def _cmd_demo_gpt_generate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    workdir = Path(args.output).resolve() if args.output else Path("gpt-generate-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    ours = gpt2_generate_greedy(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
    )

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    with torch.no_grad():
        hf_output = model.generate(
            input_ids=torch.tensor([args.prompt_ids], dtype=torch.long),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
        )
    hf_ids = hf_output[0].detach().cpu().tolist()

    print(f"demo workspace written to {workdir}")
    print(f"prompt_ids={args.prompt_ids}")
    print(f"our_generated_ids={ours['generated_ids']}")
    print(f"hf_generated_ids={hf_ids}")
    print(f"cache_lengths={ours['cache_lengths']}")
    print(f"runtime_cache={ours['runtime_cache']}")
    print(f"step_times_ms={ours['step_times_ms']}")
    print(f"avg_step_ms={ours['avg_step_ms']:.3f}")
    print(f"total_time_s={ours['total_time_s']:.6f}")
    print(
        f"rss_mb_before={ours['rss_before_mb']:.2f} "
        f"rss_mb_after={ours['rss_after_mb']:.2f} "
        f"rss_mb_peak={ours['rss_peak_mb']:.2f}"
    )
    print(f"match={ours['generated_ids'] == hf_ids}")
    return 0


def _cmd_demo_gpt_sample(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-sample-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    sampled = gpt2_generate_sample(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
    )
    print(f"demo workspace written to {workdir}")
    print(f"prompt_ids={args.prompt_ids}")
    print(f"generated_ids={sampled['generated_ids']}")
    print(f"runtime_cache={sampled['runtime_cache']}")
    print(f"step_times_ms={sampled['step_times_ms']}")
    return 0


def _cmd_demo_gpt_resume(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-resume-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    session_dir = workdir / "session"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    engine = GPT2StreamingEngine(export_dir, cache_mode="session")
    first_part = engine.generate_advanced(
        args.prompt_ids,
        max_new_tokens=args.first_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    engine.save_session(session_dir, generated_ids=first_part["generated_ids"], last_logits=first_part["last_logits"])
    resumed = gpt2_resume_generation(
        export_dir,
        session_dir,
        max_new_tokens=args.second_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode="session",
    )

    if args.do_sample:
        full = gpt2_generate_sample(
            export_dir,
            prompt_ids=args.prompt_ids,
            max_new_tokens=args.first_new_tokens + args.second_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            cache_mode="fresh",
        )
    else:
        full = gpt2_generate_greedy(
            export_dir,
            prompt_ids=args.prompt_ids,
            max_new_tokens=args.first_new_tokens + args.second_new_tokens,
            cache_mode="fresh",
        )

    print(f"demo workspace written to {workdir}")
    print(f"first_part_ids={first_part['generated_ids']}")
    print(f"resumed_ids={resumed['generated_ids']}")
    print(f"full_ids={full['generated_ids']}")
    print(f"match={resumed['generated_ids'] == full['generated_ids']}")
    return 0


def _cmd_demo_gpt_remote(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    workdir = Path(args.output).resolve() if args.output else Path("gpt-remote-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    export_dir = workdir / "export"
    export_huggingface_model(
        args.model_ref,
        export_dir,
        block_rows=args.block_rows,
        local_files_only=False,
        trust_remote_code=args.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_ref,
        local_files_only=False,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    input_ids = torch.tensor([args.prompt_ids], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    hf_logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_gpt2_causal_lm_logits(export_dir, token_ids=args.prompt_ids)
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))

    ours = gpt2_generate_greedy(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        cache_mode="fresh",
    )
    with torch.no_grad():
        hf_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
    hf_ids = hf_output[0].detach().cpu().tolist()

    print(f"demo workspace written to {workdir}")
    print(f"model_ref={args.model_ref}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_generated_ids={ours['generated_ids']}")
    print(f"hf_generated_ids={hf_ids}")
    print(f"match={ours['generated_ids'] == hf_ids}")
    return 0


def _cmd_benchmark_gpt_cache(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-cache-benchmark").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    benchmark = benchmark_gpt2_generation_cache(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"demo workspace written to {workdir}")
    print(f"prompt_ids={benchmark['prompt_ids']}")
    print(f"max_new_tokens={benchmark['max_new_tokens']}")
    for name, result in benchmark["runs"].items():
        print(
            f"{name}: total_time_s={result['total_time_s']:.6f} "
            f"avg_step_ms={result['avg_step_ms']:.3f} "
            f"speedup_vs_no_cache={result['speedup_vs_no_cache']:.3f} "
            f"runtime_cache={result['runtime_cache']}"
        )
    return 0


def _cmd_benchmark_gpt_suite(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-suite-benchmark").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    report = benchmark_gpt2_generation_suite(
        export_dir,
        prompt_lengths=args.prompt_lengths,
        max_new_tokens=args.max_new_tokens,
        warm_repeats=args.warm_repeats,
    )
    report_path = workdir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"demo workspace written to {workdir}")
    print(f"report={report_path}")
    for prompt_length in args.prompt_lengths:
        entry = report["suite"][str(prompt_length)]
        no_cache = entry["runs"]["no_cache"]
        fresh = entry["runs"]["fresh_cache"]
        warm = entry["session_warm_avg"]
        print(
            f"prompt_len={prompt_length}: "
            f"no_cache={no_cache['total_time_s']:.6f}s "
            f"fresh_cache={fresh['total_time_s']:.6f}s "
            f"session_warm_avg={warm['total_time_s']:.6f}s"
        )
    return 0


def _cmd_prepare_model(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    info = runtime.prepare_model(
        model_ref=args.model_ref,
        alias=args.alias,
        block_rows=args.block_rows,
        trust_remote_code=args.trust_remote_code,
        force=args.force,
    )
    _print_json(info)
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    _print_json(
        {
            "workspace_root": workspace_root(args.workspace_root),
            "models": runtime.list_models(),
        }
    )
    return 0


def _cmd_model_info(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    _print_json(runtime.model_info(args.alias))
    return 0


def _cmd_run_gpt(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    session_id = args.session_id
    if args.save_session and not session_id:
        session_id = _default_session_id(args.alias)
    result = runtime.generate(
        alias=args.alias,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        session_id=session_id if (args.save_session or session_id) else None,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved generation result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_run_gpt_text(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    session_id = args.session_id
    if args.save_session and not session_id:
        session_id = _default_session_id(args.alias)
    messages = None
    if args.system_prompt or args.user_prompt:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        if args.user_prompt:
            messages.append({"role": "user", "content": args.user_prompt})
    result = runtime.generate_text(
        alias=args.alias,
        prompt=args.prompt,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        session_id=session_id if (args.save_session or session_id) else None,
        add_special_tokens=args.add_special_tokens,
        skip_special_tokens=not args.keep_special_tokens,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved text generation result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_resume_gpt(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    result = runtime.resume(
        alias=args.alias,
        session_id=args.session_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        save_session=not args.no_save_session,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved resumed result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_resume_gpt_text(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    result = runtime.resume_text(
        alias=args.alias,
        session_id=args.session_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        save_session=not args.no_save_session,
        skip_special_tokens=not args.keep_special_tokens,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved resumed text result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_list_tools(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    _print_json({"tools": runtime.tool_manifest()})
    return 0


def _cmd_call_tool(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    arguments = json.loads(args.arguments) if args.arguments else {}
    result = runtime.call_tool(args.tool_name, arguments)
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved tool result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_agent_add_text(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    runner = runtime.agent_runner()
    result = runner.add_knowledge_text(
        args.agent_name,
        args.text,
        source=args.source,
        metadata=json.loads(args.metadata) if args.metadata else None,
    )
    _print_json(result)
    return 0


def _cmd_agent_add_file(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    runner = runtime.agent_runner()
    result = runner.add_knowledge_file(
        args.agent_name,
        args.file_path,
        metadata=json.loads(args.metadata) if args.metadata else None,
    )
    _print_json(result)
    return 0


def _cmd_agent_search(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    runner = runtime.agent_runner()
    result = runner.search_knowledge(args.agent_name, args.query, top_k=args.top_k)
    _print_json(result)
    return 0


def _cmd_agent_run(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    runner = runtime.agent_runner()
    result = runner.run(
        goal=args.goal,
        agent_name=args.agent_name,
        default_model_alias=args.default_model_alias,
        local_planner_alias=args.local_planner_alias,
        remote_model_ref=args.remote_model_ref,
        prefer_remote=args.prefer_remote,
        trust_remote_code=args.trust_remote_code,
        max_steps=args.max_steps,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved agent run to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_serve_api(args: argparse.Namespace) -> int:
    root = workspace_root(args.workspace_root)
    print(f"serving helix api on http://{args.host}:{args.port}")
    print(f"workspace_root={root}")
    serve_api(host=args.host, port=args.port, root=root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Block-compressed streaming tensor prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert a .npy matrix into the block store.")
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--block-rows", type=int, default=256)
    convert.add_argument("--level", type=int, default=6)
    convert.set_defaults(func=_cmd_convert)

    verify = subparsers.add_parser("verify", help="Verify all compressed blocks.")
    verify.add_argument("store", type=Path)
    verify.set_defaults(func=_cmd_verify)

    matvec = subparsers.add_parser("matvec", help="Run streaming matrix-vector multiply.")
    matvec.add_argument("store", type=Path)
    matvec.add_argument("vector", type=Path)
    matvec.add_argument("--output", type=Path)
    matvec.set_defaults(func=_cmd_matvec)

    benchmark = subparsers.add_parser("benchmark", help="Compare dense and streaming execution.")
    benchmark.add_argument("--rows", type=int, default=4096)
    benchmark.add_argument("--cols", type=int, default=2048)
    benchmark.add_argument("--block-rows", type=int, default=256)
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.set_defaults(func=_cmd_benchmark)

    demo = subparsers.add_parser("demo", help="Create a full local demo workspace.")
    demo.add_argument("--rows", type=int, default=1024)
    demo.add_argument("--cols", type=int, default=512)
    demo.add_argument("--block-rows", type=int, default=128)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", type=Path)
    demo.set_defaults(func=_cmd_demo)

    convert_hf = subparsers.add_parser(
        "convert-hf",
        help="Load a Hugging Face model and export supported tensors into block stores.",
    )
    convert_hf.add_argument("model_ref")
    convert_hf.add_argument("output", type=Path)
    convert_hf.add_argument("--block-rows", type=int, default=256)
    convert_hf.add_argument("--level", type=int, default=6)
    convert_hf.add_argument("--local-files-only", action="store_true")
    convert_hf.add_argument("--trust-remote-code", action="store_true")
    convert_hf.set_defaults(func=_cmd_convert_hf)

    convert_npz = subparsers.add_parser(
        "convert-npz",
        help="Export a local .npz tensor bundle through the Hugging Face-style manifest pipeline.",
    )
    convert_npz.add_argument("input", type=Path)
    convert_npz.add_argument("output", type=Path)
    convert_npz.add_argument("--block-rows", type=int, default=256)
    convert_npz.set_defaults(func=_cmd_convert_npz)

    build_tiny_bert = subparsers.add_parser(
        "build-tiny-bert",
        help="Create a tiny local Hugging Face masked LM for controlled export tests.",
    )
    build_tiny_bert.add_argument("output", type=Path)
    build_tiny_bert.add_argument("--vocab-size", type=int, default=64)
    build_tiny_bert.add_argument("--hidden-size", type=int, default=32)
    build_tiny_bert.add_argument("--max-position-embeddings", type=int, default=16)
    build_tiny_bert.add_argument("--type-vocab-size", type=int, default=2)
    build_tiny_bert.add_argument("--num-hidden-layers", type=int, default=0)
    build_tiny_bert.set_defaults(func=_cmd_build_tiny_bert)

    build_tiny_gpt2 = subparsers.add_parser(
        "build-tiny-gpt2",
        help="Create a tiny local GPT2 causal LM for export tests.",
    )
    build_tiny_gpt2.add_argument("output", type=Path)
    build_tiny_gpt2.add_argument("--vocab-size", type=int, default=64)
    build_tiny_gpt2.add_argument("--hidden-size", type=int, default=32)
    build_tiny_gpt2.add_argument("--max-position-embeddings", type=int, default=16)
    build_tiny_gpt2.add_argument("--num-layers", type=int, default=2)
    build_tiny_gpt2.add_argument("--num-heads", type=int, default=4)
    build_tiny_gpt2.set_defaults(func=_cmd_build_tiny_gpt2)

    demo_hf_infer = subparsers.add_parser(
        "demo-hf-infer",
        help="Build, export, and validate a first real inference path on exported weights.",
    )
    demo_hf_infer.add_argument("--output", type=Path)
    demo_hf_infer.add_argument("--vocab-size", type=int, default=64)
    demo_hf_infer.add_argument("--hidden-size", type=int, default=32)
    demo_hf_infer.add_argument("--max-position-embeddings", type=int, default=16)
    demo_hf_infer.add_argument("--type-vocab-size", type=int, default=2)
    demo_hf_infer.add_argument("--block-rows", type=int, default=8)
    demo_hf_infer.add_argument("--token-id", type=int, default=7)
    demo_hf_infer.add_argument("--token-type-id", type=int, default=0)
    demo_hf_infer.add_argument("--top-k", type=int, default=5)
    demo_hf_infer.set_defaults(func=_cmd_demo_hf_infer)

    demo_bert_block = subparsers.add_parser(
        "demo-bert-block",
        help="Build, export, and validate a full first BERT transformer block on exported weights.",
    )
    demo_bert_block.add_argument("--output", type=Path)
    demo_bert_block.add_argument("--vocab-size", type=int, default=64)
    demo_bert_block.add_argument("--hidden-size", type=int, default=32)
    demo_bert_block.add_argument("--max-position-embeddings", type=int, default=16)
    demo_bert_block.add_argument("--type-vocab-size", type=int, default=2)
    demo_bert_block.add_argument("--block-rows", type=int, default=8)
    demo_bert_block.add_argument("--token-ids", type=int, nargs="+", default=[7, 11])
    demo_bert_block.add_argument("--top-k", type=int, default=5)
    demo_bert_block.set_defaults(func=_cmd_demo_bert_block)

    demo_bert_stack = subparsers.add_parser(
        "demo-bert-stack",
        help="Build, export, and validate a multi-layer BERT masked LM on exported weights.",
    )
    demo_bert_stack.add_argument("--output", type=Path)
    demo_bert_stack.add_argument("--vocab-size", type=int, default=64)
    demo_bert_stack.add_argument("--hidden-size", type=int, default=32)
    demo_bert_stack.add_argument("--max-position-embeddings", type=int, default=16)
    demo_bert_stack.add_argument("--type-vocab-size", type=int, default=2)
    demo_bert_stack.add_argument("--num-hidden-layers", type=int, default=2)
    demo_bert_stack.add_argument("--block-rows", type=int, default=8)
    demo_bert_stack.add_argument("--token-ids", type=int, nargs="+", default=[7, 11, 13])
    demo_bert_stack.add_argument("--top-k", type=int, default=5)
    demo_bert_stack.set_defaults(func=_cmd_demo_bert_stack)

    demo_gpt_causal = subparsers.add_parser(
        "demo-gpt-causal",
        help="Build, export, and validate a tiny GPT2 causal LM on exported weights.",
    )
    demo_gpt_causal.add_argument("--output", type=Path)
    demo_gpt_causal.add_argument("--vocab-size", type=int, default=64)
    demo_gpt_causal.add_argument("--hidden-size", type=int, default=32)
    demo_gpt_causal.add_argument("--max-position-embeddings", type=int, default=16)
    demo_gpt_causal.add_argument("--num-layers", type=int, default=2)
    demo_gpt_causal.add_argument("--num-heads", type=int, default=4)
    demo_gpt_causal.add_argument("--block-rows", type=int, default=8)
    demo_gpt_causal.add_argument("--token-ids", type=int, nargs="+", default=[3, 5, 8])
    demo_gpt_causal.add_argument("--top-k", type=int, default=5)
    demo_gpt_causal.set_defaults(func=_cmd_demo_gpt_causal)

    demo_gpt_generate = subparsers.add_parser(
        "demo-gpt-generate",
        help="Run greedy token-by-token GPT generation with KV cache and compare against HF generate().",
    )
    demo_gpt_generate.add_argument("--output", type=Path)
    demo_gpt_generate.add_argument("--vocab-size", type=int, default=64)
    demo_gpt_generate.add_argument("--hidden-size", type=int, default=32)
    demo_gpt_generate.add_argument("--max-position-embeddings", type=int, default=16)
    demo_gpt_generate.add_argument("--num-layers", type=int, default=2)
    demo_gpt_generate.add_argument("--num-heads", type=int, default=4)
    demo_gpt_generate.add_argument("--block-rows", type=int, default=8)
    demo_gpt_generate.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8])
    demo_gpt_generate.add_argument("--max-new-tokens", type=int, default=4)
    demo_gpt_generate.set_defaults(func=_cmd_demo_gpt_generate)

    demo_gpt_sample = subparsers.add_parser(
        "demo-gpt-sample",
        help="Run sampled GPT generation with temperature/top-k/top-p on the streaming engine.",
    )
    demo_gpt_sample.add_argument("--output", type=Path)
    demo_gpt_sample.add_argument("--vocab-size", type=int, default=64)
    demo_gpt_sample.add_argument("--hidden-size", type=int, default=32)
    demo_gpt_sample.add_argument("--max-position-embeddings", type=int, default=16)
    demo_gpt_sample.add_argument("--num-layers", type=int, default=2)
    demo_gpt_sample.add_argument("--num-heads", type=int, default=4)
    demo_gpt_sample.add_argument("--block-rows", type=int, default=8)
    demo_gpt_sample.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8])
    demo_gpt_sample.add_argument("--max-new-tokens", type=int, default=4)
    demo_gpt_sample.add_argument("--temperature", type=float, default=0.9)
    demo_gpt_sample.add_argument("--top-k", type=int, default=10)
    demo_gpt_sample.add_argument("--top-p", type=float, default=0.9)
    demo_gpt_sample.add_argument("--seed", type=int, default=7)
    demo_gpt_sample.add_argument("--cache-mode", default="fresh")
    demo_gpt_sample.set_defaults(func=_cmd_demo_gpt_sample)

    demo_gpt_resume = subparsers.add_parser(
        "demo-gpt-resume",
        help="Save a GPT session with KV cache and resume generation from disk.",
    )
    demo_gpt_resume.add_argument("--output", type=Path)
    demo_gpt_resume.add_argument("--vocab-size", type=int, default=64)
    demo_gpt_resume.add_argument("--hidden-size", type=int, default=32)
    demo_gpt_resume.add_argument("--max-position-embeddings", type=int, default=16)
    demo_gpt_resume.add_argument("--num-layers", type=int, default=2)
    demo_gpt_resume.add_argument("--num-heads", type=int, default=4)
    demo_gpt_resume.add_argument("--block-rows", type=int, default=8)
    demo_gpt_resume.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8])
    demo_gpt_resume.add_argument("--first-new-tokens", type=int, default=2)
    demo_gpt_resume.add_argument("--second-new-tokens", type=int, default=2)
    demo_gpt_resume.add_argument("--do-sample", action="store_true")
    demo_gpt_resume.add_argument("--temperature", type=float, default=0.9)
    demo_gpt_resume.add_argument("--top-k", type=int, default=10)
    demo_gpt_resume.add_argument("--top-p", type=float, default=0.9)
    demo_gpt_resume.add_argument("--seed", type=int, default=7)
    demo_gpt_resume.set_defaults(func=_cmd_demo_gpt_resume)

    demo_gpt_remote = subparsers.add_parser(
        "demo-gpt-remote",
        help="Download a small Hugging Face GPT model, export it, and validate logits/generation.",
    )
    demo_gpt_remote.add_argument("--output", type=Path)
    demo_gpt_remote.add_argument("--model-ref", default="sshleifer/tiny-gpt2")
    demo_gpt_remote.add_argument("--block-rows", type=int, default=8)
    demo_gpt_remote.add_argument("--prompt-ids", type=int, nargs="+", default=[1, 2, 3])
    demo_gpt_remote.add_argument("--max-new-tokens", type=int, default=3)
    demo_gpt_remote.add_argument("--trust-remote-code", action="store_true")
    demo_gpt_remote.set_defaults(func=_cmd_demo_gpt_remote)

    benchmark_gpt_cache = subparsers.add_parser(
        "benchmark-gpt-cache",
        help="Compare GPT generation with no cache, fresh cache, and warmed session cache.",
    )
    benchmark_gpt_cache.add_argument("--output", type=Path)
    benchmark_gpt_cache.add_argument("--vocab-size", type=int, default=64)
    benchmark_gpt_cache.add_argument("--hidden-size", type=int, default=32)
    benchmark_gpt_cache.add_argument("--max-position-embeddings", type=int, default=16)
    benchmark_gpt_cache.add_argument("--num-layers", type=int, default=2)
    benchmark_gpt_cache.add_argument("--num-heads", type=int, default=4)
    benchmark_gpt_cache.add_argument("--block-rows", type=int, default=8)
    benchmark_gpt_cache.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8])
    benchmark_gpt_cache.add_argument("--max-new-tokens", type=int, default=4)
    benchmark_gpt_cache.set_defaults(func=_cmd_benchmark_gpt_cache)

    benchmark_gpt_suite = subparsers.add_parser(
        "benchmark-gpt-suite",
        help="Benchmark GPT cache modes across multiple prompt lengths and save a JSON report.",
    )
    benchmark_gpt_suite.add_argument("--output", type=Path)
    benchmark_gpt_suite.add_argument("--vocab-size", type=int, default=64)
    benchmark_gpt_suite.add_argument("--hidden-size", type=int, default=32)
    benchmark_gpt_suite.add_argument("--max-position-embeddings", type=int, default=16)
    benchmark_gpt_suite.add_argument("--num-layers", type=int, default=2)
    benchmark_gpt_suite.add_argument("--num-heads", type=int, default=4)
    benchmark_gpt_suite.add_argument("--block-rows", type=int, default=8)
    benchmark_gpt_suite.add_argument("--prompt-lengths", type=int, nargs="+", default=[1, 2, 4, 8])
    benchmark_gpt_suite.add_argument("--max-new-tokens", type=int, default=4)
    benchmark_gpt_suite.add_argument("--warm-repeats", type=int, default=2)
    benchmark_gpt_suite.set_defaults(func=_cmd_benchmark_gpt_suite)

    prepare_model = subparsers.add_parser(
        "prepare-model",
        help="Prepare a reusable model workspace from a Hugging Face ref or local model path.",
    )
    prepare_model.add_argument("model_ref")
    prepare_model.add_argument("--alias")
    prepare_model.add_argument("--workspace-root", type=Path)
    prepare_model.add_argument("--block-rows", type=int, default=256)
    prepare_model.add_argument("--trust-remote-code", action="store_true")
    prepare_model.add_argument("--force", action="store_true")
    prepare_model.set_defaults(func=_cmd_prepare_model)

    list_models = subparsers.add_parser(
        "list-models",
        help="List prepared model workspaces and any saved sessions.",
    )
    list_models.add_argument("--workspace-root", type=Path)
    list_models.set_defaults(func=_cmd_list_models)

    model_info = subparsers.add_parser(
        "model-info",
        help="Show metadata for a prepared model alias.",
    )
    model_info.add_argument("alias")
    model_info.add_argument("--workspace-root", type=Path)
    model_info.set_defaults(func=_cmd_model_info)

    run_gpt = subparsers.add_parser(
        "run-gpt",
        help="Run generation on a prepared GPT workspace alias and optionally persist a session.",
    )
    run_gpt.add_argument("alias")
    run_gpt.add_argument("--workspace-root", type=Path)
    run_gpt.add_argument("--prompt-ids", type=int, nargs="+", required=True)
    run_gpt.add_argument("--max-new-tokens", type=int, default=4)
    run_gpt.add_argument("--do-sample", action="store_true")
    run_gpt.add_argument("--temperature", type=float, default=0.9)
    run_gpt.add_argument("--top-k", type=int, default=10)
    run_gpt.add_argument("--top-p", type=float, default=0.9)
    run_gpt.add_argument("--seed", type=int)
    run_gpt.add_argument("--cache-mode", choices=["none", "fresh", "session"], default="session")
    run_gpt.add_argument("--save-session", action="store_true")
    run_gpt.add_argument("--session-id")
    run_gpt.add_argument("--output-json", type=Path)
    run_gpt.set_defaults(func=_cmd_run_gpt)

    run_gpt_text = subparsers.add_parser(
        "run-gpt-text",
        help="Run text generation on a prepared GPT workspace alias using its tokenizer.",
    )
    run_gpt_text.add_argument("alias")
    run_gpt_text.add_argument("--workspace-root", type=Path)
    run_gpt_text.add_argument("--prompt")
    run_gpt_text.add_argument("--system-prompt")
    run_gpt_text.add_argument("--user-prompt")
    run_gpt_text.add_argument("--max-new-tokens", type=int, default=64)
    run_gpt_text.add_argument("--do-sample", action="store_true")
    run_gpt_text.add_argument("--temperature", type=float, default=0.9)
    run_gpt_text.add_argument("--top-k", type=int, default=10)
    run_gpt_text.add_argument("--top-p", type=float, default=0.9)
    run_gpt_text.add_argument("--seed", type=int)
    run_gpt_text.add_argument("--cache-mode", choices=["none", "fresh", "session"], default="session")
    run_gpt_text.add_argument("--save-session", action="store_true")
    run_gpt_text.add_argument("--session-id")
    run_gpt_text.add_argument("--add-special-tokens", action="store_true")
    run_gpt_text.add_argument("--keep-special-tokens", action="store_true")
    run_gpt_text.add_argument("--output-json", type=Path)
    run_gpt_text.set_defaults(func=_cmd_run_gpt_text)

    resume_gpt = subparsers.add_parser(
        "resume-gpt",
        help="Resume generation from a saved session for a prepared GPT workspace alias.",
    )
    resume_gpt.add_argument("alias")
    resume_gpt.add_argument("session_id")
    resume_gpt.add_argument("--workspace-root", type=Path)
    resume_gpt.add_argument("--max-new-tokens", type=int, default=4)
    resume_gpt.add_argument("--do-sample", action="store_true")
    resume_gpt.add_argument("--temperature", type=float, default=0.9)
    resume_gpt.add_argument("--top-k", type=int, default=10)
    resume_gpt.add_argument("--top-p", type=float, default=0.9)
    resume_gpt.add_argument("--seed", type=int)
    resume_gpt.add_argument("--cache-mode", choices=["none", "fresh", "session"], default="session")
    resume_gpt.add_argument("--no-save-session", action="store_true")
    resume_gpt.add_argument("--output-json", type=Path)
    resume_gpt.set_defaults(func=_cmd_resume_gpt)

    resume_gpt_text = subparsers.add_parser(
        "resume-gpt-text",
        help="Resume text generation from a saved session for a prepared GPT workspace alias.",
    )
    resume_gpt_text.add_argument("alias")
    resume_gpt_text.add_argument("session_id")
    resume_gpt_text.add_argument("--workspace-root", type=Path)
    resume_gpt_text.add_argument("--max-new-tokens", type=int, default=64)
    resume_gpt_text.add_argument("--do-sample", action="store_true")
    resume_gpt_text.add_argument("--temperature", type=float, default=0.9)
    resume_gpt_text.add_argument("--top-k", type=int, default=10)
    resume_gpt_text.add_argument("--top-p", type=float, default=0.9)
    resume_gpt_text.add_argument("--seed", type=int)
    resume_gpt_text.add_argument("--cache-mode", choices=["none", "fresh", "session"], default="session")
    resume_gpt_text.add_argument("--no-save-session", action="store_true")
    resume_gpt_text.add_argument("--keep-special-tokens", action="store_true")
    resume_gpt_text.add_argument("--output-json", type=Path)
    resume_gpt_text.set_defaults(func=_cmd_resume_gpt_text)

    list_tools = subparsers.add_parser(
        "list-tools",
        help="List the tool registry exposed by the current runtime.",
    )
    list_tools.add_argument("--workspace-root", type=Path)
    list_tools.set_defaults(func=_cmd_list_tools)

    call_tool = subparsers.add_parser(
        "call-tool",
        help="Invoke one registered runtime tool with JSON arguments.",
    )
    call_tool.add_argument("tool_name")
    call_tool.add_argument("--workspace-root", type=Path)
    call_tool.add_argument("--arguments", default="{}")
    call_tool.add_argument("--output-json", type=Path)
    call_tool.set_defaults(func=_cmd_call_tool)

    agent_add_text = subparsers.add_parser(
        "agent-add-text",
        help="Add inline text to an agent knowledge base for later RAG retrieval.",
    )
    agent_add_text.add_argument("agent_name")
    agent_add_text.add_argument("text")
    agent_add_text.add_argument("--workspace-root", type=Path)
    agent_add_text.add_argument("--source", default="inline-text")
    agent_add_text.add_argument("--metadata")
    agent_add_text.set_defaults(func=_cmd_agent_add_text)

    agent_add_file = subparsers.add_parser(
        "agent-add-file",
        help="Ingest a file into an agent knowledge base. PDFs work if pypdf is installed.",
    )
    agent_add_file.add_argument("agent_name")
    agent_add_file.add_argument("file_path", type=Path)
    agent_add_file.add_argument("--workspace-root", type=Path)
    agent_add_file.add_argument("--metadata")
    agent_add_file.set_defaults(func=_cmd_agent_add_file)

    agent_search = subparsers.add_parser(
        "agent-search",
        help="Search an agent knowledge base.",
    )
    agent_search.add_argument("agent_name")
    agent_search.add_argument("query")
    agent_search.add_argument("--workspace-root", type=Path)
    agent_search.add_argument("--top-k", type=int, default=5)
    agent_search.set_defaults(func=_cmd_agent_search)

    agent_run = subparsers.add_parser(
        "agent-run",
        help="Run the planner -> tool -> observation -> final loop with memory and RAG.",
    )
    agent_run.add_argument("goal")
    agent_run.add_argument("--agent-name", default="default-agent")
    agent_run.add_argument("--workspace-root", type=Path)
    agent_run.add_argument("--default-model-alias")
    agent_run.add_argument("--local-planner-alias")
    agent_run.add_argument("--remote-model-ref")
    agent_run.add_argument("--prefer-remote", action="store_true")
    agent_run.add_argument("--trust-remote-code", action="store_true")
    agent_run.add_argument("--max-steps", type=int, default=4)
    agent_run.add_argument("--generation-max-new-tokens", type=int, default=128)
    agent_run.add_argument("--output-json", type=Path)
    agent_run.set_defaults(func=_cmd_agent_run)

    serve_api_parser = subparsers.add_parser(
        "serve-api",
        help="Expose prepared models through a small local JSON API.",
    )
    serve_api_parser.add_argument("--workspace-root", type=Path)
    serve_api_parser.add_argument("--host", default="127.0.0.1")
    serve_api_parser.add_argument("--port", type=int, default=8080)
    serve_api_parser.set_defaults(func=_cmd_serve_api)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
