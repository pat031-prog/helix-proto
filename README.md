# helix-proto

`helix-proto` is a local streaming runtime for block-compressed Hugging Face exports.

It started as a format experiment and now includes:

- block-by-block compression with SHA-256 verification
- row-wise partial reads for large exported tensors
- GPT2 token-by-token generation with KV cache
- tokenizer-aware text generation on top of prepared GPT workspaces
- session save/resume
- runtime cache modes for small tensors
- reusable model workspaces
- a registry of runtime tools for future agent loops
- an agent loop with `plan -> tool -> observation -> final`
- persistent agent memory and lexical RAG
- a small local JSON API

## Install

Core package:

```bash
pip install -e .
```

Hugging Face path:

```bash
pip install -e ".[hf]"
```

Dev tools:

```bash
pip install -e ".[dev]"
```

## Product flow

Prepare a model once:

```bash
python -m helix_proto.cli prepare-model sshleifer/tiny-gpt2 --alias tiny-gpt2
```

List prepared models:

```bash
python -m helix_proto.cli list-models
```

Run generation on a prepared alias and save the session:

```bash
python -m helix_proto.cli run-gpt tiny-gpt2 --prompt-ids 1 2 3 --max-new-tokens 4 --save-session
```

Run text generation on top of the workspace tokenizer:

```bash
python -m helix_proto.cli run-gpt-text tiny-gpt2 --prompt "Hello world" --max-new-tokens 32 --save-session
```

Resume that session later:

```bash
python -m helix_proto.cli resume-gpt tiny-gpt2 tiny-gpt2-1234567890 --max-new-tokens 4
```

You can also resume the text path:

```bash
python -m helix_proto.cli resume-gpt-text tiny-gpt2 tiny-gpt2-1234567890 --max-new-tokens 32
```

Run the agent loop on top of that runtime:

```bash
python -m helix_proto.cli agent-run "Summarize what Josh means by fallback meta-agents" --agent-name helix --default-model-alias tiny-gpt2
```

Open the built-in browser UI:

```bash
python -m helix_proto.cli serve-api --workspace-root ./workspace-agentic
```

Then visit:

- `http://127.0.0.1:8080/`
- `http://127.0.0.1:8080/app`

## Workspace layout

Prepared models live under `workspace/models/<alias>/`:

- `model_info.json`: registry metadata
- `export/`: compressed tensor stores and `manifest.json`
- `tokenizer/`: saved tokenizer for text and chat interfaces
- `sessions/`: resumable GPT session state

This gives the project a stable `prepare once, run many` workflow instead of rebuilding demo folders for every run.

## Local API

You can expose prepared models through a tiny local API:

```bash
python -m helix_proto.cli serve-api --host 127.0.0.1 --port 8080
```

Available routes:

- `GET /health`
- `GET /models`
- `GET /models/<alias>`
- `GET /tools`
- `GET /agent/knowledge/search?agent_name=helix&query=fallback`
- `POST /prepare`
- `POST /generate`
- `POST /chat`
- `POST /chat/stream`
- `POST /resume`
- `POST /chat/resume`
- `POST /tools/<tool_name>`
- `POST /agent/knowledge/add-text`
- `POST /agent/knowledge/add-file`
- `POST /agent/memory/search`
- `POST /agent/run`
- `POST /agent/run/stream`

Example request:

```bash
curl -X POST http://127.0.0.1:8080/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"alias\":\"tiny-gpt2\",\"prompt\":\"Hello world\",\"max_new_tokens\":16,\"save_session\":true}"
```

Streaming example:

```bash
curl -N -X POST http://127.0.0.1:8080/chat/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"alias\":\"tiny-gpt2\",\"prompt\":\"Hello world\",\"max_new_tokens\":16}"
```

## Technical scope

The storage format supports numeric 1D and 2D tensors today. That covers embeddings, linear layers, layer norms, biases, and enough GPT2/BERT weights for the current inference paths.

The GPT runtime currently supports GPT2-style causal generation with:

- greedy decode
- temperature / top-k / top-p sampling
- KV cache
- runtime cache modes: `none`, `fresh`, `session`
- session serialization and resume

The agentic integration starts one layer above that runtime:

- text encoding/decoding through saved tokenizers
- chat prompt rendering from structured messages
- a tool registry with JSON schemas and callable runtime tools
- tool invocation over CLI or HTTP
- a planner loop that can choose tools, read observations, and end with a final answer
- fallback routing across local planner, remote planner, and heuristic safety mode
- persistent knowledge and episodic memory under `workspace/agents/`

That makes the local model usable as a worker inside a future planning loop, instead of keeping it as a pure numeric engine.

## Agentic flow

The current agent runner is designed around:

1. retrieve relevant memory and knowledge
2. ask a planner for the next action
3. call one tool
4. observe the result
5. repeat until a final answer

Planner routing order is:

- local planner alias first, if configured
- remote Hugging Face model ref second, if configured
- heuristic fallback last, so the loop still completes even if model planning fails

## Agent memory and RAG

Agent state lives under `workspace/agents/<agent-name>/`:

- `knowledge.jsonl`: chunked knowledge documents
- `memory.jsonl`: episodic events from goals, tool observations, and final answers
- `runs/<run-id>/trace.json`: full execution trace for each agent run

You can ingest text or files:

```bash
python -m helix_proto.cli agent-add-text helix "Josh mentioned fallback meta-agents and local workers." --source notes
python -m helix_proto.cli agent-add-file helix "C:\\Users\\Big Duck\\Desktop\\Gmail - PyPi just put out.pdf"
python -m helix_proto.cli agent-search helix "fallback meta-agents"
```

PDF ingestion works if `pypdf` is installed:

```bash
pip install -e ".[agent]"
```

## Browser UI

A lightweight browser UI ships in [`web/`](./web):

- landing page at `/`
- app at `/app`
- chat tab backed by `/chat/stream`
- agent tab backed by `/agent/run/stream`
- knowledge tab backed by `/agent/knowledge/add-text` and `/agent/knowledge/search`

It is intentionally dependency-light so the product is visible now, without waiting for a separate frontend build pipeline.

The UI is also ready for split deploys:

- backend on Render with persistent workspace storage
- frontend on Vercel from the same repo
- runtime API URL configurable in the browser and persisted in `localStorage`

## Deploy from one repo

This repository is set up so two platforms can read different parts of the same codebase:

- `Render` deploys the backend runtime from the repo root using [`render.yaml`](./render.yaml) and [`Dockerfile`](./Dockerfile)
- `Vercel` deploys the static frontend using [`vercel.json`](./vercel.json), which maps `/` and `/app` to files under [`web/`](./web)

### Render backend

Recommended env values:

- `PORT=8080`
- `HELIX_WORKSPACE_ROOT=/app/workspace`
- `HELIX_CORS_ORIGINS=https://your-vercel-project.vercel.app`

The included `render.yaml` also mounts a persistent disk at `/app/workspace`, so prepared models, sessions, knowledge, and traces survive restarts.

### Vercel frontend

The frontend can be deployed from the same repo without a build step.

Recommended Vercel project settings:

1. Set `Root Directory` to `web`
2. Set `Framework Preset` to `Other`
3. Leave `Build Command` empty
4. Leave `Output Directory` empty

The `web/vercel.json` file handles clean routes for `/` and `/app`.

After the first deploy:

1. Open `/app`
2. Paste the Render backend URL into the `Render backend URL` field
3. Click `Connect`

That URL is saved locally in the browser, so the app can call the Render API and SSE endpoints from the Vercel-hosted frontend.

## Validation commands

Format path:

```bash
python -m helix_proto.cli demo
python -m helix_proto.cli benchmark
```

Inference path:

```bash
python -m helix_proto.cli demo-gpt-remote --model-ref sshleifer/tiny-gpt2
python -m helix_proto.cli benchmark-gpt-suite
pytest -q
```

## Important commands

- `prepare-model`: export a reusable model workspace from a local or remote model
- `list-models`: inspect the registry of prepared models
- `model-info`: inspect one prepared model
- `run-gpt`: generate from a prepared GPT alias
- `run-gpt-text`: generate text from a prepared GPT alias using its tokenizer
- `resume-gpt`: continue from a saved GPT session
- `resume-gpt-text`: continue a saved text session
- `list-tools`: inspect the runtime tool registry
- `call-tool`: invoke one tool by name with JSON arguments
- `agent-add-text`: add inline knowledge to an agent
- `agent-add-file`: ingest a file into agent knowledge
- `agent-search`: search the agent knowledge base
- `agent-run`: execute the planner/tool/final loop
- `serve-api`: expose prepared models through JSON
- `convert-hf`: raw export without the workspace layer
- `demo-gpt-remote`: one-shot validation against a real remote Hugging Face model
- `benchmark-gpt-cache`: compare cache modes
- `benchmark-gpt-suite`: benchmark across prompt lengths
