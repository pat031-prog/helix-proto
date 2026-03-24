from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix_proto.memory import (
    add_knowledge_file,
    add_knowledge_text,
    append_memory_event,
    save_run_trace,
    search_knowledge,
    search_memory,
)
from helix_proto.tools import ToolRegistry, ToolSpec, build_runtime_tool_registry
from helix_proto.workspace import slugify, workspace_root


_REMOTE_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "have",
    "this",
    "your",
    "what",
    "about",
    "into",
    "they",
    "them",
    "then",
    "when",
    "where",
    "were",
    "will",
    "would",
    "there",
    "their",
    "para",
    "como",
    "esto",
    "that",
    "edge",
}


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _shorten(text: str, *, limit: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _synthesize_hits(goal: str, hits: list[dict[str, Any]], *, max_sentences: int = 4) -> str:
    if not hits:
        return "No relevant context was found."
    query_terms = {
        token
        for token in re.findall(r"[A-Za-z0-9_]+", goal.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }
    candidates: list[tuple[float, str]] = []
    for item in hits[:4]:
        text = str(item.get("text", ""))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            cleaned = _shorten(sentence, limit=220)
            if not cleaned:
                continue
            sentence_terms = set(re.findall(r"[A-Za-z0-9_]+", cleaned.lower()))
            overlap = len(query_terms & sentence_terms)
            source_bonus = float(item.get("score", 0.0))
            score = overlap * 10.0 + source_bonus
            if score > 0:
                candidates.append((score, cleaned))
    if not candidates:
        return "\n\n".join(_shorten(item["text"]) for item in hits[:3])
    candidates.sort(key=lambda item: item[0], reverse=True)
    summary: list[str] = []
    for _, sentence in candidates:
        if sentence in summary:
            continue
        summary.append(sentence)
        if len(summary) >= max_sentences:
            break
    return " ".join(summary)


def _planner_prompt(
    *,
    goal: str,
    tools: list[dict[str, Any]],
    scratchpad: list[dict[str, Any]],
    memory_hits: list[dict[str, Any]],
    knowledge_hits: list[dict[str, Any]],
    default_model_alias: str | None,
) -> str:
    examples = {
        "tool": {
            "kind": "tool",
            "thought": "Need more knowledge before answering.",
            "tool_name": "rag.search",
            "arguments": {"query": "user topic", "top_k": 3},
        },
        "final": {
            "kind": "final",
            "thought": "Enough context gathered.",
            "final": "Concise answer to the user.",
        },
    }
    return (
        "You are HelixAgent. Decide exactly one next action.\n"
        "Return only one JSON object.\n"
        'Valid shapes:\n'
        f"{json.dumps(examples['tool'])}\n"
        f"{json.dumps(examples['final'])}\n"
        "Rules:\n"
        "- Use at most one tool per response.\n"
        "- If you need context, use rag.search or memory.search first.\n"
        "- If a default model alias exists, you may use gpt.generate_text without asking the user.\n"
        "- If enough context already exists, return kind=final.\n"
        f"- Default model alias: {default_model_alias!r}\n\n"
        f"Goal:\n{goal}\n\n"
        f"Tools:\n{json.dumps(tools, indent=2)}\n\n"
        f"Relevant memory:\n{json.dumps(memory_hits, indent=2)}\n\n"
        f"Relevant knowledge:\n{json.dumps(knowledge_hits, indent=2)}\n\n"
        f"Scratchpad:\n{json.dumps(scratchpad, indent=2)}\n"
    )


@dataclass(slots=True)
class PlannerDecision:
    kind: str
    thought: str
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    final: str | None = None
    planner: str | None = None
    raw_text: str | None = None


class _BasePlanner:
    name = "base"

    def decide(self, state: dict[str, Any]) -> PlannerDecision:  # pragma: no cover - interface only
        raise NotImplementedError


class RuntimePlanner(_BasePlanner):
    name = "local"

    def __init__(self, runtime: Any, alias: str) -> None:
        self.runtime = runtime
        self.alias = alias

    def decide(self, state: dict[str, Any]) -> PlannerDecision:
        prompt = _planner_prompt(
            goal=state["goal"],
            tools=state["tool_manifest"],
            scratchpad=state["scratchpad"],
            memory_hits=state["memory_hits"],
            knowledge_hits=state["knowledge_hits"],
            default_model_alias=state.get("default_model_alias"),
        )
        result = self.runtime.generate_text(
            alias=self.alias,
            prompt=prompt,
            max_new_tokens=192,
            do_sample=False,
            cache_mode="session",
        )
        generated = result.get("completion_text") or result.get("generated_text") or ""
        parsed = _extract_first_json_object(generated)
        if parsed is None:
            raise ValueError("local planner did not return valid JSON")
        return PlannerDecision(
            kind=str(parsed.get("kind", "")).strip().lower(),
            thought=str(parsed.get("thought", "")).strip(),
            tool_name=parsed.get("tool_name"),
            arguments=dict(parsed.get("arguments", {}) or {}),
            final=parsed.get("final"),
            planner=self.name,
            raw_text=generated,
        )


class RemotePlanner(_BasePlanner):
    name = "remote"

    def __init__(self, model_ref: str, *, trust_remote_code: bool = False) -> None:
        self.model_ref = model_ref
        self.trust_remote_code = trust_remote_code

    def _model(self):
        cached = _REMOTE_MODEL_CACHE.get(self.model_ref)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("remote planner needs optional HF dependencies") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_ref,
            local_files_only=False,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_ref,
            local_files_only=False,
            trust_remote_code=self.trust_remote_code,
        )
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        _REMOTE_MODEL_CACHE[self.model_ref] = (tokenizer, model)
        return tokenizer, model

    def decide(self, state: dict[str, Any]) -> PlannerDecision:
        import torch

        tokenizer, model = self._model()
        prompt = _planner_prompt(
            goal=state["goal"],
            tools=state["tool_manifest"],
            scratchpad=state["scratchpad"],
            memory_hits=state["memory_hits"],
            knowledge_hits=state["knowledge_hits"],
            default_model_alias=state.get("default_model_alias"),
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        parsed = _extract_first_json_object(completion)
        if parsed is None:
            raise ValueError("remote planner did not return valid JSON")
        return PlannerDecision(
            kind=str(parsed.get("kind", "")).strip().lower(),
            thought=str(parsed.get("thought", "")).strip(),
            tool_name=parsed.get("tool_name"),
            arguments=dict(parsed.get("arguments", {}) or {}),
            final=parsed.get("final"),
            planner=self.name,
            raw_text=completion,
        )


class HeuristicPlanner(_BasePlanner):
    name = "heuristic"

    def decide(self, state: dict[str, Any]) -> PlannerDecision:
        goal = state["goal"]
        goal_lower = goal.lower()
        default_model_alias = state.get("default_model_alias")
        observations = state["observations"]
        knowledge_hits = state["knowledge_hits"]
        memory_hits = state["memory_hits"]

        if not observations:
            if any(term in goal_lower for term in ["previous", "before", "memoria", "remember", "earlier"]):
                return PlannerDecision(
                    kind="tool",
                    thought="Need memory recall before answering.",
                    tool_name="memory.search",
                    arguments={"query": goal, "top_k": 4},
                    planner=self.name,
                )
            if knowledge_hits:
                return PlannerDecision(
                    kind="tool",
                    thought="Knowledge base already has relevant context worth checking explicitly.",
                    tool_name="rag.search",
                    arguments={"query": goal, "top_k": 4},
                    planner=self.name,
                )
            if any(term in goal_lower for term in ["list models", "list-models", "available models", "modelos"]):
                return PlannerDecision(
                    kind="tool",
                    thought="Need the registry listing.",
                    tool_name="workspace.list_models",
                    arguments={},
                    planner=self.name,
                )
            if default_model_alias:
                return PlannerDecision(
                    kind="tool",
                    thought="Use the default worker model to draft the answer.",
                    tool_name="gpt.generate_text",
                    arguments={
                        "alias": default_model_alias,
                        "prompt": goal,
                        "max_new_tokens": state.get("generation_max_new_tokens", 96),
                    },
                    planner=self.name,
                )
            if knowledge_hits or memory_hits:
                context = knowledge_hits or memory_hits
                return PlannerDecision(
                    kind="final",
                    thought="Can answer directly from retrieved context.",
                    final=_synthesize_hits(goal, context),
                    planner=self.name,
                )
            return PlannerDecision(
                kind="final",
                thought="No tool or model is available, so answer conservatively.",
                final="No model or knowledge source is configured strongly enough to answer this yet.",
                planner=self.name,
            )

        last = observations[-1]
        tool_name = last["tool_name"]
        tool_result = last["observation"]

        if tool_name in {"rag.search", "memory.search"}:
            results = tool_result.get("result", {}).get("results", [])
            if results and default_model_alias:
                context_text = "\n\n".join(
                    f"Source: {item.get('source', item.get('kind', 'memory'))}\nText: {item['text']}"
                    for item in results[:3]
                )
                return PlannerDecision(
                    kind="tool",
                    thought="Use retrieved context to draft a final answer.",
                    tool_name="gpt.generate_text",
                    arguments={
                        "alias": default_model_alias,
                        "prompt": (
                            "Answer the user using this context when helpful.\n\n"
                            f"Context:\n{context_text}\n\nUser request:\n{goal}"
                        ),
                        "max_new_tokens": state.get("generation_max_new_tokens", 128),
                    },
                    planner=self.name,
                )
            if results:
                return PlannerDecision(
                    kind="final",
                    thought="Retrieved context is enough for a direct answer.",
                    final=_synthesize_hits(goal, results),
                    planner=self.name,
                )
            return PlannerDecision(
                kind="final",
                thought="No relevant retrieval hits were found.",
                final="I searched the current knowledge and memory but did not find relevant matches.",
                planner=self.name,
            )

        if tool_name.startswith("workspace."):
            payload = tool_result.get("result", {})
            return PlannerDecision(
                kind="final",
                thought="Workspace inspection returned the needed answer.",
                final=json.dumps(payload, indent=2),
                planner=self.name,
            )

        if tool_name.startswith("gpt."):
            payload = tool_result.get("result", {})
            final_text = payload.get("completion_text") or payload.get("generated_text") or json.dumps(payload, indent=2)
            return PlannerDecision(
                kind="final",
                thought="Worker model produced the answer.",
                final=final_text,
                planner=self.name,
            )

        return PlannerDecision(
            kind="final",
            thought="Unknown tool result, returning raw observation.",
            final=json.dumps(tool_result, indent=2),
            planner=self.name,
        )


class AgentRoutingPolicy:
    def __init__(
        self,
        *,
        local_planner_alias: str | None = None,
        remote_model_ref: str | None = None,
        prefer_remote: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.local_planner_alias = local_planner_alias
        self.remote_model_ref = remote_model_ref
        self.prefer_remote = prefer_remote
        self.trust_remote_code = trust_remote_code

    def planners(self, runtime: Any) -> list[_BasePlanner]:
        planners: list[_BasePlanner] = []
        local = RuntimePlanner(runtime, self.local_planner_alias) if self.local_planner_alias else None
        remote = (
            RemotePlanner(self.remote_model_ref, trust_remote_code=self.trust_remote_code)
            if self.remote_model_ref
            else None
        )
        if self.prefer_remote:
            if remote is not None:
                planners.append(remote)
            if local is not None:
                planners.append(local)
        else:
            if local is not None:
                planners.append(local)
            if remote is not None:
                planners.append(remote)
        planners.append(HeuristicPlanner())
        return planners


class AgentRunner:
    def __init__(self, runtime: Any, *, root: str | Path | None = None) -> None:
        self.runtime = runtime
        self.root = workspace_root(root)

    def add_knowledge_text(
        self,
        agent_name: str,
        text: str,
        *,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return add_knowledge_text(agent_name, text, source=source, root=self.root, metadata=metadata)

    def add_knowledge_file(
        self,
        agent_name: str,
        file_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return add_knowledge_file(agent_name, file_path, root=self.root, metadata=metadata)

    def search_knowledge(self, agent_name: str, query: str, *, top_k: int = 5) -> dict[str, Any]:
        return search_knowledge(agent_name, query, top_k=top_k, root=self.root)

    def search_memory(self, agent_name: str, query: str, *, top_k: int = 5) -> dict[str, Any]:
        return search_memory(agent_name, query, top_k=top_k, root=self.root)

    def _agent_tool_registry(self, *, agent_name: str) -> ToolRegistry:
        return ToolRegistry(
            [
                ToolSpec(
                    name="rag.search",
                    description="Search the persistent knowledge base for this agent.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 4},
                        },
                        "required": ["query"],
                    },
                    handler=lambda args: self.search_knowledge(
                        agent_name,
                        str(args["query"]),
                        top_k=int(args.get("top_k", 4)),
                    ),
                ),
                ToolSpec(
                    name="memory.search",
                    description="Search episodic memory from previous agent runs.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 4},
                        },
                        "required": ["query"],
                    },
                    handler=lambda args: self.search_memory(
                        agent_name,
                        str(args["query"]),
                        top_k=int(args.get("top_k", 4)),
                    ),
                ),
            ]
        )

    def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        runtime_tools: ToolRegistry,
        agent_tools: ToolRegistry,
        default_model_alias: str | None,
    ) -> dict[str, Any]:
        enriched = dict(arguments)
        if tool_name in {"gpt.generate_text", "gpt.resume_text"} and default_model_alias and "alias" not in enriched:
            enriched["alias"] = default_model_alias
        try:
            return runtime_tools.call(tool_name, enriched)
        except KeyError:
            return agent_tools.call(tool_name, enriched)

    def run(
        self,
        *,
        goal: str,
        agent_name: str = "default-agent",
        default_model_alias: str | None = None,
        local_planner_alias: str | None = None,
        remote_model_ref: str | None = None,
        prefer_remote: bool = False,
        trust_remote_code: bool = False,
        max_steps: int = 4,
        generation_max_new_tokens: int = 128,
    ) -> dict[str, Any]:
        run_id = f"{slugify(agent_name)}-{int(time.time())}"
        runtime_tools = build_runtime_tool_registry(self.runtime)
        agent_tools = self._agent_tool_registry(agent_name=agent_name)
        tool_manifest = runtime_tools.manifest() + agent_tools.manifest()

        append_memory_event(
            agent_name,
            kind="goal",
            text=goal,
            root=self.root,
            metadata={"run_id": run_id},
        )

        scratchpad: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        planner_attempts: list[dict[str, Any]] = []
        policy = AgentRoutingPolicy(
            local_planner_alias=local_planner_alias,
            remote_model_ref=remote_model_ref,
            prefer_remote=prefer_remote,
            trust_remote_code=trust_remote_code,
        )

        final_answer: str | None = None
        final_planner = None

        for step_index in range(max_steps):
            memory_hits = search_memory(
                agent_name,
                goal,
                top_k=4,
                root=self.root,
                exclude_run_id=run_id,
            )["results"]
            knowledge_hits = self.search_knowledge(agent_name, goal, top_k=4)["results"]
            state = {
                "goal": goal,
                "tool_manifest": tool_manifest,
                "scratchpad": scratchpad,
                "memory_hits": memory_hits,
                "knowledge_hits": knowledge_hits,
                "observations": observations,
                "default_model_alias": default_model_alias,
                "generation_max_new_tokens": generation_max_new_tokens,
            }

            decision: PlannerDecision | None = None
            errors: list[str] = []
            for planner in policy.planners(self.runtime):
                try:
                    candidate = planner.decide(state)
                    candidate.planner = planner.name
                    if candidate.kind not in {"tool", "final"}:
                        raise ValueError(f"invalid decision kind: {candidate.kind!r}")
                    decision = candidate
                    break
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{planner.name}: {exc}")
            planner_attempts.append({"step_index": step_index, "errors": errors})
            if decision is None:
                final_answer = "The agent could not produce a valid plan."
                final_planner = "none"
                break

            step_record = {
                "step_index": step_index,
                "planner": decision.planner,
                "thought": decision.thought,
                "kind": decision.kind,
            }

            if decision.kind == "final":
                final_answer = decision.final or ""
                final_planner = decision.planner
                scratchpad.append(step_record | {"final": final_answer})
                break

            tool_name = decision.tool_name or ""
            tool_arguments = decision.arguments or {}
            tool_result = self._call_tool(
                tool_name,
                tool_arguments,
                runtime_tools=runtime_tools,
                agent_tools=agent_tools,
                default_model_alias=default_model_alias,
            )
            observation = {
                "tool_name": tool_name,
                "arguments": tool_arguments,
                "observation": tool_result,
            }
            observations.append(observation)
            scratchpad.append(
                step_record
                | {
                    "tool_name": tool_name,
                    "arguments": tool_arguments,
                    "observation_preview": _shorten(json.dumps(tool_result, ensure_ascii=True)),
                }
            )
            append_memory_event(
                agent_name,
                kind="tool_observation",
                text=(
                    f"Goal: {goal}\nTool: {tool_name}\n"
                    f"Arguments: {json.dumps(tool_arguments)}\n"
                    f"Observation: {_shorten(json.dumps(tool_result, ensure_ascii=True), limit=600)}"
                ),
                root=self.root,
                metadata={"run_id": run_id, "step_index": step_index, "planner": decision.planner},
            )

        if final_answer is None:
            if observations:
                final_answer = json.dumps(observations[-1]["observation"], indent=2)
                final_planner = "fallback-summary"
            else:
                final_answer = "The agent stopped before producing a final answer."
                final_planner = "fallback-empty"

        append_memory_event(
            agent_name,
            kind="final_answer",
            text=final_answer,
            root=self.root,
            metadata={"run_id": run_id, "planner": final_planner},
        )

        trace = {
            "agent_name": agent_name,
            "run_id": run_id,
            "goal": goal,
            "default_model_alias": default_model_alias,
            "local_planner_alias": local_planner_alias,
            "remote_model_ref": remote_model_ref,
            "prefer_remote": prefer_remote,
            "max_steps": max_steps,
            "planner_attempts": planner_attempts,
            "steps": scratchpad,
            "observations": observations,
            "final_answer": final_answer,
            "final_planner": final_planner,
            "memory_hits": memory_hits if "memory_hits" in locals() else [],
            "knowledge_hits": knowledge_hits if "knowledge_hits" in locals() else [],
        }
        trace_path = save_run_trace(agent_name, run_id, trace, root=self.root)
        trace["trace_path"] = str(trace_path)
        return trace

    def run_stream(
        self,
        *,
        goal: str,
        agent_name: str = "default-agent",
        default_model_alias: str | None = None,
        local_planner_alias: str | None = None,
        remote_model_ref: str | None = None,
        prefer_remote: bool = False,
        trust_remote_code: bool = False,
        max_steps: int = 4,
        generation_max_new_tokens: int = 128,
    ):
        run_id = f"{slugify(agent_name)}-{int(time.time())}"
        runtime_tools = build_runtime_tool_registry(self.runtime)
        agent_tools = self._agent_tool_registry(agent_name=agent_name)
        tool_manifest = runtime_tools.manifest() + agent_tools.manifest()

        append_memory_event(
            agent_name,
            kind="goal",
            text=goal,
            root=self.root,
            metadata={"run_id": run_id},
        )

        scratchpad: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        planner_attempts: list[dict[str, Any]] = []
        policy = AgentRoutingPolicy(
            local_planner_alias=local_planner_alias,
            remote_model_ref=remote_model_ref,
            prefer_remote=prefer_remote,
            trust_remote_code=trust_remote_code,
        )

        yield {
            "event": "start",
            "agent_name": agent_name,
            "run_id": run_id,
            "goal": goal,
            "default_model_alias": default_model_alias,
        }

        final_answer: str | None = None
        final_planner = None
        memory_hits: list[dict[str, Any]] = []
        knowledge_hits: list[dict[str, Any]] = []

        for step_index in range(max_steps):
            memory_hits = search_memory(
                agent_name,
                goal,
                top_k=4,
                root=self.root,
                exclude_run_id=run_id,
            )["results"]
            knowledge_hits = self.search_knowledge(agent_name, goal, top_k=4)["results"]
            state = {
                "goal": goal,
                "tool_manifest": tool_manifest,
                "scratchpad": scratchpad,
                "memory_hits": memory_hits,
                "knowledge_hits": knowledge_hits,
                "observations": observations,
                "default_model_alias": default_model_alias,
                "generation_max_new_tokens": generation_max_new_tokens,
            }

            decision: PlannerDecision | None = None
            errors: list[str] = []
            for planner in policy.planners(self.runtime):
                try:
                    candidate = planner.decide(state)
                    candidate.planner = planner.name
                    if candidate.kind not in {"tool", "final"}:
                        raise ValueError(f"invalid decision kind: {candidate.kind!r}")
                    decision = candidate
                    break
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{planner.name}: {exc}")
            planner_attempts.append({"step_index": step_index, "errors": errors})
            if decision is None:
                final_answer = "The agent could not produce a valid plan."
                final_planner = "none"
                break

            yield {
                "event": "plan",
                "step_index": step_index,
                "planner": decision.planner,
                "thought": decision.thought,
                "kind": decision.kind,
                "errors": errors,
            }

            step_record = {
                "step_index": step_index,
                "planner": decision.planner,
                "thought": decision.thought,
                "kind": decision.kind,
            }

            if decision.kind == "final":
                final_answer = decision.final or ""
                final_planner = decision.planner
                scratchpad.append(step_record | {"final": final_answer})
                yield {
                    "event": "final",
                    "step_index": step_index,
                    "planner": final_planner,
                    "final_answer": final_answer,
                }
                break

            tool_name = decision.tool_name or ""
            tool_arguments = decision.arguments or {}
            yield {
                "event": "tool_call",
                "step_index": step_index,
                "tool_name": tool_name,
                "arguments": tool_arguments,
            }

            if (
                tool_name == "gpt.generate_text"
                and hasattr(self.runtime, "stream_text")
            ):
                streamed_args = dict(tool_arguments)
                if default_model_alias and "alias" not in streamed_args:
                    streamed_args["alias"] = default_model_alias
                tool_result = None
                for stream_event in self.runtime.stream_text(**streamed_args):
                    yield {
                        "event": "tool_stream",
                        "step_index": step_index,
                        "tool_name": tool_name,
                        "payload": stream_event,
                    }
                    if stream_event.get("event") == "done":
                        tool_result = {
                            "tool": tool_name,
                            "arguments": streamed_args,
                            "result": stream_event,
                        }
                if tool_result is None:
                    raise RuntimeError("streamed tool call ended without a done event")
            else:
                tool_result = self._call_tool(
                    tool_name,
                    tool_arguments,
                    runtime_tools=runtime_tools,
                    agent_tools=agent_tools,
                    default_model_alias=default_model_alias,
                )

            observation = {
                "tool_name": tool_name,
                "arguments": tool_arguments,
                "observation": tool_result,
            }
            observations.append(observation)
            scratchpad.append(
                step_record
                | {
                    "tool_name": tool_name,
                    "arguments": tool_arguments,
                    "observation_preview": _shorten(json.dumps(tool_result, ensure_ascii=True)),
                }
            )
            append_memory_event(
                agent_name,
                kind="tool_observation",
                text=(
                    f"Goal: {goal}\nTool: {tool_name}\n"
                    f"Arguments: {json.dumps(tool_arguments)}\n"
                    f"Observation: {_shorten(json.dumps(tool_result, ensure_ascii=True), limit=600)}"
                ),
                root=self.root,
                metadata={"run_id": run_id, "step_index": step_index, "planner": decision.planner},
            )
            yield {
                "event": "tool_result",
                "step_index": step_index,
                "tool_name": tool_name,
                "result": tool_result,
            }

        if final_answer is None:
            if observations:
                final_answer = json.dumps(observations[-1]["observation"], indent=2)
                final_planner = "fallback-summary"
            else:
                final_answer = "The agent stopped before producing a final answer."
                final_planner = "fallback-empty"
            yield {
                "event": "final",
                "planner": final_planner,
                "final_answer": final_answer,
            }

        append_memory_event(
            agent_name,
            kind="final_answer",
            text=final_answer,
            root=self.root,
            metadata={"run_id": run_id, "planner": final_planner},
        )

        trace = {
            "agent_name": agent_name,
            "run_id": run_id,
            "goal": goal,
            "default_model_alias": default_model_alias,
            "local_planner_alias": local_planner_alias,
            "remote_model_ref": remote_model_ref,
            "prefer_remote": prefer_remote,
            "max_steps": max_steps,
            "planner_attempts": planner_attempts,
            "steps": scratchpad,
            "observations": observations,
            "final_answer": final_answer,
            "final_planner": final_planner,
            "memory_hits": memory_hits,
            "knowledge_hits": knowledge_hits,
        }
        trace_path = save_run_trace(agent_name, run_id, trace, root=self.root)
        yield {
            "event": "done",
            "trace_path": str(trace_path),
            "final_answer": final_answer,
            "run_id": run_id,
        }
