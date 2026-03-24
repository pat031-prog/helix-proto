import json
from pathlib import Path

from helix_proto.agent import AgentRunner
from helix_proto.memory import add_knowledge_text, search_knowledge


def test_knowledge_search_returns_relevant_chunk(tmp_path: Path) -> None:
    add_knowledge_text(
        "helix",
        "Josh described a fallback meta-agent pattern for compressed local models.",
        source="notes",
        root=tmp_path,
    )

    result = search_knowledge("helix", "fallback meta-agent", root=tmp_path, top_k=3)

    assert result["results"]
    assert "meta-agent" in result["results"][0]["text"]


def test_agent_runner_uses_rag_then_worker_model(tmp_path: Path) -> None:
    class FakeRuntime:
        def list_models(self):  # noqa: ANN202
            return [{"alias": "tiny-agent"}]

        def model_info(self, alias):  # noqa: ANN001,ANN202
            return {"alias": alias, "model_type": "gpt2"}

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            prompt = kwargs["prompt"]
            if "Context:" in prompt:
                return {
                    "completion_text": "Use a local worker first, then fallback to remote planning when needed.",
                    "generated_text": "Use a local worker first, then fallback to remote planning when needed.",
                    "new_ids": [],
                    "generated_ids": [],
                }
            return {
                "completion_text": "generic completion",
                "generated_text": "generic completion",
                "new_ids": [],
                "generated_ids": [],
            }

        def resume_text(self, **kwargs):  # noqa: ANN003,ANN202
            return {"completion_text": "resumed"}

    runner = AgentRunner(FakeRuntime(), root=tmp_path)
    runner.add_knowledge_text(
        "helix-agent",
        "The best policy is local worker first, remote planner fallback second.",
        source="design-notes",
    )

    result = runner.run(
        goal="How should the fallback policy work?",
        agent_name="helix-agent",
        default_model_alias="tiny-agent",
        max_steps=4,
    )

    assert result["final_answer"].startswith("Use a local worker first")
    assert result["steps"][0]["tool_name"] == "rag.search"
    assert result["steps"][1]["tool_name"] == "gpt.generate_text"
    trace_path = Path(result["trace_path"])
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["final_answer"] == result["final_answer"]


def test_agent_runner_falls_back_when_local_planner_is_invalid(tmp_path: Path) -> None:
    class FakeRuntime:
        def list_models(self):  # noqa: ANN202
            return [{"alias": "tiny-agent"}]

        def model_info(self, alias):  # noqa: ANN001,ANN202
            return {"alias": alias, "model_type": "gpt2"}

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            if kwargs["alias"] == "planner-local":
                return {"completion_text": "not valid json", "generated_text": "not valid json"}
            return {
                "completion_text": "Recovered through heuristic fallback.",
                "generated_text": "Recovered through heuristic fallback.",
                "new_ids": [],
                "generated_ids": [],
            }

        def resume_text(self, **kwargs):  # noqa: ANN003,ANN202
            return {"completion_text": "resumed"}

    runner = AgentRunner(FakeRuntime(), root=tmp_path)
    runner.add_knowledge_text(
        "fallback-agent",
        "Fallback uses heuristic control when the planner emits invalid JSON.",
        source="design",
    )

    result = runner.run(
        goal="Explain the fallback behavior.",
        agent_name="fallback-agent",
        default_model_alias="tiny-agent",
        local_planner_alias="planner-local",
        max_steps=4,
    )

    assert result["final_answer"] == "Recovered through heuristic fallback."
    assert any("local:" in " ".join(item["errors"]) for item in result["planner_attempts"])
