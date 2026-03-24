const tabs = [...document.querySelectorAll(".tab-button")];
const panels = [...document.querySelectorAll(".panel")];

function setActiveTab(tabName) {
  for (const button of tabs) {
    button.classList.toggle("active", button.dataset.tab === tabName);
  }
  for (const panel of panels) {
    panel.classList.toggle("active", panel.dataset.panel === tabName);
  }
}

tabs.forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

async function loadModels() {
  const response = await fetch("/models");
  const payload = await response.json();
  const aliases = (payload.models || []).map((item) => item.alias);
  const selects = [
    document.getElementById("chat-alias"),
    document.getElementById("agent-model-alias"),
  ];
  for (const select of selects) {
    if (!select) continue;
    const current = select.value;
    const keepEmpty = select.id === "agent-model-alias";
    select.innerHTML = keepEmpty ? '<option value="">None</option>' : "";
    for (const alias of aliases) {
      const option = document.createElement("option");
      option.value = alias;
      option.textContent = alias;
      select.appendChild(option);
    }
    if (current && aliases.includes(current)) {
      select.value = current;
    }
  }
}

async function postSSE(url, payload, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    const errorText = await response.text();
    throw new Error(errorText || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";

    for (const chunk of chunks) {
      const lines = chunk.split("\n");
      let eventName = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        }
        if (line.startsWith("data:")) {
          data += line.slice(5).trim();
        }
      }
      if (!data) continue;
      onEvent(eventName, JSON.parse(data));
    }
  }
}

document.getElementById("chat-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const alias = document.getElementById("chat-alias").value;
  const prompt = document.getElementById("chat-prompt").value;
  const maxNewTokens = Number(document.getElementById("chat-max-tokens").value || 32);
  setText("chat-meta", "Streaming...");
  setText("chat-output", "");

  try {
    await postSSE(
      "/chat/stream",
      {
        alias,
        prompt,
        max_new_tokens: maxNewTokens,
        save_session: true,
      },
      (name, payload) => {
        if (name === "start") {
          setText("chat-meta", `Session: ${payload.session_id || "transient"}`);
        }
        if (name === "token") {
          setText("chat-output", payload.generated_text || "");
        }
        if (name === "done") {
          setText("chat-output", payload.generated_text || "");
          setText("chat-meta", `Done. Session: ${payload.session_id || "transient"}`);
        }
        if (name === "error") {
          setText("chat-meta", `Error: ${payload.error}`);
        }
      }
    );
  } catch (error) {
    setText("chat-meta", `Error: ${error.message}`);
  }
});

document.getElementById("agent-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const goal = document.getElementById("agent-goal").value;
  const agentName = document.getElementById("agent-name").value || "default-agent";
  const defaultModelAlias = document.getElementById("agent-model-alias").value || null;
  setText("agent-meta", "Running agent...");
  setText("agent-output", "");

  try {
    await postSSE(
      "/agent/run/stream",
      {
        goal,
        agent_name: agentName,
        default_model_alias: defaultModelAlias,
        max_steps: 4,
      },
      (name, payload) => {
        const current = document.getElementById("agent-output").textContent;
        if (name === "plan") {
          setText("agent-output", `${current}\n[plan:${payload.planner}] ${payload.thought}\n`);
        } else if (name === "tool_call") {
          setText("agent-output", `${current}[tool] ${payload.tool_name} ${JSON.stringify(payload.arguments)}\n`);
        } else if (name === "tool_stream" && payload.payload?.event === "token") {
          setText("agent-output", `${current}${payload.payload.token_text || ""}`);
        } else if (name === "final") {
          setText("agent-output", `${current}\n[final]\n${payload.final_answer}\n`);
        } else if (name === "done") {
          setText("agent-meta", `Done. Trace: ${payload.trace_path}`);
        } else if (name === "error") {
          setText("agent-meta", `Error: ${payload.error}`);
        }
      }
    );
  } catch (error) {
    setText("agent-meta", `Error: ${error.message}`);
  }
});

document.getElementById("knowledge-add-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const agentName = document.getElementById("knowledge-agent").value || "default-agent";
  const source = document.getElementById("knowledge-source").value || "web-note";
  const text = document.getElementById("knowledge-text").value;
  const response = await fetch("/agent/knowledge/add-text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent_name: agentName, source, text }),
  });
  const payload = await response.json();
  setText("knowledge-meta", response.ok ? "Knowledge added." : `Error: ${payload.error}`);
  if (response.ok) {
    setText("knowledge-output", JSON.stringify(payload, null, 2));
  }
});

document.getElementById("knowledge-search-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const agentName = document.getElementById("knowledge-search-agent").value || "default-agent";
  const query = document.getElementById("knowledge-query").value;
  const topK = Number(document.getElementById("knowledge-top-k").value || 4);
  const response = await fetch(
    `/agent/knowledge/search?agent_name=${encodeURIComponent(agentName)}&query=${encodeURIComponent(query)}&top_k=${topK}`
  );
  const payload = await response.json();
  setText("knowledge-meta", response.ok ? `${(payload.results || []).length} hits.` : `Error: ${payload.error}`);
  setText("knowledge-output", JSON.stringify(payload, null, 2));
});

loadModels().catch((error) => {
  setText("chat-meta", `Error loading models: ${error.message}`);
});
