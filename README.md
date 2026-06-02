# loongcli

A general-purpose AI Agent runtime built from scratch, inspired by Claude Code's architecture. It implements the core systems that power modern AI coding assistants: streaming agent loop, tool orchestration, context management, and plan-driven execution.

## Architecture

```
┌─────────────────────────────────────────────┐
│                   TUI Layer                 │
│  Rich streaming · Session resume · Commands │
├─────────────────────────────────────────────┤
│                 Agent Loop                  │
│  ReAct cycle · Loop detection · Compaction  │
├──────────┬──────────┬───────────────────────┤
│  Tools   │ Security │    MCP Integration    │
│ Registry │ 3-layer  │  Stdio + HTTP transport│
│ + Router │ perms    │  Auto-retry           │
├──────────┴──────────┴───────────────────────┤
│              Plan / Goal System             │
│  Plan CRUD · /goal 3-phase execution        │
├─────────────────────────────────────────────┤
│          Memory · Skills · Hooks            │
│  KV store · Skill framework · Lifecycle     │
└─────────────────────────────────────────────┘
```

## Key Systems

### Agent Loop (`loongcli/core/agent.py`)

Streaming ReAct loop with tool calling. Handles the core LLM ↔ tool execution cycle with:

- **Token-by-token streaming** via async generators
- **Tool call loop detection** — identical calls repeated 3+ times trigger a warning
- **Per-turn tool call cap** — prevents runaway tool usage
- **Auto-compaction** — when context approaches the token limit, older turns are summarized and compressed while preserving recent context

### Tool System (`loongcli/tools/`)

Pluggable tool registry with role-based access control:

- **10 built-in tools**: ReadFile, WriteFile, EditFile, Glob, Grep, Shell, Plan, Memorize, Recall, Skill
- **Role-based routing** — main agent and sub-agents see different tool sets
- **SubAgent delegation** — spawn child agents with isolated tool access and message-passing

### Three-Layer Permission System (`loongcli/security/`)

```
Safety Floor (hardcoded) → Rule Engine (configurable) → User Confirm (interactive)
```

Dangerous operations (rm -rf, format, registry edits) are blocked at the floor level. Configurable rules handle medium-risk operations. Everything else prompts for user confirmation.

### Plan-Driven Goal Execution (`/goal`)

Three-phase autonomous execution with human-in-the-loop approval:

1. **Planning** — Model creates a plan with tool access restricted to `plan` only (enforced at API schema level)
2. **Approval** — User reviews the plan, can approve / reject / provide feedback for re-planning
3. **Execution** — Full tool access, LLM-based intent detection (COMPLETED / NEEDS_INPUT / STUCK / CONTINUE) drives auto-continuation

### Context Window Management (`loongcli/core/compact.py`)

Turn-based compaction for million-token context windows:

- Segments conversation into user turns (user message + all subsequent assistant/tool messages)
- Keeps the 3 most recent turns intact
- Summarizes older turns via LLM, preserving active skill state
- Tool results in kept turns are replaced with placeholders to save tokens

### MCP Integration (`loongcli/mcp/`)

Model Context Protocol support for extensible tool ecosystems:

- Stdio and Streamable HTTP transports
- Auto-retry with exponential backoff on network errors
- Tool schemas merged into the main registry

## Setup

```bash
pip install -e .

# Configure API key
mkdir ~/.loongcli
echo '{"api_key": "your-deepseek-api-key"}' > ~/.loongcli/config.json

# Run
loongcli
```

## Testing

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -q
```

500+ unit tests covering core agent loop, tool execution, permission checks, compaction, plan management, goal execution, and TUI rendering.

## Project Structure

```
loongcli/
├── core/           # Agent loop, LLM client, compaction, config, events
├── tools/          # Tool registry, 10 built-in tools, role routing
├── tui/            # Terminal UI, session management, commands
├── security/       # Permission checker, safety floor
├── mcp/            # MCP client (stdio + HTTP)
├── memory/         # KV store, conversation persistence
├── plan/           # Plan store, CRUD operations
├── skills/         # Skill registry, auto-activation
└── hooks/          # Lifecycle hook system
tests/              # 500+ unit tests
```

## License

MIT
