# app.py
import os
import json
import shutil
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# ---------------------------
# Environment & configuration
# ---------------------------
load_dotenv(override=True)

st.set_page_config(page_title="SQLcl MCP + Ollama + Oracle 23ai", page_icon="🛰️")
st.title("🛰️ SQLcl MCP + Ollama (gpt-oss:20b) + Oracle 23ai — Safe Writes")

# Ollama (OpenAI-compatible) settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# SQLcl binary (absolute path recommended)
SQLCL_BIN = os.getenv("SQLCL_BIN", "sql")
DEFAULT_CONN = os.getenv("SQLCL_CONN_NAME", "")

# OpenAI client targeting Ollama
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# Advertise SQLcl MCP tools with correct schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list-connections",
            "description": "List saved SQLcl connections on this machine.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "connect",
            "description": "Connect to a saved SQLcl connection by name.",
            "parameters": {
                "type": "object",
                "properties": {"connection_name": {"type": "string"}},
                "required": ["connection_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run-sql",
            "description": "Execute SQL or PL/SQL against the current connection.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disconnect",
            "description": "Disconnect the current SQLcl session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = (
    "You are a local assistant. Use SQLcl MCP tools 'list-connections', 'connect', "
    "'run-sql', and 'disconnect' to work with Oracle 23ai. Prefer read-only operations "
    "unless the user explicitly asks for DDL/DML. When connecting, use the parameter "
    "'connection_name'. For any write-like SQL (INSERT, UPDATE, DELETE, MERGE, DDL, PL/SQL), "
    "you must wait for the user's explicit confirmation before executing."
)

# ---------------------------
# Helpers
# ---------------------------
def _resolve_sqlcl_bin() -> str:
    configured = SQLCL_BIN
    if os.path.isabs(configured):
        return configured
    resolved = shutil.which(configured)
    return resolved or configured

def _build_mcp_params() -> StdioServerParameters:
    env = os.environ.copy()
    if os.getenv("JAVA_HOME"):
        env["JAVA_HOME"] = os.getenv("JAVA_HOME")
    return StdioServerParameters(
        command=_resolve_sqlcl_bin(),
        args=["-mcp"],
        env=env,
        cwd=os.getcwd(),
    )

def render_payload(res) -> Optional[List[str] | dict]:
    """Normalize MCP tool results across client variants."""
    if getattr(res, "structuredContent", None) is not None:
        return res.structuredContent
    out = []
    for c in getattr(res, "content", []):
        t = getattr(c, "text", None)
        if t:
            out.append(t)
    return out or None

def parse_connections(payload) -> List[str]:
    """SQLcl returns ['conn1,conn2,...'] as a single CSV string within a list."""
    if isinstance(payload, list) and payload and isinstance(payload[0], str):
        return [x.strip() for x in payload[0].split(",") if x.strip()]
    return payload or []

# ---- Activity log helpers (append + render) ----
def log_event(kind: str, msg: str):
    st.session_state.setdefault("activity", []).append((kind, msg))

def render_activity():
    with st.expander("⚙️ Activity", expanded=True):
        if not st.session_state.get("activity"):
            st.caption("No activity yet.")
            return
        for kind, msg in st.session_state["activity"]:
            if kind == "llm":
                st.markdown(f"🧠 **LLM** — {msg}")
            elif kind == "tool":
                st.markdown(f"🧰 **Tool** — {msg}")
            elif kind == "mcp":
                st.markdown(f"🔌 **MCP** — {msg}")
            else:
                st.markdown(f"• {msg}")

# ---------------------------
# Write/Read classification
# ---------------------------
READ_PREFIXES = {
    "select", "with", "show", "describe", "desc", "explain"  # read-ish
}

# DML that can be committed/rolled back
DML_PREFIXES = {
    "insert", "update", "delete", "merge"
}

# DDL auto-commits in Oracle (cannot roll back DDL)
DDL_PREFIXES = {
    "create", "alter", "drop", "truncate", "rename", "comment", "grant", "revoke", "analyze"
}

# Transaction control
TXN_PREFIXES = {"commit", "rollback", "savepoint", "set transaction"}

# PL/SQL blocks are often write-like or side-effecting
PLSQL_PREFIXES = {"begin", "declare", "call"}

def _first_token(sql: str) -> str:
    return (sql or "").strip().split(None, 1)[0].lower() if sql and sql.strip() else ""

def classify_sql(sql: str) -> Tuple[str, str]:
    """
    Returns (kind, note)
    kind ∈ {'read','dml','ddl','txn','plsql','unknown'}
    """
    t = _first_token(sql)
    if t in READ_PREFIXES:
        return "read", "Read-only"
    if t in DML_PREFIXES:
        return "dml", "DML (transactional)"
    if t in DDL_PREFIXES:
        return "ddl", "DDL (auto-commit)"
    if t in TXN_PREFIXES:
        return "txn", "Transaction control"
    if t in PLSQL_PREFIXES:
        return "plsql", "PL/SQL block"
    return "unknown", "Unknown type (treated as write-like for safety)"

def is_write_like(kind: str) -> bool:
    return kind in {"dml", "ddl", "txn", "plsql", "unknown"}  # conservatively treat unknown as write-like

# ---------------------------
# Session state
# ---------------------------
if "mcp_params" not in st.session_state:
    st.session_state.mcp_params = _build_mcp_params()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if "conn_names" not in st.session_state:
    st.session_state.conn_names = []

st.session_state.setdefault("active_conn", DEFAULT_CONN)
st.session_state.setdefault("activity", [])
# Pending write confirmation: dict(name, args, kind)
st.session_state.setdefault("pending_tool_call", None)

# Global toggles
st.session_state.setdefault("bypass_write_confirmation", False)
st.session_state.setdefault("default_dml_post_action", "commit")  # or "rollback"

# ---------------------------
# MCP call bridge (with visible progress)
# ---------------------------
async def mcp_call(tool: str, args: Dict[str, Any] | None = None):
    """
    Executes an MCP tool call by spawning `sql -mcp` and opening a ClientSession.
    Shows a step-by-step status while running.
    """
    args = args or {}
    params = st.session_state.mcp_params

    # Mask potentially sensitive values
    shown_args = {k: ("***" if "pass" in k.lower() else v) for k, v in args.items()}
    step = st.status(f"Running **{tool}** via MCP …", expanded=True)
    step.write(f"Spawning MCP: `{params.command} {' '.join(params.args)}`")
    step.write(f"Args: `{json.dumps(shown_args)}`")
    log_event("tool", f"{tool} {shown_args}")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Auto-connect if needed
            default_conn = st.session_state.get("active_conn") or DEFAULT_CONN
            if tool != "connect" and default_conn:
                try:
                    step.write(f"Ensuring connection: `{default_conn}`")
                    await session.call_tool("connect", {"connection_name": default_conn})
                except Exception as e:
                    step.write(f":warning: Auto-connect failed: {e}")

            result = await session.call_tool(tool, arguments=args)
            payload = render_payload(result)

            # Tiny summary for common tools
            if tool == "run-sql" and isinstance(payload, list):
                step.write("Result received (preview):")
                preview = payload[0] if payload else "(empty)"
                st.code(preview[:1000])
            elif tool == "list-connections":
                names = parse_connections(payload)
                step.write(f"Connections: {', '.join(names) if names else '(none)'}")
            elif tool == "connect":
                step.write("Connection established." if payload else "Connect result received.")

            step.update(label=f"Finished **{tool}**", state="complete")
            return payload

def safe_mcp_call(name: str, args: Dict[str, Any] | None = None):
    try:
        return asyncio.run(mcp_call(name, args))
    except Exception as e:
        st.error(f"MCP error while calling `{name}`: {e}")
        log_event("mcp", f"Error on {name}: {e}")
        return {"error": str(e)}

# ---------------------------
# Sidebar: connection control + write safety controls
# ---------------------------
with st.sidebar:
    st.subheader("SQLcl Connection")
    pulse = st.status("Idle", expanded=False)

    if st.button("🔄 List saved connections"):
        pulse.update(label="Listing connections…", state="running")
        payload = safe_mcp_call("list-connections", {})
        st.session_state.conn_names = parse_connections(payload)
        if not st.session_state.conn_names:
            st.warning("No saved SQLcl connections found. Use `conn -save` in SQLcl first.")
        pulse.update(label="Done", state="complete")

    if st.session_state.conn_names:
        choice = st.selectbox(
            "Choose connection",
            st.session_state.conn_names,
            index=(
                st.session_state.conn_names.index(st.session_state.active_conn)
                if st.session_state.active_conn in st.session_state.conn_names
                else (
                    st.session_state.conn_names.index(DEFAULT_CONN)
                    if DEFAULT_CONN in st.session_state.conn_names
                    else 0
                )
            ),
        )
        if st.button("🔌 Connect"):
            pulse.update(label=f"Connecting to `{choice}`…", state="running")
            res = safe_mcp_call("connect", {"connection_name": choice})
            st.session_state.active_conn = choice
            st.success(f"Connected via MCP to `{choice}`")
            if res:
                st.caption(str(res))
            pulse.update(label="Connected", state="complete")
    else:
        st.caption("Tip: click “List saved connections” to populate profiles.")

    st.divider()
    st.caption(f"Ollama model: {OLLAMA_MODEL}")
    st.caption(f"SQLcl: {_resolve_sqlcl_bin()}")

    st.subheader("Write Safety")
    st.session_state.bypass_write_confirmation = st.checkbox(
        "Bypass confirmation for write-like SQL (NOT recommended)",
        value=st.session_state.bypass_write_confirmation,
    )
    st.session_state.default_dml_post_action = st.selectbox(
        "Default action after DML",
        ["commit", "rollback"],
        index=0 if st.session_state.default_dml_post_action == "commit" else 1,
        help="Applied if you confirm a DML without choosing explicitly.",
    )
    st.caption("Note: DDL auto-commits in Oracle and cannot be rolled back.")

# ---------------------------
# Pending write confirmation banner
# ---------------------------
def render_pending_confirmation():
    pending = st.session_state.get("pending_tool_call")
    if not pending:
        return False

    name = pending["name"]
    sql = pending["args"]["sql"]
    kind = pending["kind"]
    _, note = classify_sql(sql)

    st.warning(f"⚠️ Pending **{kind.upper()}** confirmation ({note}). Review before executing:")
    st.code(sql, language="sql")

    cols = st.columns(3)
    proceed_commit = cols[0].button("Proceed & COMMIT", type="primary", use_container_width=True)
    proceed_rollback = cols[1].button("Proceed & ROLLBACK", use_container_width=True)
    cancel = cols[2].button("Cancel", use_container_width=True)

    if cancel:
        st.session_state.pending_tool_call = None
        st.info("Write cancelled.")
        return True

    if proceed_commit or proceed_rollback:
        # Run the SQL, then commit/rollback if DML; warn if DDL
        payload = safe_mcp_call(name, {"sql": sql})

        if kind == "dml":
            action = "commit" if proceed_commit else "rollback"
            st.info(f"Post-DML `{action.upper()}` …")
            safe_mcp_call("run-sql", {"sql": action})
            st.success(f"DML executed and {action.upper()} applied.")
        elif kind == "ddl":
            st.warning("DDL executed. Oracle auto-commits DDL; no rollback is possible.")
        elif kind in {"plsql", "txn", "unknown"}:
            # For PL/SQL we treat as write-like; let user choose a follow-up
            action = "commit" if proceed_commit else "rollback"
            st.info(f"Post-exec `{action.upper()}` …")
            safe_mcp_call("run-sql", {"sql": action})
            st.success(f"Executed and {action.upper()} applied.")

        st.session_state.pending_tool_call = None
        # Optionally add a line to chat so the model sees result context:
        st.session_state.messages.append({
            "role": "tool",
            "name": name,
            "tool_call_id": "manual-confirmation",
            "content": json.dumps(payload, ensure_ascii=False, default=str),
        })
        return True

    return False

# ---------------------------
# Chat UI
# ---------------------------
def render_chat():
    for m in st.session_state.messages:
        role = m["role"]
        st.chat_message("user" if role == "user" else "assistant").markdown(m["content"])

def intercept_write_and_queue(tool_name: str, args: Dict[str, Any]) -> bool:
    """Return True if we queued a pending confirmation instead of executing now."""
    if tool_name != "run-sql":
        return False
    sql = (args or {}).get("sql", "") or ""
    kind, _ = classify_sql(sql)

    if not is_write_like(kind):
        return False  # read-only, allow immediate

    if st.session_state.bypass_write_confirmation:
        # still apply default post action for DML after execution
        payload = safe_mcp_call("run-sql", {"sql": sql})
        if kind == "dml":
            action = st.session_state.default_dml_post_action
            st.info(f"[Bypass mode] Post-DML `{action.upper()}` …")
            safe_mcp_call("run-sql", {"sql": action})
            st.success(f"[Bypass mode] DML executed and {action.upper()} applied.")
            st.session_state.messages.append({
                "role": "tool",
                "name": "run-sql",
                "tool_call_id": "bypass-write",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            })
        elif kind == "ddl":
            st.warning("[Bypass mode] DDL executed (auto-commit).")
        else:
            # plsql/txn/unknown -> follow default action
            action = st.session_state.default_dml_post_action
            st.info(f"[Bypass mode] Post-exec `{action.upper()}` …")
            safe_mcp_call("run-sql", {"sql": action})
            st.success(f"[Bypass mode] Executed and {action.upper()} applied.")
        return True

    # Ask for confirmation
    st.session_state.pending_tool_call = {"name": tool_name, "args": {"sql": sql}, "kind": kind}
    st.stop()  # halt this run; the confirmation UI will render on rerun

def chat_once(user_text: str):
    # Sidebar pulse shows busy state while we work
    busy = st.sidebar.status("Working…", expanded=False)

    st.session_state.messages.append({"role": "user", "content": user_text})

    # First pass: let the model decide whether to call tools
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=st.session_state.messages,
        tools=TOOLS,
    )
    msg = response.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    if tool_calls:
        names = [tc.function.name for tc in tool_calls]
        log_event("llm", f"Chose tools: {names}")

        # Execute MCP tools requested by the model
        for call in tool_calls:
            name = call.function.name
            args_json = call.function.arguments or "{}"
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError:
                args = {}

            # Normalize parameter for connect
            if name == "connect" and "name" in args and "connection_name" not in args:
                args["connection_name"] = args.pop("name")

            # Intercept write-like SQL and require confirmation
            if intercept_write_and_queue(name, args):
                # We either asked for confirmation or executed in bypass mode.
                busy.update(label="Done", state="complete")
                return

            # Safe to run immediately (read-only or non run-sql)
            res = safe_mcp_call(name, args)
            st.session_state.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(res, ensure_ascii=False, default=str),
            })
            log_event("tool", f"{name} ✅")

        # Second pass: provide the final user-facing answer (STREAMING)
        final_placeholder = st.chat_message("assistant").empty()
        stream_buf: List[str] = []

        with st.spinner("LLM composing…"):
            log_event("llm", "Streaming final answer")
            stream = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=st.session_state.messages,
                stream=True,
            )
            for chunk in stream:
                delta = getattr(chunk.choices[0], "delta", None)
                if not delta:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    stream_buf.append(piece)
                    final_placeholder.markdown("".join(stream_buf))

        final_text = "".join(stream_buf) if stream_buf else "(no content)"
        st.session_state.messages.append({"role": "assistant", "content": final_text})
    else:
        # No tool call needed; show assistant response immediately
        st.session_state.messages.append({"role": "assistant", "content": msg.content})

    busy.update(label="Done", state="complete")

# ---------------------------
# Render UI
# ---------------------------
# If there is a pending confirmation, render it now (and handle buttons)
if st.session_state.get("pending_tool_call"):
    handled = render_pending_confirmation()
    if handled:
        # After handling, continue rendering below
        pass

render_activity()
render_chat()

# Chat input
if prompt := st.chat_input(placeholder="Ask anything. Try: list your connections, or run `select systimestamp from dual`."):
    chat_once(prompt)
    st.rerun()
