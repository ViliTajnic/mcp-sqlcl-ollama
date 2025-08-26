# MCP-SQLcl-Ollama

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue)](https://github.com/ViliTajnic/mcp-sqlcl-ollama)  
A local developer console that lets an LLM use **SQLcl’s MCP server** to interact with **Oracle 23ai** through a chat interface powered by **Ollama `gpt‑oss:20b`**.

Live locally—**no secrets leave your machine**.

---

##  Highlights

-  Uses SQLcl MCP tools: `list-connections`, `connect`, `run-sql`, `disconnect`  
-  Leverages **function calling** via the OpenAI‑compatible API  
-  Adds **write-safety middleware**: confirm DDL/DML/PL/SQL with commit/rollback options  
-  Includes an **Activity panel** showing LLM decisions and streaming answers  
-  Runs totally **local**: Streamlit + Ollama + SQLcl MCP

---

##  Project Structure

```
mcp-sqlcl-ollama/
├── app.py              # Streamlit app with chat + MCP integration
├── mcp_smoke.py        # Quick test of MCP connectivity
├── requirements.txt
├── .env.example
└── README.md           # You’re reading it!
```

---

##  Quick Start

1. **Clone & set up virtual environment**
   ```bash
   git clone https://github.com/ViliTajnic/mcp-sqlcl-ollama.git
   cd mcp-sqlcl-ollama
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure**
   ```bash
   cp .env.example .env
   ```
   Edit `.env`, setting:
   - `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY`
   - `SQLCL_BIN` (absolute path if not on `PATH`)
   - `JAVA_HOME` (your JDK)
   - `SQLCL_CONN_NAME` (optional default profile)

4. **Set up a saved SQLcl connection**
   Run in SQLcl:
   ```sql
   SQL> conn -save my23ai -savepwd appuser/app_pass@//localhost:1521/FREEPDB1
   SQL> connmgr list
   ```

5. **(Optional) Test MCP connection**
   ```bash
   python mcp_smoke.py
   ```

6. **Launch the app**
   ```bash
   streamlit run app.py
   ```

   Use sidebar to list and connect, then chat!

---

##  Architecture Overview

### 1. Chat / Planning
- Streamlit → **Ollama `/v1/chat/completions`**
- Provides a schema of tools to the LLM
- The model returns tool calls (e.g., “run-sql” with specific SQL)

### 2. Execution Layer
- Streamlit → **SQLcl MCP** via STDIO
- Executes `list-connections`, `connect`, `run-sql`, etc.
- Captures results safely, with write confirmation before running DML/DDL

### 3. Final Answer
- Results from MCP tools are appended as tool messages
- Streamlit asks Ollama again to generate the final answer (streamed in UI)

```
User → Chat  
  ⇒ Ollama: tool_calls → Streamlit executes via SQLcl → Results → Ollama  
  ⇒ Response (streamed) → UI
```

---

##  Write Safety Features

- SQL classified into: **read**, **DML**, **DDL**, **PL/SQL**, **txn**, or **unknown**
- Write-like operations require user confirmation:
  - **Proceed & COMMIT**  
  - **Proceed & ROLLBACK**  
  - **Cancel**
- DDL operations are **auto-committed by Oracle**
- Sidebar includes a **bypass toggle** and default DML action (commit/rollback)

---

##  Configuration (`.env`)

```ini
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=gpt-oss:20b

SQLCL_BIN=/usr/local/bin/sql
JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-24.jdk/Contents/Home
SQLCL_CONN_NAME=my23ai
```

- Use `conn -save ...` in SQLcl to manage profiles
- Ensure `SQLCL_BIN` points to SQLcl 25.2+ executable

---

##  Usage Examples

- “List saved connections”
- “Connect to `my23ai`”
- “Run: `select systimestamp from dual`”
- “Create table demo(id number primary key)” → confirm DDL
- “Insert into demo values (1)” → confirm DML, commit/rollback

---

##  Tips & Troubleshooting

| Issue                      | Fix                                                              |
|---------------------------|-------------------------------------------------------------------|
| `ExceptionGroup` on startup | Ensure `SQLCL_BIN` is valid; set `JAVA_HOME`; test `sql -mcp` manually |
| No connections listed      | Add a SQLcl profile with `conn -save …`; check `connmgr list`     |
| DML/DDL not reversible     | Oracle auto-commits DDL; DML rollback only if you choose to rollback |
| `.env` not loaded          | Run from project root or set env vars manually                    |

---

##  Security

- No DB credentials traverse the LLM.  
- LLM only uses saved local SQLcl profiles for authentication.  
- Entire stack runs locally—Streamlit, Ollama, SQLcl.

---

##  License

[MIT License](LICENSE.md) © 2025 Vili Tajnić

