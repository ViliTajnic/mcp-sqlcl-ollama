# SQLcl MCP + Ollama + Oracle 23ai

## Pre-reqs
1. **Ollama** with gpt-oss:20b:
   ```bash
   ollama pull gpt-oss:20b
   ```
2. **SQLcl 25.2+**:
   ```sql
   SQL> conn -save my23ai -savepwd appuser/app_pass@//localhost:1521/FREEPDB1
   SQL> connmgr list
   ```
3. **Python venv**:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

## Run the Streamlit app
```bash
streamlit run app.py
```

## Example prompts
- "List the saved DB connections"
- "Connect to my23ai and run: select systimestamp from dual"
- "Show 5 rows from HR.EMPLOYEES"
