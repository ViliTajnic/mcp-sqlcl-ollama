import os, asyncio, json, shutil
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

load_dotenv(override=True)

env = os.environ.copy()
cmd = os.getenv("SQLCL_BIN", "sql")
if not os.path.isabs(cmd):
    cmd = shutil.which(cmd) or cmd

params = StdioServerParameters(command=cmd, args=["-mcp"], env=env, cwd=os.getcwd())

def render_payload(res):
    if getattr(res, "structuredContent", None) is not None:
        return res.structuredContent
    out = []
    for c in getattr(res, "content", []):
        t = getattr(c, "text", None)
        if t: out.append(t)
    return out or None

def parse_connections(payload):
    # SQLcl returns: ['conn1,conn2,conn3,...']
    if isinstance(payload, list) and payload and isinstance(payload[0], str):
        return [x.strip() for x in payload[0].split(",") if x.strip()]
    return payload

async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()

            # 1) list connections
            res = await s.call_tool("list-connections", {})
            raw = render_payload(res)
            names = parse_connections(raw)
            print("Connections:", names)

            # choose a connection (env or first)
            conn = os.getenv("SQLCL_CONN_NAME") or (names[0] if names else None)
            if not conn:
                print("No saved connections found.")
                return

            # 2) connect (IMPORTANT: use 'connection_name')
            res2 = await s.call_tool("connect", {"connection_name": conn})
            print(f"connect({conn}) =>", render_payload(res2))

            # 3) run a quick SQL
            res3 = await s.call_tool("run-sql", {"sql": "select systimestamp from dual"})
            print("systimestamp =>", render_payload(res3))

asyncio.run(main())
