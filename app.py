"""
Databricks App — Simple Claude Agent
=====================================
Single-file app. Upload this as your Databricks App entry point.

Databricks auto-injects:
  DATABRICKS_HOST  — your workspace URL
  DATABRICKS_TOKEN — a short-lived workspace token

No external API keys needed.
"""

import ast
import json
import operator
import os
import uuid
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pyspark.sql import SparkSession

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL    = os.environ.get("CLAUDE_MODEL", "databricks-claude-sonnet-4")
DB_HOST  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
DB_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

SYSTEM_PROMPT = """You are a helpful data assistant running inside Databricks.
You have access to two tools:
- calculator: for any arithmetic or mathematical expressions
- run_sql: to query Unity Catalog tables with read-only Spark SQL

Always use a tool when the question involves numbers or data.
Be concise and direct in your answers."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Use for any arithmetic or numerical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(120 * 0.15) + 30'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a read-only SQL query against Unity Catalog. Use SHOW CATALOGS / SHOW TABLES to explore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A SELECT or SHOW SQL statement."}
                },
                "required": ["query"],
            },
        },
    },
]

# In-process session store
SESSIONS: dict[str, list[dict]] = {}

app   = FastAPI(title="Claude Agent")
spark = SparkSession.builder.getOrCreate()

# ── Tool implementations ───────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    SAFE_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.USub: operator.neg,
    }
    def _eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp):
            return SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported: {ast.dump(node)}")
    try:
        return str(_eval(ast.parse(expression.strip(), mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


def run_sql(query: str) -> str:
    blocked = ["insert","update","delete","drop","create","alter","truncate","merge"]
    if any(k in query.lower() for k in blocked):
        return json.dumps({"error": "Only SELECT/SHOW queries allowed."})
    try:
        rows = [r.asDict() for r in spark.sql(query).limit(50).collect()]
        return json.dumps(rows, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_HANDLERS = {
    "calculator": lambda a: calculator(a["expression"]),
    "run_sql":    lambda a: run_sql(a["query"]),
}

# ── Agent loop ─────────────────────────────────────────────────────────────────

async def run_agent(messages: list[dict]) -> str:
    url     = f"{DB_HOST}/serving-endpoints/{MODEL}/invocations"
    headers = {"Authorization": f"Bearer {DB_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for _ in range(10):
            payload = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "tools": TOOLS,
                "max_tokens": 2048,
            }
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()

            msg        = resp.json()["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            messages.append(msg)

            if not tool_calls:
                return msg.get("content", "")

            # Execute tools
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                result  = TOOL_HANDLERS.get(fn_name, lambda _: "Unknown tool")(fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

    return "Reached max reasoning steps."


async def word_stream(text: str) -> AsyncIterator[str]:
    """Chunk the answer into small bursts for a streaming feel."""
    import asyncio
    words, buf = text.split(), []
    for i, w in enumerate(words):
        buf.append(w)
        if len(buf) >= 4 or i == len(words) - 1:
            yield f"data: {json.dumps({'delta': ' '.join(buf) + ' '})}\n\n"
            buf = []
            await asyncio.sleep(0.02)
    yield f"data: {json.dumps({'done': True})}\n\n"

# ── API routes ─────────────────────────────────────────────────────────────────

class ChatReq(BaseModel):
    message: str
    session_id: str | None = None

class ClearReq(BaseModel):
    session_id: str


@app.post("/chat")
async def chat(req: ChatReq):
    sid      = req.session_id or str(uuid.uuid4())
    messages = SESSIONS.setdefault(sid, [])
    messages.append({"role": "user", "content": req.message})

    try:
        answer = await run_agent(messages)
    except Exception as e:
        answer = f"⚠ Error: {e}"

    messages.append({"role": "assistant", "content": answer})
    SESSIONS[sid] = messages[-40:]

    async def stream():
        yield f"data: {json.dumps({'session_id': sid})}\n\n"
        async for chunk in word_stream(answer):
            yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/clear")
async def clear(req: ClearReq):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"model": MODEL, "host": DB_HOST}

# ── Frontend ───────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Databricks Agent</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0f16;--surf:#161923;--surf2:#1e2233;
  --border:#272d42;--text:#dde1ef;--muted:#737a96;
  --accent:#7c6dfa;--accent2:#6558e8;
  --user:#1a2445;--radius:14px;
  --font:"Inter",system-ui,sans-serif
}
body{font-family:var(--font);background:var(--bg);color:var(--text);
     height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* header */
header{padding:14px 22px;background:var(--surf);border-bottom:1px solid var(--border);
       display:flex;align-items:center;gap:12px;flex-shrink:0}
.logo{width:34px;height:34px;background:var(--accent);border-radius:9px;
      display:flex;align-items:center;justify-content:center;font-size:16px}
header h1{font-size:15px;font-weight:600}
header small{font-size:11px;color:var(--muted);display:block;margin-top:1px}
.badge{margin-left:auto;padding:3px 10px;background:#162a1a;
       border:1px solid #1f4d25;border-radius:20px;font-size:11px;color:#4caf75}

/* messages */
#msgs{flex:1;overflow-y:auto;padding:22px;display:flex;
      flex-direction:column;gap:18px;scroll-behavior:smooth}
#msgs::-webkit-scrollbar{width:5px}
#msgs::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

.row{display:flex;gap:10px;max-width:780px;width:100%}
.row.user{align-self:flex-end;flex-direction:row-reverse}
.row.bot{align-self:flex-start}

.av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px}
.row.user .av{background:var(--accent)}
.row.bot  .av{background:var(--surf2);border:1px solid var(--border)}

.bbl{padding:11px 15px;border-radius:var(--radius);font-size:14px;
     line-height:1.65;max-width:660px}
.row.user .bbl{background:var(--user);border:1px solid #283870;
               border-bottom-right-radius:4px}
.row.bot  .bbl{background:var(--surf2);border:1px solid var(--border);
               border-bottom-left-radius:4px}

/* tool call pill */
.tool-pill{display:inline-flex;align-items:center;gap:5px;
           margin:6px 0 2px;padding:4px 10px;
           background:#120d2e;border:1px solid #3b3280;
           border-radius:20px;font-size:11px;color:#a99dff}
.tool-pill svg{opacity:.7}

/* cursor */
.cur{display:inline-block;width:2px;height:13px;background:var(--accent);
     margin-left:2px;vertical-align:middle;animation:blink .7s steps(1) infinite}
@keyframes blink{50%{opacity:0}}

/* empty state */
#empty{flex:1;display:flex;flex-direction:column;align-items:center;
       justify-content:center;gap:8px;color:var(--muted);text-align:center;
       padding-bottom:60px}
#empty .icon{font-size:42px;margin-bottom:4px}
#empty h2{color:var(--text);font-size:17px;font-weight:600}

/* chips */
.chips{display:flex;flex-wrap:wrap;gap:7px;max-width:780px;margin:0 auto 10px}
.chip{padding:6px 14px;background:var(--surf2);border:1px solid var(--border);
      border-radius:20px;font-size:12px;color:var(--muted);cursor:pointer;
      transition:all .15s}
.chip:hover{border-color:var(--accent);color:var(--text)}

/* footer */
footer{padding:14px 22px;background:var(--surf);border-top:1px solid var(--border);
       flex-shrink:0}
.irow{display:flex;gap:8px;max-width:780px;margin:0 auto}
textarea{flex:1;background:var(--surf2);border:1px solid var(--border);
         border-radius:var(--radius);color:var(--text);font-family:var(--font);
         font-size:14px;padding:11px 15px;resize:none;outline:none;
         line-height:1.5;max-height:130px;transition:border-color .15s}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--muted)}
.send{width:42px;height:42px;background:var(--accent);border:none;
      border-radius:10px;color:#fff;cursor:pointer;display:flex;
      align-items:center;justify-content:center;flex-shrink:0;
      align-self:flex-end;transition:background .15s,transform .1s}
.send:hover{background:var(--accent2)}
.send:active{transform:scale(.94)}
.send:disabled{opacity:.35;cursor:not-allowed}
.clr{width:auto;padding:0 13px;background:transparent;
     border:1px solid var(--border);border-radius:10px;
     color:var(--muted);font-size:12px;cursor:pointer;align-self:flex-end;
     height:42px;transition:all .15s}
.clr:hover{border-color:var(--text);color:var(--text)}
</style>
</head>
<body>

<header>
  <div class="logo">⚡</div>
  <div>
    <h1>Databricks Agent</h1>
    <small>Claude · calculator · Spark SQL</small>
  </div>
  <div class="badge">● Online</div>
</header>

<div id="msgs">
  <div id="empty">
    <div class="icon">🤖</div>
    <h2>What can I help you with?</h2>
    <p>Ask me anything — I can do math and query your Unity Catalog tables.</p>
  </div>
</div>

<footer>
  <div class="chips" id="chips">
    <div class="chip" onclick="ask(this)">What catalogs exist in this workspace?</div>
    <div class="chip" onclick="ask(this)">Compound interest: $10k at 8% for 20 years</div>
    <div class="chip" onclick="ask(this)">Show me 5 rows from any available table</div>
    <div class="chip" onclick="ask(this)">What's 15% tip on a $87.50 bill?</div>
  </div>
  <div class="irow">
    <textarea id="inp" rows="1" placeholder="Ask anything…"
      onkeydown="onKey(event)" oninput="resize(this)"></textarea>
    <button class="clr" onclick="clear_()">Clear</button>
    <button class="send" id="sbtn" onclick="send()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.2"
           stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
  </div>
</footer>

<script>
let sid = null, busy = false;
const msgsEl = document.getElementById("msgs");
const inp    = document.getElementById("inp");
const sbtn   = document.getElementById("sbtn");
const empty  = document.getElementById("empty");
const chips  = document.getElementById("chips");

function resize(el){el.style.height="auto";el.style.height=Math.min(el.scrollHeight,130)+"px"}
function onKey(e){if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}}
function ask(el){inp.value=el.textContent;resize(inp);send()}
function scroll(){msgsEl.scrollTop=msgsEl.scrollHeight}

function addBubble(role, html){
  empty.style.display="none";
  chips.style.display="none";
  const row=document.createElement("div");
  row.className=`row ${role}`;
  const av=document.createElement("div");
  av.className="av";
  av.textContent=role==="user"?"👤":"🤖";
  const bbl=document.createElement("div");
  bbl.className="bbl";
  bbl.innerHTML=html;
  row.appendChild(av);
  row.appendChild(bbl);
  msgsEl.appendChild(row);
  scroll();
  return bbl;
}

function fmt(text){
  // tool pill detection
  const toolMatch = text.match(/\[tool:([^\]]+)\]/g);
  let body = text.replace(/\[tool:[^\]]+\]/g,"").trim();
  // light markdown
  body = body
    .replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>")
    .replace(/`([^`]+)`/g,"<code style='background:#0f0d20;padding:1px 5px;border-radius:4px;font-size:12px'>$1</code>")
    .split("\n\n").map(p=>`<p style='margin-bottom:8px'>${p.trim()}</p>`).join("");
  let pills="";
  if(toolMatch) pills=toolMatch.map(m=>{
    const name=m.replace(/\[tool:|\]/g,"");
    return `<div class="tool-pill">
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
      ${name}</div>`;
  }).join("");
  return pills+body;
}

async function send(){
  const msg=inp.value.trim();
  if(!msg||busy)return;
  addBubble("user",msg);
  inp.value="";resize(inp);
  sbtn.disabled=true;busy=true;

  const bbl=addBubble("bot","");
  const cur=document.createElement("span");
  cur.className="cur";
  bbl.appendChild(cur);

  let full="";
  try{
    const r=await fetch("/chat",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({message:msg,session_id:sid})
    });
    const reader=r.body.getReader();
    const dec=new TextDecoder();
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      for(const line of dec.decode(value).split("\n")){
        if(!line.startsWith("data: "))continue;
        const d=JSON.parse(line.slice(6));
        if(d.session_id){sid=d.session_id}
        if(d.delta){full+=d.delta;bbl.textContent=full;bbl.appendChild(cur)}
        if(d.done){cur.remove();bbl.innerHTML=fmt(full)}
      }
      scroll();
    }
  }catch(e){
    cur.remove();
    bbl.innerHTML=`<span style="color:#f87171">Error: ${e.message}</span>`;
  }
  sbtn.disabled=false;busy=false;inp.focus();
}

async function clear_(){
  if(sid)await fetch("/clear",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:sid})});
  sid=null;
  msgsEl.innerHTML="";
  msgsEl.appendChild(empty);
  empty.style.display="flex";
  chips.style.display="flex";
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)
