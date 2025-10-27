from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os, subprocess, json, tempfile, shlex, pathlib

# If ingest_gscholar.py is in the repo root:
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "ingest_gscholar.py"

app = FastAPI()

# CORS: restrict to your domain if you like
ALLOWED_ORIGINS = [
    "https://tcsprinters.lol",
    "https://www.tcsprinters.lol",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # or ["*"] during testing
    allow_methods=["*"],
    allow_headers=["*"],
)

def _serp_key() -> str:
    k = os.getenv("SERPAPI_KEY", "").strip()
    if not k:
        raise HTTPException(500, "SERPAPI_KEY not configured on server")
    return k

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest")
async def ingest(request: Request):
    body = await request.json()
    scholar_id = (body.get("id") or "").strip()
    if not scholar_id:
        raise HTTPException(400, "Missing Scholar ID 'id'")
    if not SCRIPT.exists():
        raise HTTPException(500, f"Script not found: {SCRIPT}")

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "data.json")
        cmd = (
            f'python "{SCRIPT}" '
            f'--id {shlex.quote(scholar_id)} '
            f'--max 400 --pages both --crossref --arxiv-lookup '
            f'--out "{out}"'
        )
        env = os.environ.copy()
        env["SERPAPI_KEY"] = _serp_key()
        try:
            # 10-minute hard cap
            subprocess.run(cmd, shell=True, check=True, env=env, timeout=600)
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"ok": True, "data": data}
        except subprocess.CalledProcessError as e:
            raise HTTPException(500, f"ingest_gscholar failed: {e}")
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "Ingestion timed out")
