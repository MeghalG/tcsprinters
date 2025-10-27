#!/usr/bin/env python3
# add_author.py — append/update authors by Google Scholar ID and re-run processing
# Usage:
#   python add_author.py Jco2N80AAAAJ [ANOTHER_ID ...]
# Options:
#   --serpapi-key XXXXX               # optional; overrides env
#   --data data.json                  # default: data.json
#   --ingest ingest_gscholar.py       # default: ingest_gscholar.py
#   --addconf add_conferences.py      # default: add_conferences.py
#   -v/--verbose

import argparse, os, sys, subprocess, datetime, shutil, json
from pathlib import Path

def sh(cmd, env=None):
    print("➜", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="Google Scholar author IDs")
    ap.add_argument("--serpapi-key", default=None, help="SERPAPI key (overrides env)")
    ap.add_argument("--data", default="data.json")
    ap.add_argument("--ingest", default="ingest_gscholar.py")
    ap.add_argument("--addconf", default="add_conferences.py")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # prepare env for child processes
    child_env = dict(os.environ)
    if args.serpapi_key:
        child_env["SERPAPI_KEY"] = args.serpapi_key.strip()

    # require SERPAPI_KEY to be present one way or another
    if not child_env.get("SERPAPI_KEY"):
        print("ERROR: SERPAPI_KEY not provided. Set env or use --serpapi-key.", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data)
    ingest_path = Path(args.ingest)
    addconf_path = Path(args.addconf)

    if not ingest_path.exists():
        print(f"ERROR: {ingest_path} not found.", file=sys.stderr); sys.exit(1)
    if not addconf_path.exists():
        print(f"ERROR: {addconf_path} not found.", file=sys.stderr); sys.exit(1)

    if not data_path.exists():
        data_path.write_text(json.dumps({"printers": [], "years": []}, indent=2), encoding="utf-8")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = data_path.with_suffix(f".{ts}.bak.json")
    shutil.copyfile(data_path, backup)
    print(f"🗂  Backup written: {backup.name}")

    # ingest for each ID (replace-into behavior keeps others)
    for gid in args.ids:
        cmd = [
            sys.executable, str(ingest_path),
            "--id", gid,
            "--in", str(data_path),
            "--out", str(data_path),
            "--max", "400",
            "--workers", "16",
            "--pages", "both",
            "--crossref",
            "--arxiv-lookup",
        ]
        if args.verbose: cmd.append("--verbose")
        sh(cmd, env=child_env)

    # classify conferences
    sh([sys.executable, str(addconf_path), "--in", str(data_path), "--out", str(data_path)], env=child_env)

    print("✅ Done. data.json updated.")

if __name__ == "__main__":
    main()
