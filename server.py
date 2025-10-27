#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import subprocess, json, os, sys

app = Flask(__name__, static_folder='.', static_url_path='')

# Resolve everything relative to this server.py
BASE_DIR  = Path(__file__).resolve().parent
DATA_JSON = (BASE_DIR / 'data.json')
INGEST    = (BASE_DIR / 'ingest_gscholar.py')
CONF      = (BASE_DIR / 'add_conferences.py')

def ensure_data_file():
    if not DATA_JSON.exists():
        # Minimal skeleton expected by the ingest script
        DATA_JSON.write_text(json.dumps({"printers": [], "years": []}, ensure_ascii=False, indent=2), encoding="utf-8")

@app.post("/api/ingest")
def api_ingest():
    try:
        payload = request.get_json(force=True) or {}
        scholar_id = (payload.get('id') or '').strip()
        if not scholar_id:
            return jsonify({"error":"missing id"}), 400

        ensure_data_file()

        # Build command (the ingest script *always* replaces matching profiles; no --replace flag)
        cmd = [
            sys.executable, str(INGEST),
            '--in', str(DATA_JSON), '--out', str(DATA_JSON),
            '--id', scholar_id
        ]

        # knobs
        if payload.get('arxiv_lookup', True): cmd.append('--arxiv-lookup')
        if payload.get('crossref', True):     cmd.append('--crossref')
        if payload.get('pages'):               cmd += ['--pages', str(payload['pages'])]
        if payload.get('max'):                 cmd += ['--max', str(int(payload['max']))]
        if payload.get('workers'):             cmd += ['--workers', str(int(payload['workers']))]

        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
        if p.returncode != 0:
            return jsonify({"error":"ingest failed", "cmd": cmd, "stdout":p.stdout, "stderr":p.stderr}), 500

        # Tag conferences (non-fatal on failure)
        p2 = subprocess.run([sys.executable, str(CONF), '--in', str(DATA_JSON), '--out', str(DATA_JSON)],
                            capture_output=True, text=True, cwd=str(BASE_DIR))
        if p2.returncode != 0:
            return jsonify({"ok": True, "warn":"add_conferences failed", "stdout":p.stdout, "stderr":p2.stderr})

        return jsonify({"ok": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/")
def root():
    return send_from_directory(str(BASE_DIR), 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
