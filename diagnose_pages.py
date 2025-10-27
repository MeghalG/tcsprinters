#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_pages.py — single-URL diagnostics for ingest_gscholar page counts

Usage:
  python diagnose_pages.py --id sJB6rcgAAAAJ --max 40
  python diagnose_pages.py              # defaults to sJB6rcgAAAAJ

This does NOT try multiple URL variants. It fetches exactly the URL your
pipeline uses and explains why page_count may be 0/1 (HTML payload, truncation,
no EOF, parse error, etc.).
"""

from __future__ import annotations
import argparse, io, re, json
from typing import Optional, Dict, Any
import fitz  # pip install PyMuPDF

# Import your existing module (must be in the same directory or PYTHONPATH)
import ingest_gscholar as ig

def is_pdf_header(data: bytes) -> bool:
    return data.startswith(b"%PDF-")

def looks_complete_pdf(data: bytes) -> bool:
    # Check last 64 KiB for '%%EOF'
    tail = data[-65536:] if len(data) >= 65536 else data
    return b"%%EOF" in tail

def single_get_bytes(url: str, timeout: int = 45) -> tuple[Optional[bytes], Dict[str, Any]]:
    """
    One GET to exactly the given URL using ig.SESSION.
    Returns (data_or_none, diag_dict).
    """
    h = {
        "Accept": "application/pdf,*/*;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        # UA comes from ig.SESSION (whatever you configured there)
    }
    diag: Dict[str, Any] = {}
    try:
        with ig.SESSION.get(url, stream=True, timeout=timeout, headers=h, allow_redirects=True) as r:
            diag["status"] = r.status_code
            diag["content_type"] = r.headers.get("Content-Type", "")
            cl = r.headers.get("Content-Length")
            diag["content_length"] = int(cl) if cl and cl.isdigit() else None
            diag["url_final"] = r.url

            buf = io.BytesIO()
            total = 0
            for chunk in r.iter_content(262144):
                if not chunk:
                    break
                buf.write(chunk)
                total += len(chunk)
                if total >= ig.FULL_MAX:
                    diag["hit_full_max"] = True
                    break

            data = buf.getvalue()
            diag["bytes_got"] = len(data)

            # Basic checks (purely informational)
            diag["pdf_header_ok"] = is_pdf_header(data)
            diag["eof_ok"] = looks_complete_pdf(data)

            # If CL exists and we clearly got less, flag it
            if diag["content_length"] is not None and diag["bytes_got"] + 1024 < diag["content_length"]:
                diag["truncated_by_cl"] = True

            return data, diag

    except Exception as e:
        diag["error"] = repr(e)
        return None, diag

def pagecount_from_bytes(data: bytes) -> tuple[int, Optional[str]]:
    """
    Return (page_count, parse_error_str_or_None).
    """
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return int(doc.page_count), None
    except Exception as e:
        return 0, repr(e)

def arxiv_id_from_pdf_url(u: str) -> Optional[str]:
    m = re.search(r"/pdf/([^/?#]+?)(?:\.pdf)?(?:$|[?#])", u or "")
    return m.group(1) if m else None

def choose_pdf_url(work: Dict[str, Any]) -> Optional[str]:
    """
    Prefer the pipeline's pdf_url; if missing but preprint is arXiv, synthesize once.
    (Still only one URL; no variant probing.)
    """
    u = work.get("pdf_url")
    if u:
        return u
    pre = (work.get("preprint") or "")
    if pre.startswith("arxiv:"):
        arxid = pre.split(":", 1)[1]
        return f"https://arxiv.org/pdf/{arxid}.pdf"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="sJB6rcgAAAAJ", help="Google Scholar author ID")
    ap.add_argument("--max", type=int, default=80, help="Max works to fetch")
    ap.add_argument("--pages-mode", choices=["none","pdf","crossref","both"], default="both",
                    help="Match your ingest run setting (for context)")
    ap.add_argument("--crossref", action="store_true", help="Match your ingest run setting (for context)")
    ap.add_argument("--arxiv-lookup", action="store_true", help="Match your ingest run setting (for context)")
    ap.add_argument("--timeout", type=int, default=45, help="HTTP read timeout (seconds)")
    ap.add_argument("--only-problems", action="store_true", help="Show only items with pages <= 1")
    args = ap.parse_args()

    # Build once using your pipeline (to mirror what you see)
    prof = ig.build_profile(uid=args.id, limit=args.max,
                            use_crossref=args.crossref,
                            arxiv_lookup=args.arxiv_lookup,
                            pages_mode=args.pages_mode,
                            workers=8, verbose=False)

    works = prof.get("works", [])
    print(f"\n== Profile: {prof.get('name','')} ({len(works)} works) ==")

    for w in works:
        title = (w.get("title") or "").strip()
        pages = int(w.get("pages") or 0)
        pdf_url = choose_pdf_url(w)
        src = w.get("source_used_for_pages", "unknown")
        preprint = w.get("preprint", "None")

        if args.only_problems and pages > 1:
            continue

        print("\n---")
        print(f"Title       : {title}")
        print(f"Year/Venue  : {w.get('year')} | {w.get('venue')}")
        print(f"Preprint    : {preprint}")
        print(f"Pipeline    : pages={pages} (source={src})")
        print(f"pdf_url     : {pdf_url or '∅'}")

        # Show your recorded trace (helpful to see what your code tried)
        trace = w.get("pages_trace") or []
        if trace:
            print("pages_trace :")
            for lab, u, p in trace:
                print(f"  - [{lab:12}] {u} -> {p}")
        else:
            print("pages_trace : ∅")

        if not pdf_url:
            print("diag        : no pdf_url to test (skipping)")
            continue

        data, diag = single_get_bytes(pdf_url, timeout=args.timeout)
        # Compact diagnostic summary
        print("HTTP diag   :", json.dumps({
            k: diag[k] for k in [
                "status", "content_type", "content_length", "bytes_got",
                "pdf_header_ok", "eof_ok", "truncated_by_cl"
            ] if k in diag
        }, ensure_ascii=False))

        if "error" in diag:
            print("error       :", diag["error"])
            continue

        if not data or not diag.get("pdf_header_ok", False):
            print("conclusion  : NON-PDF payload (HTML/interstitial) or empty response.")
            continue

        n, perr = pagecount_from_bytes(data)
        if perr:
            print(f"parse_error : {perr}")
        print(f"PyMuPDF     : page_count={n}")

        # A little classification to guide next patch (but NOT changing behavior)
        if n == 0:
            print("conclusion  : parser returned 0 pages → malformed or non-PDF bytes.")
        elif n == 1 and (not diag.get("eof_ok") or (diag.get("bytes_got", 0) < 300_000)):
            print("conclusion  : likely TRUNCATED (tiny size and/or no %%EOF).")
        elif n >= 2:
            print("conclusion  : looks GOOD (multi-page PDF).")
        else:
            print("conclusion  : single-page but appears complete (might be legit 1-page).")

if __name__ == "__main__":
    main()
