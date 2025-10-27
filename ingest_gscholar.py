#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_gscholar.py — arXiv-first, SIMPLE page counting with PyMuPDF

- Pull works from Google Scholar via SerpAPI for one or more author IDs.
- Page counts: download full PDF (bounded) and count with PyMuPDF; otherwise Crossref fallback.
- Detects arXiv from links OR from venue/title strings.
- Sets online fields immediately (never says "no online version" if a link exists).
- Replace-in-place: if an author exists in --in, overwrite that profile; preserve others.
"""

from __future__ import annotations

import argparse, io, json, os, re, time, hashlib
from typing import Dict, List, Optional, Tuple, Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path as _Path

import html, unicodedata, re
from functools import lru_cache


import requests
from requests.adapters import HTTPAdapter, Retry

# ---- hard deps for simple counting ----
import fitz  # PyMuPDF

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search.json"
ARXIV_API   = "https://export.arxiv.org/api/query"
CROSSREF    = "https://api.crossref.org/works"

PDF_TIMEOUT = 30
FULL_MAX    = 128_000_000  # cap full fetch at 128MB (safety)
HEADERS_PDF = {"User-Agent":"ingest-gscholar/simple-1.0", "Accept":"application/pdf"}

# ---------- HTTP session ----------
def make_session(pool:int=16)->requests.Session:
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.3,
                    status_forcelist=(429,500,502,503,504),
                    allowed_methods=False)
    ad = HTTPAdapter(max_retries=retries, pool_connections=pool, pool_maxsize=pool)
    s.mount("http://", ad); s.mount("https://", ad)
    s.headers.update({"User-Agent":"ingest-gscholar/simple-1.0"})
    return s

SESSION = make_session()

# ---------- Disk cache ----------
_CACHE = _Path(".cache"); _CACHE.mkdir(exist_ok=True)
def _ckey(s:str)->str: return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _cget(key:str, ttl:int)->Optional[str]:
    p=_CACHE/_ckey(key)
    if not p.exists(): return None
    try:
        obj=json.loads(p.read_text(encoding="utf-8"))
        if (time.time()-obj.get("_ts",0))>ttl: return None
        return obj.get("data")
    except Exception:
        return None

def _cput(key:str, data:str)->None:
    p=_CACHE/_ckey(key)
    try:
        p.write_text(json.dumps({"_ts":time.time(),"data":data}), encoding="utf-8")
    except Exception:
        pass

def get_json_cached(url:str, params:dict, ttl:int=24*3600, timeout:int=30)->dict:
    key=url+"?"+ "&".join(f"{k}={params[k]}" for k in sorted(params))
    c=_cget(key, ttl)
    if c is not None:
        try: return json.loads(c)
        except Exception: pass
    r=SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data=r.json()
    _cput(key, json.dumps(data))
    return data

def get_text_cached(url:str, params:dict, ttl:int=24*3600, timeout:int=30, headers:Optional[dict]=None)->str:
    key=url+"?"+ "&".join(f"{k}={params[k]}" for k in sorted(params))
    c=_cget(key, ttl)
    if c is not None: return c
    r=SESSION.get(url, params=params, timeout=timeout, headers=headers or {})
    r.raise_for_status()
    txt=r.text
    _cput(key, txt)
    return txt

# ---------- Helpers ----------
ARXIV_URL_RE  = re.compile(r"https?://arxiv\.org/(?:abs|pdf)/([^/?#]+?)(?:\.pdf)?(?:$|[?#])", re.I)
EPRINT_RE     = re.compile(r"eprint\.iacr\.org/(\d{4})/(\d+)", re.I)
ECCC_RE       = re.compile(r"eccc(?:\.weizmann\.ac\.il|\.hpi\.de)/(?:report|papers)/(\d{4})/(\d+)", re.I)
DOI_RE        = re.compile(r"\b10\.\d{4,9}/[^\s#>]+", re.I)
_ARXIV_PDF_SIMPLE_RE = re.compile(r"^https://arxiv\.org/pdf/([^/?#]+?)(?:\.pdf)?(?:$|[?#])", re.I)

def require_key():
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY missing. export SERPAPI_KEY=...")

def clean_int(x)->Optional[int]:
    try: return int(x) if x is not None else None
    except: return None

def _normalize_pdf_url(u: str) -> str:
    if not u: return u
    u = re.sub(r"^http://","https://", u.strip())
    m = _ARXIV_PDF_SIMPLE_RE.match(u)
    if m:
        base = m.group(1)
        qs = ""
        if "?" in u:
            qs = "?" + u.split("?", 1)[1]
        return f"https://arxiv.org/pdf/{base}.pdf{qs}"
    return u

# ---------- SerpAPI ----------
def serp_author_page(uid:str, start:int=0, num:int=100)->Dict:
    p={"engine":"google_scholar_author", "author_id":uid, "api_key":SERPAPI_KEY,
       "start":start, "num":num, "sort":"pubdate"}
    return get_json_cached(SERPAPI_URL, p, ttl=24*3600)

def serp_list_articles(uid:str, limit:int, verbose:bool=False)->List[Dict]:
    out,start=[],0
    while len(out) < limit:
        page = serp_author_page(uid, start, 100)
        status = ((page.get("search_metadata") or {}).get("status") or "").lower()
        if status != "success":
            break
        arts = page.get("articles") or []
        if not isinstance(arts, list): break
        out.extend(arts)
        nextp = (page.get("serpapi_pagination") or {}).get("next")
        if len(arts) < 100 or not nextp: break
        start += 100
        if verbose: print(f"[serpapi] {len(out)} so far...")
    return out[:limit]

# ---------- URL extraction ----------
def _flatten_urls(x:Any)->Iterable[str]:
    if isinstance(x,str):
        if x.startswith("http"): yield x
        return
    if isinstance(x,dict):
        for _,v in x.items(): yield from _flatten_urls(v)
    elif isinstance(x,list):
        for it in x: yield from _flatten_urls(it)

def collect_links(article:Dict)->List[str]:
    urls=set()
    for k in ("link","result_id","publication_url","source","pdf"):
        u = article.get(k)
        if isinstance(u,str) and u.startswith("http"): urls.add(u)
    for u in _flatten_urls(article):
        if isinstance(u,str) and u.startswith("http"): urls.add(u)
    return [re.sub(r"^http://","https://", u.strip()) for u in urls]

# ---------- arXiv helpers ----------
def arxiv_pdf_from_id(arxiv_id:str)->str:
    return "https://arxiv.org/pdf/"+arxiv_id+".pdf"

@lru_cache(maxsize=4096)
def arxiv_pdf_candidates(arxiv_id: str):
    base = re.sub(r"v\d+$", "", arxiv_id or "")
    cands = [arxiv_pdf_from_id(base)]
    for v in range(1,9):
        cands.append(arxiv_pdf_from_id(f"{base}v{v}"))
    return cands

def find_arxiv_id_in_links(links:List[str])->Optional[str]:
    for u in links or []:
        m=ARXIV_URL_RE.search(u or "")
        if m: return m.group(1)
    return None

def _norm_fast(s: str) -> str:
    """Normalize a title string for fast token comparison."""
    if not s:
        return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKD", s)
    # unify dash/quote variants
    s = (s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
           .replace("\u2018", "'").replace("\u2019", "'")
           .replace("\u201c", '"').replace("\u201d", '"'))
    # keep only letters/digits/spaces, collapse spaces, lowercase
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _tokens(s: str):
    """Split normalized title into tokens (keep all)."""
    return [w for w in s.split() if len(w) >= 2]

def _score_tokens(target_set, cand_set):
    """Compute containment and Jaccard overlaps."""
    inter = target_set & cand_set
    if not inter:
        return 0.0, 0.0
    containment = len(inter) / max(1, min(len(target_set), len(cand_set)))
    jaccard = len(inter) / max(1, len(target_set | cand_set))
    return containment, jaccard

@lru_cache(maxsize=8192)
def arxiv_title_lookup_pdf(title: str, verbose: bool=False) -> Optional[str]:
    """
    Fast fuzzy arXiv lookup by token overlap.
    Tries exact title first, then token-based search.
    """
    raw = (title or "").strip()
    if not raw:
        return None

    T = _norm_fast(raw)
    tset = set(_tokens(T))
    if not tset:
        return None

    def fetch(query: str) -> str:
        return get_text_cached(
            ARXIV_API,
            params={"search_query": query, "max_results": 15},
            timeout=20, ttl=7*24*3600,
            headers={"User-Agent":"ingest-gscholar/token-lookup"}
        )

    # 1) Try exact quoted title first
    try:
        txt = fetch(f'ti:"{raw}"')
    except Exception:
        txt = ""

    # 2) If no entries, fall back to token-based AND search
    if "<entry>" not in txt:
        toks = list(tset)[:10]
        q = " AND ".join(f"ti:{w}" for w in toks) if toks else f'ti:"{raw}"'
        try:
            txt = fetch(q)
        except Exception:
            txt = ""

    # 3) Parse and score all candidates
    best_pdf, best_pair = None, (0.0, 0.0)
    for ent in re.finditer(r"<entry>(.*?)</entry>", txt, flags=re.S|re.I):
        block = ent.group(1)
        mt = re.search(r"<title>(.*?)</title>", block, flags=re.S|re.I)
        mid = re.search(r"<id>.*?/abs/([^<]+)</id>", block, flags=re.S|re.I)
        if not mt or not mid:
            continue
        cand_title = re.sub(r"\s+", " ", (mt.group(1) or "").strip())
        cand_title = re.sub(r"<.*?>", "", cand_title)
        Nt = _norm_fast(cand_title)
        cset = set(_tokens(Nt))
        if not cset:
            continue

        containment, jacc = _score_tokens(tset, cset)

        if (containment, jacc) > best_pair:
            best_pair = (containment, jacc)
            best_pdf = f"https://arxiv.org/pdf/{mid.group(1)}.pdf"

    # 4) Accept if enough overlap
    cont_thr = 0.88
    jacc_thr = 0.70
    if best_pdf and (best_pair[0] >= cont_thr or best_pair[1] >= jacc_thr):
        if verbose:
            print(f"[arxiv] token-match ok: containment={best_pair[0]:.2f}, "
                  f"jaccard={best_pair[1]:.2f} -> {best_pdf}")
        return best_pdf

    if verbose:
        print(f"[arxiv] no token match (containment={best_pair[0]:.2f}, "
              f"jaccard={best_pair[1]:.2f}) for: {title}")
    return None

def _looks_complete_pdf(data: bytes) -> bool:
    # Check last 64 KiB for '%%EOF'
    tail = data[-65536:] if len(data) >= 65536 else data
    return b"%%EOF" in tail

def _is_pdf_header(data: bytes) -> bool:
    return data.startswith(b"%PDF-")



# --- SIMPLE + ROBUST full-fetch page counter using PyMuPDF ---
@lru_cache(maxsize=16384)
def fetch_pdf_pages(url: str, verbose: bool = False) -> int:
    """
    Single-URL page counter with cookie priming:
      1) Try the PDF URL.
      2) If we get HTML (interstitial), GET the abs page once to set cookies.
      3) Retry the SAME PDF URL with Referer.
    No multi-version guessing.
    """
    def log(msg):
        if verbose:
            print(msg)

    def is_pdf(b: bytes) -> bool:
        return b.startswith(b"%PDF-")

    base = _normalize_pdf_url(url)

    # Derive abs page only if it's really arXiv
    abs_url = None
    m = _ARXIV_PDF_SIMPLE_RE.match(base)
    if m:
        arxid = m.group(1)  # e.g., 2508.09422 or 2508.09422v2
        abs_url = f"https://arxiv.org/abs/{arxid}"

    # Browser-ish headers
    h_common = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/pdf,*/*;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def get_bytes(u: str, headers: dict) -> tuple[bytes | None, dict]:
        diag = {}
        try:
            with SESSION.get(u, stream=True, allow_redirects=True,
                             timeout=(10, max(45, PDF_TIMEOUT)), headers=headers) as r:
                diag["status"] = r.status_code
                diag["content_type"] = (r.headers.get("Content-Type") or "").lower()
                cl = r.headers.get("Content-Length")
                diag["content_length"] = int(cl) if cl and cl.isdigit() else None

                buf = io.BytesIO()
                total = 0
                for ch in r.iter_content(262144):
                    if not ch:
                        break
                    buf.write(ch)
                    total += len(ch)
                    if total >= FULL_MAX:
                        diag["hit_full_max"] = True
                        break

                data = buf.getvalue()
                diag["bytes_got"] = len(data)
                diag["pdf_header_ok"] = is_pdf(data)

                # If server promised a length and we clearly got less, treat as incomplete
                if diag["content_length"] is not None and total + 1024 < diag["content_length"]:
                    diag["truncated_by_cl"] = True
                    return None, diag

                if not diag["pdf_header_ok"]:
                    return None, diag
                return data, diag
        except Exception as e:
            diag["error"] = repr(e)
            return None, diag

    # 1) First attempt directly
    data, d1 = get_bytes(base, h_common)
    if verbose:
        log(f"[pdf:first] {base} -> status={d1.get('status')} ct={d1.get('content_type')} "
            f"CL={d1.get('content_length')} got={d1.get('bytes_got')} pdf={d1.get('pdf_header_ok')}")

    # 2) If HTML/non-PDF, prime cookies by hitting the abs page once
    if data is None and abs_url:
        try:
            # Abs fetch to set cookies/session, ignoring response body
            _ = SESSION.get(abs_url, timeout=15, headers={
                **h_common,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin",
                "Upgrade-Insecure-Requests": "1",
            }, allow_redirects=True)
            if verbose:
                log(f"[pdf:prime] hit {abs_url} to set cookies (status={_.status_code})")
        except Exception as e:
            if verbose:
                log(f"[pdf:prime] abs fetch failed: {e}")

        # 3) Retry the SAME PDF URL with a Referer
        data, d2 = get_bytes(base, {**h_common, "Referer": abs_url})
        if verbose:
            log(f"[pdf:retry] {base} -> status={d2.get('status')} ct={d2.get('content_type')} "
                f"CL={d2.get('content_length')} got={d2.get('bytes_got')} pdf={d2.get('pdf_header_ok')}")

    if not data:
        return 0

    # Simple PyMuPDF read (as you requested)
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return int(doc.page_count)
    except Exception as e:
        if verbose:
            log(f"[pdf] parse fail: {e}")
        return 0





# ---------- Crossref ----------
@lru_cache(maxsize=8192)
def crossref_pages_by_doi(doi:str, year:Optional[int], verbose:bool=False)->int:
    try:
        r=SESSION.get(f"{CROSSREF}/{doi}", timeout=20)
        r.raise_for_status()
        it=r.json().get("message",{})
        pg=it.get("page","") or it.get("pageStart") or ""
        if isinstance(pg,str) and "-" in pg:
            a,b=pg.split("-",1)
            A=int("".join(ch for ch in a if ch.isdigit()) or "0")
            B=int("".join(ch for ch in b if ch.isdigit()) or "0")
            if A>0 and B>=A: return B-A+1
        return 0
    except Exception as e:
        if verbose: print(f"[crossref/doi] {doi} -> {e}")
        return 0

@lru_cache(maxsize=8192)
def crossref_pages_by_title(title:str, year:Optional[int], verbose:bool=False)->int:
    try:
        data=get_json_cached(CROSSREF, params={"query.bibliographic":title, "rows":6}, timeout=20, ttl=7*24*3600)
        best=0
        for it in data.get("message",{}).get("items",[]):
            pg=it.get("page","") or ""
            if not (isinstance(pg,str) and "-" in pg): continue
            a,b=pg.split("-",1)
            A=int("".join(ch for ch in a if ch.isdigit()) or "0")
            B=int("".join(ch for ch in b if ch.isdigit()) or "0")
            if A<=0 or B<A: continue
            y=None
            if "issued" in it and "date-parts" in it["issued"] and it["issued"]["date-parts"]:
                y=it["issued"]["date-parts"][0][0]
            if year is None or y is None or abs(int(y)-int(year))<=1:
                best=max(best, B-A+1)
        return best
    except Exception as e:
        if verbose: print(f"[crossref/title] {title[:60]} -> {e}")
        return 0

# ---------- detection/normalization ----------
def label_for_url(u:str)->str:
    low=(u or "").lower()
    if "arxiv.org" in low: return "arxiv"
    if "eprint.iacr.org" in low: return "eprint"
    if "eccc.weizmann.ac.il" in low or "eccc.hpi.de" in low: return "eccc"
    return "other"

def to_pdf_url(u:str)->Optional[str]:
    if not u: return None
    u=re.sub(r"^http://","https://", u.strip())
    low=u.lower()
    if "arxiv.org/abs/" in low:
        base=re.sub(r"/abs/","/pdf/",u).split("?",1)[0].rstrip("/")
        return base+".pdf"
    if "arxiv.org/pdf/" in low:
        base=u.split("?",1)[0].rstrip("/")
        return base if base.endswith(".pdf") else (base+".pdf")
    if "eprint.iacr.org" in low:
        base=u.split("?",1)[0].rstrip("/")
        return base if base.endswith(".pdf") else (base+".pdf")
    return u.split("?",1)[0] if low.endswith(".pdf") else None

def extract_doi(links:List[str])->Optional[str]:
    for u in links or []:
        m=DOI_RE.search(u or "")
        if m: return m.group(0)
    return None

def detect_preprint(article:Dict, arxiv_lookup:bool, verbose:bool=False)->Tuple[str, Optional[str], List[str]]:
    links = collect_links(article)

    # scan links + venue/title strings (Scholar may write "arXiv:NNNN.NNNNN" there)
    fields_to_scan = links + [
        article.get("venue", "") or article.get("publication", "") or "",
        article.get("title", "") or ""
    ]

    arxiv_id=None
    for u in fields_to_scan:
        m=ARXIV_URL_RE.search(u or "")
        if m: arxiv_id=m.group(1); break
        m2=re.search(r"arxiv:(\d{4}\.\d{4,5})(v\d+)?", u or "", re.I)
        if m2: arxiv_id=m2.group(1); break

    arxiv_pdf = arxiv_pdf_from_id(arxiv_id) if arxiv_id else None
    preprint  = f"arxiv:{arxiv_id}" if arxiv_id else None

    if not preprint:
        for u in links:
            m=EPRINT_RE.search(u or "")
            if m: preprint=f"eprint:{m.group(1)}/{m.group(2)}"; break
            m=ECCC_RE.search(u or "")
            if m: preprint=f"eccc:{m.group(1)}/{m.group(2)}"; break

    if not arxiv_pdf and arxiv_lookup:
        tpdf=arxiv_title_lookup_pdf(article.get("title") or "", verbose=verbose)
        if tpdf:
            arxiv_pdf=tpdf
            m=re.search(r"/pdf/([^/?#]+)\.pdf", tpdf)
            if m: preprint=f"arxiv:{m.group(1)}"

    return preprint or "None", arxiv_pdf, links

def set_online_fields(arxiv_pdf: Optional[str], links: List[str]) -> Tuple[Optional[str], Optional[str], bool]:
    if arxiv_pdf:
        return arxiv_pdf, "arxiv", True
    for u in links:
        lab = label_for_url(u)
        if lab in ("eprint","eccc","arxiv"):
            pdf=to_pdf_url(u)
            if pdf: return pdf, lab, True
    for u in links:
        pdf=to_pdf_url(u)
        if pdf: return pdf, "other", True
    return None, None, False

def extract_citations(article:Dict)->int:
    cb=article.get("cited_by")
    if isinstance(cb,dict) and "value" in cb:
        try: return int(cb["value"])
        except: pass
    for k in ("cited_by_count","citations","citation_count"):
        if k in article:
            try: return int(article[k])
            except: pass
    return 0

def normalize_one(article:Dict, use_crossref:bool, arxiv_lookup:bool, pages_mode:str, verbose:bool=False)->Dict:
    title = (article.get("title") or "").strip()
    year  = clean_int(article.get("year"))
    venue = (article.get("publication") or article.get("journal") or article.get("venue") or "").strip()
    idurl = (article.get("link") or article.get("result_id") or "").strip()
    citations = extract_citations(article)

    preprint, arxiv_pdf, links = detect_preprint(article, arxiv_lookup, verbose=verbose)

    # Online fields immediately
    pdf_url, pdf_label, has_online = set_online_fields(arxiv_pdf, links)

    pages, source = 0, "unknown"
    trace: List[Tuple[str,str,int]] = []

    # 1) Try arXiv pages first (full fetch)
    if pages_mode!="none" and arxiv_pdf:
        m = re.search(r"/pdf/([^/]+)\.pdf(?:$|[?#])", arxiv_pdf)
        tried = arxiv_pdf_candidates(m.group(1)) if m else [arxiv_pdf]
        for pdf in tried:
            pgs=fetch_pdf_pages(pdf, verbose=verbose)
            trace.append(("arxiv", _normalize_pdf_url(pdf), pgs))
            if pgs>0:
                pages, source = pgs, "arxiv"
                break

    # 2) Other PDFs (full fetch)
    if pages==0 and pages_mode in ("pdf","both"):
        for u in links:
            lab=label_for_url(u)
            pdf=to_pdf_url(u)
            if not pdf or lab=="arxiv": continue
            p=fetch_pdf_pages(pdf, verbose=verbose)
            trace.append((lab or "other", _normalize_pdf_url(pdf), p))
            if p>0:
                pages, source = p, (lab or "other")
                if not pdf_url:
                    pdf_url, pdf_label, has_online = pdf, (lab or "other"), True
                break

    # 3) Crossref fallback
    if pages==0 and use_crossref and pages_mode in ("crossref","both"):
        doi=extract_doi(links) or extract_doi([idurl])
        if doi:
            p=crossref_pages_by_doi(doi, year, verbose=verbose)
            trace.append(("crossref-doi", doi, p))
            if p>0: pages, source = p, "crossref-doi"
        if pages==0 and title:
            p=crossref_pages_by_title(title, year, verbose=verbose)
            trace.append(("crossref-title", title, p))
            if p>0: pages, source = p, "crossref-title"

    return {
        "title": title,
        "year": year,
        "venue": venue,
        "pages": pages,
        "id": idurl or title[:200],
        "preprint": preprint,
        "citations": citations,
        "source_used_for_pages": source,
        "pages_trace": trace,
        "pdf_url": pdf_url,
        "pdf_label": pdf_label,
        "has_online": bool(has_online),
    }

# ---------- per-author build ----------
def build_profile(uid:str, limit:int, use_crossref:bool, arxiv_lookup:bool, pages_mode:str, workers:int, verbose:bool=False)->Dict:
    page0=serp_author_page(uid,0)
    author=page0.get("author") or {}
    raw=serp_list_articles(uid, limit, verbose=verbose)

    out:List[Dict]=[]
    with ThreadPoolExecutor(max_workers=workers) as tp:
        futs=[tp.submit(normalize_one,a,use_crossref,arxiv_lookup,pages_mode,False) for a in raw]
        for f in as_completed(futs):
            r=f.result()
            if r: out.append(r)
    years=sorted({w["year"] for w in out if w.get("year")})
    return {
        "name": author.get("name") or "",
        "institution": author.get("affiliations") or "",
        "scholar_user": uid,
        "start_year": years[0] if years else None,
        "works": out
    }

# ---------- IO helpers (REPLACE behavior) ----------
def load_existing(path:Optional[str])->Dict[str,Any]:
    if not path or not os.path.exists(path): return {}
    try:
        with open(path,"r",encoding="utf-8") as f:
            j=json.load(f)
    except Exception:
        return {}
    return j if isinstance(j,dict) else {}

def replace_into(existing:Dict[str,Any], new_profiles:List[Dict])->Dict[str,Any]:
    def key_of(p:Dict)->str:
        su=(p.get("scholar_user") or "").strip().lower()
        return f"id::{su}" if su else f"name::{(p.get('name') or '').strip().lower()}"
    np_map={ key_of(p): p for p in new_profiles }
    if isinstance(existing,dict) and isinstance(existing.get("printers"),list):
        idx={ key_of(p): p for p in (existing.get("printers") or []) }
        idx.update(np_map)
        printers=list(idx.values())
        years=sorted({w.get("year") for pr in printers for w in (pr.get("works") or []) if w.get("year")})
        return {"printers":printers,"years":years}
    years=sorted({w.get("year") for pr in new_profiles for w in (pr.get("works") or []) if w.get("year")})
    return {"printers":new_profiles,"years":years}

# ---------- CLI ----------
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--id", dest="ids", action="append", required=True, help="Google Scholar author ID(s)")
    ap.add_argument("--in", dest="inp", default=None, help="Existing JSON to replace into (optional)")
    ap.add_argument("--out", dest="out", default="data.json")
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pages", choices=["none","pdf","crossref","both"], default="both",
                    help="Page sources to use (arXiv is always attempted unless 'none')")
    ap.add_argument("--crossref", action="store_true")
    ap.add_argument("--arxiv-lookup", action="store_true")
    ap.add_argument("-v","--verbose", action="store_true")
    return ap.parse_args()

def main():
    args=parse_args()
    require_key()
    new_profiles=[build_profile(uid=u, limit=args.max, use_crossref=args.crossref,
                                arxiv_lookup=args.arxiv_lookup, pages_mode=args.pages,
                                workers=args.workers, verbose=args.verbose) for u in args.ids]
    existing=load_existing(args.inp)
    outdoc=replace_into(existing, new_profiles)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(outdoc,f,ensure_ascii=False,indent=2)
    print(f"[done] wrote {args.out}")

if __name__=="__main__":
    main()
