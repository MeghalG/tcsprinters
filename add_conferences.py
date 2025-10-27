#!/usr/bin/env python3
# Add/overwrite a "conference" field on each work based on its venue (and preprint status).
# Also add a numeric "conference_rank" for rough prestige sorting (lower = more prestigious),
# and a top-level "conference_order" list you can use in the UI if desired.
#
# Output examples:
#   "FOCS 2024", "STOC 2023", "APPROX/RANDOM 2023", "CRYPTO 2022", "preprint", "Other", "TOCT"
#
# Usage:
#   python add_conferences.py --in data.json --out data.json

import argparse, json, re
from typing import Optional, List, Set, Tuple, Dict

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# ---- Canonical conference regexes ----
# Each tuple: (canonical_label, [regex patterns])
CONFS: List[Tuple[str, List[str]]] = [
    ("FOCS",  [r"\bFOCS\b", r"foundations of computer science"]),
    ("STOC",  [r"\bSTOC\b", r"symposium on theory of computing", r"acm.*(stoc|symposium on theory of computing)"]),
    ("SODA",  [r"\bSODA\b", r"siam.*discrete algorithms", r"acm[- ]?siam.*discrete algorithms"]),
    ("CRYPTO", [
        r"\bcrypt(o)?\b",                # Crypto / CRYPTO
        r"iacr.*crypto",                 # IACR Crypto
        r"international cryptology conference",
    ]),
    ("EUROCRYPT", [
        r"\beurocrypt\b",
        r"european cryptology conference",
    ]),
    ("ASIACRYPT", [
        r"\basiacrypt\b",
        r"asia[- ]?pacific cryptology conference",
    ]),
    ("ITCS",  [r"\bITCS\b", r"innovations in theoretical computer science"]),
    ("ICALP", [r"\bICALP\b", r"automata[, ]+languages[, ]+and programming", r"international colloquium on automata"]),
    ("CCC",   [r"\bCCC\b", r"computational complexity conference", r"ieee.*conference on computational complexity"]),
    ("COLT",  [r"\bCOLT\b", r"conference on learning theory"]),
    ("NeurIPS",[r"\bneurips\b", r"\bnips\b", r"neural information processing systems"]),
    ("ICML",  [r"\bICML\b", r"international conference on machine learning"]),
    ("IJCAI", [r"\bIJCAI\b", r"international joint conference on artificial intelligence"]),
    ("ALT",   [r"\bALT\b", r"algorithmic learning theory"]),
    ("RANDOM",[
        r"\brandom\b(?! access)",
        r"\brandomization\b",
        r"\bapprox/?random\b",
        r"\bapprox[-–—]*random\b",
        r"randomization and computation",
        r"approximation,?\s*randomization,?\s*and\s*combinatorial optimization",
    ]),
    ("APPROX",[
        r"\bapprox\b",
        r"\bapproximation\b",
        r"\bapprox/?random\b",
        r"\bapprox[-–—]*random\b",
        r"approximation,?\s*randomization,?\s*and\s*combinatorial optimization",
        r"algorithms and(?:\s*combinatorial)? optimization",
    ]),
    ("ICLR",  [r"\bICLR\b", r"international conference on learning representations"]),
    ("PODS",  [r"\bPODS\b", r"principles of database systems", r"sigmod[- ]sigact[- ]sigai.*principles of database systems"]),
    ("SIGMOD",[r"\bSIGMOD\b", r"international conference on management of data", r"acm.*sigmod"]),
]

# Preprint hints
PREPRINT_HINTS = [
    r"\barxiv\b", r"arxiv\.org", r"eprint\.iacr\.org",
    r"\beccc\b", r"electronic colloquium on computational complexity",
    r"\bpreprint\b",
]

# ---- Prestige / rough ordering (lower = better) ----
# STOC/FOCS > SODA > CRYPTO > EUROCRYPT/ASIACRYPT > (CCC/COLT/ITCS/ICALP) > (NeurIPS/ICML/ICLR/IJCAI/ALT) > (PODS/SIGMOD) > APPROX/RANDOM > TOCT > preprint > Other
PRESTIGE_ORDER: List[List[str]] = [
    ["FOCS", "STOC"],                  # 0
    ["SODA"],                          # 1
    ["CRYPTO"],                        # 2
    ["EUROCRYPT", "ASIACRYPT"],        # 3
    ["CCC", "COLT", "ITCS", "ICALP"],  # 4
    ["NeurIPS", "ICML", "ICLR", "IJCAI", "ALT"],  # 5
    ["PODS", "SIGMOD"],                # 6
    ["APPROX", "RANDOM"],              # 7
    ["TOCT"],                          # 8  (journal)
    ["preprint"],                      # 9
    ["Other"],                         # 10
]
# Build a rank map
CONF_RANK: Dict[str, int] = {}
for i, group in enumerate(PRESTIGE_ORDER):
    for lab in group:
        CONF_RANK[lab] = i

def find_year(text: str) -> Optional[str]:
    if not text: return None
    m = YEAR_RE.search(text)
    return m.group(0) if m else None

def match_confs(venue: str) -> Set[str]:
    """Return set of canonical conference labels seen in venue."""
    v = (venue or "").lower()
    found: Set[str] = set()
    for label, pats in CONFS:
        for p in pats:
            if re.search(p, v, flags=re.I):
                found.add(label)
                break
    return found

def is_preprint(venue: str, preprint_field: str) -> bool:
    v = (venue or "").lower()
    if preprint_field and str(preprint_field).strip().lower() != "none":
        return True
    for p in PREPRINT_HINTS:
        if re.search(p, v, flags=re.I):
            return True
    return False

def classify_work(venue: str, preprint_field: str, work_year: Optional[int] = None) -> str:
    """
    Returns one of:
      - "<CONF> <YEAR>" or "APPROX/RANDOM <YEAR>"
      - "preprint"
      - "Other"
    """
    confs = match_confs(venue or "")
    year_from_venue = find_year(venue or "")
    year = year_from_venue or (str(work_year) if work_year else None)

    # Combine APPROX & RANDOM if both appear
    if "APPROX" in confs and "RANDOM" in confs:
        return f"APPROX/RANDOM {year}" if year else "APPROX/RANDOM"

    if confs:
        # If multiple labels, pick the most prestigious one
        best = sorted(confs, key=lambda c: CONF_RANK.get(c, 9999))[0]
        return f"{best} {year}" if year else best

    if is_preprint(venue or "", preprint_field or ""):
        return "preprint"

    return "Other"

def rank_for_conference_label(label: str) -> int:
    """Numeric rank for a canonical label (no year)."""
    # Normalize combined label
    if label.startswith("APPROX/RANDOM"):
        return CONF_RANK.get("APPROX", 9999)  # same tier as APPROX/RANDOM
    return CONF_RANK.get(label, 9999)

def extract_canonical(label_with_year: str) -> str:
    """Strip the year to get the canonical label."""
    if not label_with_year:
        return "Other"
    # Combined special case
    if label_with_year.startswith("APPROX/RANDOM"):
        return "APPROX/RANDOM"
    # Take the leading alphabetic (and slash) token
    m = re.match(r"^([A-Za-z/]+)", label_with_year)
    return (m.group(1) if m else label_with_year).strip()

def process(payload: dict) -> dict:
    """Add/overwrite 'conference' and 'conference_rank' per work; write 'conference_order'."""
    # Update works
    for p in payload.get("printers", []):
        for w in p.get("works", []):
            venue = w.get("venue") or ""
            pre   = w.get("preprint") or ""
            conf_label = classify_work(venue, pre, w.get("year"))
            w["conference"] = conf_label
            # Rank is by canonical (no year)
            canonical = extract_canonical(conf_label)
            w["conference_rank"] = rank_for_conference_label(canonical)

    # Build an ordered list of conferences present, sorted by rank then alpha
    seen: Set[str] = set()
    for p in payload.get("printers", []):
        for w in p.get("works", []):
            lab = extract_canonical(w.get("conference", "Other"))
            seen.add(lab)
    ordered = sorted(seen, key=lambda lab: (rank_for_conference_label(lab), lab))
    payload["conference_order"] = ordered

    return payload

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="infile", required=True, help="Input data.json")
    ap.add_argument("--out", dest="outfile", required=True, help="Output data.json")
    args = ap.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out = process(payload)

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.outfile}")

if __name__ == "__main__":
    main()
