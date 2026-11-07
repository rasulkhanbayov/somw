"""Copy the Claude Code session record of this project, with secrets removed, for the PRIVATE repo only.

The record holds every command, inline analysis and output of the project (Oct 10 - Nov 3, 2026), including numbers
that were first computed inline. It also contains the PokeFlex download link (a personal token) and may contain the
user's e-mail address or other tokens, so it is redacted: PokeFlex `dt=` tokens, Hugging Face / GitHub / OpenAI-style
tokens and e-mail addresses are replaced. The script then searches the output again and stops if any pattern remains.
Output: $BIG_ROOT/freeze_private/claude_session/<file> (jsonl files and the tool-results folder, text only)
Usage: python freeze/redact_transcript.py [--src ~/.claude/projects/-export-home-rakh90100-cvpr-physics-plausibility]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from areas import PRIVATE_STAGE  # noqa: E402

PATTERNS = [
    (re.compile(r"dt=[0-9a-f]{20,}"), "dt=<REDACTED_POKEFLEX_TOKEN>"),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "<REDACTED_HF_TOKEN>"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"sk-[A-Za-z0-9_-]{30,}"), "<REDACTED_API_KEY>"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}"), "<REDACTED_EMAIL>"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(Path.home() / ".claude/projects/-export-home-rakh90100-cvpr-physics-plausibility"))
    args = ap.parse_args()
    src, out = Path(args.src), PRIVATE_STAGE / "claude_session"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file() or "memory" in p.relative_to(src).parts:
            continue
        try:
            s = p.read_text()
        except UnicodeDecodeError:                      # binary tool results (images, html blobs) are skipped
            continue
        for rx, rep in PATTERNS:
            s = rx.sub(rep, s)
        t = out / p.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(s)
        n += 1
    left = [(t, rx.pattern) for t in out.rglob("*") if t.is_file() for rx, _ in PATTERNS if rx.search(t.read_text())]
    if left:
        sys.exit(f"redaction incomplete: {left[:5]}")
    print(f"[redact] {n} files -> {out}; no token or e-mail pattern remains")


if __name__ == "__main__":
    main()
