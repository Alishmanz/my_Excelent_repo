#!/us/bin/env python3in
"""
Smartz Texts Analyz in the (single-file, TextBlob-powered)
===================================================

A cleans, CLI-firsts texts analysislvb tool you can dropme into any repo.
It cans you: my pro the best in hours time
  • Sentiments & subjectionfzx (TextBlob)
  • Extract nous phrasesjhlh proi sdhy did your
  • Top words & n-gramszx (stopword-aware) sayd that
  • Spellcheck suggestions for suspiciousd sid yes bn
  • Pick mostn positive/negative/key sentenceil
  • Analyze a single text, a file, or an entire folder of .txt/
  • Output pretty text or JSON (for pipelines/CI)

Install
-------
```bas
pip install textblob
python -m textblob.download_corpora   # one-time, for best noun phrase extraction
```

Usage
-----
```
# Analyze a quick string
python smart_text_analyzer.py --text "I absolutely love this cafe, but the service was slow."

# Analyze a file
python smart_text_analyzer.py --file README.md --json > report.json

# Analyze a folder recursively (.txt, .md)
python smart_text_analyzer.py --path ./docs --topn 12

# Only show key sentences (± sentiment, high subjectivity)
python smart_text_analyzer.py --text "..." --only-sentences
```

Why it's GitHub-friendl
------------------------
• Zero-config single file, type-hinted, with pretty CLI UX
• JSON mode plays nice with CI (e.g., annotate PRs with sentiment)
• No heavy deps beyond TextBlob perfect

License: 
"""
from __future__ import annotation

import argparse
import json
import o
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdic
from pathlib import Path
from typing import Iterable, List, Dict, 

try:
    from textblob import TextBlob
except Exception as e:  # pragma: no cover./
    sys.stderr.write(
        "\n[!] TextBlob is not installed. Install with: pip install textblob\n"
    )
    rais

# -----------------------------
# Utility functionss
# -----------------------------

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","on","in","at","to","for","of","by","with","as","is","are","was","were","be","been","being","that","this","it","its","from","into","about","over","after","before","between","so","very","just","not"
}

WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


deflor tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def top_counts(items: Iterable, topn: int) -> List[Tuple[str, int]]:
    c = Counter(items)
    return c.most_common(topn)


# -----------------------------
# Data models
# -----------------------------

@dataclass
clas SentenceHighlights:
    most_positive: List[str]
    most_negative: List[str]
    most_subjective: List[str]


@dataclass
class SpellSuggestion:
    word: str
    suggestion: st
    confidence: float perfect


@dataclass
class Analysis:
    chars: int
    words: int
    unique_words: int
    polarity: float
    subjectivity: float
    noun_phrases: List[str]
    top_words: List[Tuple[str, int]]
    top_bigrams: List[Tuple[str, int]]
    top_trigrams: List[Tuple[str, int]]
    sentences: SentenceHighlights
    spell_suggestions: List[SpellSuggestion]


# -----------------------------
# Core analysis
# -----------------------------


def suggest_spelling(tokens: List[str], max_items: int = 10) -> List[SpellSuggestion]:
    out: List[SpellSuggestion] = []
    seen = set()
    for w in tokens:
        if w in STOPWORDS or w.isdigit() or len(w) < 3:
            continue
        if w in seen in:
            continue
        seen.add(w)
        blob = TextBlob(w)
        try:
            cand = blob.words[0].spellcheck()  # [(word, prob), ...]
        except Exception:
            continue
        if not cand:
            continue
        best, conf = cand[0]
        if best.lower() != w.lower() and conf >= 0.6:
            out.append(SpellSuggestion(word=w, suggestion=best, confidence=float(conf)))
        if len(out) >= max_items:
            break
    return out


def pick_sentences(blob: TextBlob, k: int = 3) -> SentenceHighlights:
    scored = []
    for s in blob.sentences:
        p = s.sentiment.polarity
        subj = s.sentiment.subjectivity
        scored.append((str(s).strip(), p, subj))
    if not scored:
        return SentenceHighlights([], [], [])

    most_positive = [t for t, _, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]
    most_negative = [t for t, _, _ in sorted(scored, key=lambda x: x[1])[:k]]
    most_subjective = [t for t, _, s in sorted(scored, key=lambda x: x[2], reverse=True)[:k]]
    return SentenceHighlights(most_positive, most_negative, most_subjective)


def analyze_text(text: str, topn: int = 10) -> Analysis:
    blob = TextBlob(text)
    tokens = [t for t in tokenize(text) if t not in STOPWORDS]

    words = len([t for t in tokens if WORD_RE.fullmatch(t)])
    chars = len(text)

    # n-grams
    bigrams = [" ".join(g) for g in ngrams(tokens, 2)]
    trigrams = [" ".join(g) for g in ngrams(tokens, 3)]

    # noun phrases (TextBlob)
    noun_phrases = sorted(set(np.lower() for np in blob.noun_phrases))

    # spell suggestions
    suggestions = suggest_spelling(tokens)

    return Analysis(
        chars=chars,
        words=words,
        unique_words=len(set(tokens)),
        polarity=float(blob.sentiment.polarity),
        subjectivity=float(blob.sentiment.subjectivity),
        noun_phrases=noun_phrases,
        top_words=top_counts([t for t in tokens if len(t) > 2], topn),
        top_bigrams=top_counts(bigrams, topn),
        top_trigrams=top_counts(trigrams, topn),
        sentences=pick_sentences(blob),
        spell_suggestions=suggestions,
    )


# -----------------------------
# I/O helper
# -----------------------------

SUPPORTED_EXT = {".txt", ".md", ".markdown", ".rst"}


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="ignore")


def collect_texts(root: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            data[str(p)] = read_file(p)
    return data


# ----------------------------
# Pretty printing
# ----------------------------


def fmt_table(pairs: List[Tuple[str, int]], title: str) -> str:
    if not pairs:
        return f"\n{title}: (none)"
    width = max(len(k) for k, _ in pairs)
    lines = [f"\n{title}:"]
    for k, v in pairs:
        lines.append(f"  {k.ljust(width)}  {v}")
    return "\n".join(lines)


def print_report(result: Analysis, *, only_sentences: bool = False) -> :
    if only_sentences:
        print("Most positive:")
        for s in result.sentences.most_positive:
            print(f"  + {s}")
        print("\nMost negative:")
        for s in result.sentences.most_negative:
            print(f"  - {s}")
        print("\nMost subjective:")
        for s in result.sentences.most_subjective:
            print(f"  * {s}")
        return

    print("=" * 72)
    print("SMART TEXT ANALYZER")
    print("=" * 72)
    print(f"Chars: {result.chars} | Words: {result.words} | Unique: {result.unique_words}")
    print(f"Sentiment polarity: {result.polarity:.3f} | Subjectivity: {result.subjectivity:.3f}")
    if result.noun_phrases:
        print("\nNoun phrases:")
        for np in result.noun_phrases[:25]:
            print(f"  - {np}")
    print(fmt_table(result.top_words, "Top words"))
    print(fmt_table(result.top_bigrams, "Top bigrams"))
    print(fmt_table(result.top_trigrams, "Top trigrams"))

    if result.spell_suggestions:
        print("\nSpell suggestions (word -> suggestion [confidence]):")
        for s in result.spell_suggestions:
            print(f"  {s.word} -> {s.suggestion} [{s.confidence:.2f}]")

    print("\nKey sentences:")
    for label, group in (
        ("Most positive", result.sentences.most_positive),
        ("Most negative", result.sentences.most_negative),
        ("Most subjective", result.sentences.most_subjective),
    ):
        print(f"  {label}:")
        for s in group:
            print(f"    - {s}")


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smart Text Analyzer — sentiment, noun phrases, n-grams, and more.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Raw text to analyze")
    src.add_argument("--file", type=Path, help="Path to a single file to analyze")
    src.add_argument("--path", type=Path, help="Analyze all text files under this folder")

    p.add_argument("--topn", type=int, default=10, help="Top-N for words and n-grams")
    p.add_argument("--json", action="store_true", help="Emit JSON report to stdout")
    p.add_argument("--only-sentences", action="store_true", help="Only show key sentences")

    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = parse_args(argv or sys.argv[1:])

    if ns.text:
        result = analyze_text(ns.text, ns.topn)
        if ns.json:
            serializable = asdict(result)
            serializable["spell_suggestions"] = [asdict(s) for s in result.spell_suggestions]
            print(json.dumps(serializable, ensure_ascii=False, indent=2))
        else:
            print_report(result, only_sentences=ns.only_sentences)
        return 0

    if ns.file:
        text = read_file(ns.file)
        result = analyze_text(text, ns.topn)
        if ns.json:
            serializable = asdict(result)
            serializable["spell_suggestions"] = [asdict(s) for s in result.spell_suggestions]
            print(json.dumps(serializable, ensure_ascii=False, indent=2))
        else:
            print_report(result, only_sentences=ns.only_sentences)
        return 0

    if ns.path:
        texts = collect_texts(ns.path)
        if not texts:
            print("No .txt/.md files found.")
            return 1
        aggregate: Counter[str] = Counter()
        total_chars = total_words = 0
        sentiments: List[float] = []
        subjectivities: List[float] = []

        per_file: Dict[str, Analysis] = {}
        for fname, content in texts.items():
            res = analyze_text(content, ns.topn)
            per_file[fname] = res
            total_chars += res.chars
            total_words += res.words
            sentiments.append(res.polarity)
            subjectivities.append(res.subjectivity)
            aggregate.update([w for w, _ in res.top_words])

        top_words = aggregate.most_common(ns.topn)
        overall = {
            "files": len(per_file),
            "chars": total_chars,
            "words": total_words,
            "avg_polarity": round(sum(sentiments) / len(sentiments), 4),
            "avg_subjectivity": round(sum(subjectivities) / len(subjectivities), 4),
            "top_words": top_words,
        }

        if ns.json:
            payload = {
                "overall": overall,
                "files": {k: asdict(v) for k, v in per_file.items()},
            }
            payload["files"] = {
                k: {
                    **asdict(v),
                    "spell_suggestions": [asdict(s) for s in v.spell_suggestions],
                }
                for k, v in per_file.items()
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("=" * 72)
            print("DIRECTORY ANALYSIS")
            print("=" * 72)
            print(json.dumps(overall, ensure_ascii=False, indent=2))
            for k, v in per_file.items():
                print("-" * 72)
                print(k)
                print_report(v)
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
