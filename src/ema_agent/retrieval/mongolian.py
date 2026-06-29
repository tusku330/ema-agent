"""Mongolian-aware BM25 tokenization: a stemmer + stopwords "language pack".

bm25s's default pipeline tokenizes Cyrillic fine (the `(?u)` token pattern treats
Cyrillic as word chars) but then applies the *English* Snowball stemmer, which is
a no-op on Mongolian. Mongolian is agglutinative: татвар, татвараа, татварын,
татвартай share one stem but the default tokenizer emits four distinct tokens, so
the lexical half of the hybrid retriever barely fires for inflected queries. The
custom stemmer below strips the most common nominal case/number/possessive
suffixes so inflected forms collapse to a shared stem at BOTH index and query time
(bm25s reuses the stemmer for the query, so the two stay consistent).

This is a deliberately conservative, dependency-free heuristic stemmer — not a
full morphological analyzer. It strips at most ONE suffix and only when a stem of
at least ``MN_MIN_STEM`` characters remains, to avoid over-stemming short words
into collisions.

To retrieve in another language, swap this module out: pass your own
``stemmer`` / ``stopwords`` to ``HybridRetriever``.
"""

from __future__ import annotations

MN_MIN_STEM = 3

# Suffixes ordered LONGEST-FIRST so the longest valid match wins (e.g. "ийн"
# before "н"). Grouped by grammatical role for readability. Verb morphology is
# intentionally excluded — it's far riskier for over-stemming than nominal cases.
MN_SUFFIXES = [
    # plural / collective
    "нуудаа", "нүүдээ", "чуудаа", "чүүдээ",
    "нууд", "нүүд", "чууд", "чүүд", "ууд", "үүд", "нар", "нэр",
    # ablative / instrumental / comitative (3-char vowel-harmony variants)
    "аас", "ээс", "оос", "өөс",
    "аар", "ээр", "оор", "өөр",
    "тай", "тэй", "той",
    "руу", "рүү", "луу", "лүү",
    # genitive
    "гийн", "ийн", "ний", "ын", "ий",
    # accusative
    "ийг", "ыг",
    # dative-locative
    "нд", "ад", "эд", "од", "өд",
    # reflexive possessive
    "аа", "ээ", "оо", "өө",
    # short single-char case markers (last — only strip if a real stem remains)
    "г", "д", "т", "н",
]


def mn_stem_word(token: str) -> str:
    """Strip at most one common Mongolian nominal suffix, longest match first."""
    for suf in MN_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= MN_MIN_STEM:
            return token[: -len(suf)]
    return token


def mn_stemmer(tokens: list[str]) -> list[str]:
    """bm25s stemmer contract: list[str] -> list[str]."""
    return [mn_stem_word(t) for t in tokens]


# A small Mongolian stopword set (particles/copulas that carry no retrieval
# signal). Passed as ``language=`` because BM25Retriever forwards it straight to
# bm25s.tokenize's ``stopwords`` arg, which accepts a custom list.
MN_STOPWORDS = [
    "ба", "буюу", "болон", "нь", "юм", "вэ", "бэ", "уу", "үү",
    "энэ", "тэр", "гэж", "гэх", "байна", "бол", "мөн", "тухай",
]
