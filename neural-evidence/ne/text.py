"""Text normalisation helpers for FEVER's tokenised Wikipedia dump."""
import hashlib
import re
import unicodedata

_PTB = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":",
    "``": '"', "''": '"',
}
_PTB_RE = re.compile("|".join(re.escape(k) for k in _PTB))


def clean_wiki_text(text: str) -> str:
    """Undo FEVER/PTB tokenisation so evidence reads like normal prose."""
    text = _PTB_RE.sub(lambda m: _PTB[m.group(0)], text)
    text = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$])\s+", r"\1", text)
    text = re.sub(r"\s+'(s|re|ve|d|ll|m|t)\b", r"'\1", text)
    text = re.sub(r"\s+n't\b", "n't", text)
    return re.sub(r"\s+", " ", text).strip()


def page_title_to_text(page_id: str) -> str:
    """'Barack_Obama' -> 'Barack Obama' (with PTB brackets restored)."""
    return clean_wiki_text(page_id.replace("_", " "))


def normalize_claim(claim: str) -> str:
    """Canonical form used to detect duplicate claim texts across claim IDs."""
    claim = unicodedata.normalize("NFKC", claim).lower()
    claim = re.sub(r"[^\w\s]", " ", claim)
    return re.sub(r"\s+", " ", claim).strip()


def claim_group(claim: str) -> str:
    """Stable group key: identical (normalised) claim texts share a key."""
    return hashlib.sha1(normalize_claim(claim).encode("utf-8")).hexdigest()[:16]


def parse_wiki_lines(lines_field: str) -> list:
    """Parse the `lines` field of a FEVER wiki page into a list of sentences.

    Each row is "<sent_id>\\t<sentence>\\t<hyperlink>\\t...". Sentence IDs are
    positional, so the returned list is indexed by sentence id ('' if empty).
    """
    sentences = []
    for row in lines_field.split("\n"):
        parts = row.split("\t")
        if not parts or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        while len(sentences) < idx:
            sentences.append("")
        sentences.append(parts[1] if len(parts) > 1 else "")
    return sentences
