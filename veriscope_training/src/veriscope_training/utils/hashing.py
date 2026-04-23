from __future__ import annotations

import hashlib


def sha256_text(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def make_sample_id(
    *,
    source: str,
    original_id: str | None = None,
    url: str | None = None,
    html: str | None = None,
    text: str | None = None,
) -> str:
    components = [
        source.strip(),
        (original_id or "").strip(),
        (url or "").strip(),
        sha256_text(html or "") or "",
        sha256_text(text or "") or "",
    ]
    payload = "||".join(components)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
