from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit


VISIBLE_TEXT_EXCLUDED_TAGS = ("script", "style", "noscript", "template", "svg", "canvas")
ACTION_WORDS = ("login", "sign in", "verify", "confirm", "submit", "continue", "unlock", "update")


@dataclass
class HTMLExtractionResult:
    visible_text: str | None
    features: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_html_content(
    html: str | None,
    *,
    base_url: str | None = None,
    max_action_texts: int = 20,
) -> HTMLExtractionResult:
    if not html:
        return HTMLExtractionResult(visible_text=None, warnings=["missing_html"])

    warnings: list[str] = []
    try:
        from bs4 import BeautifulSoup
        from bs4 import Comment
        from bs4.builder import ParserRejectedMarkup
    except ImportError:
        return HTMLExtractionResult(
            visible_text=_basic_strip_tags(html),
            features={"parser": "regex_fallback", "html_length": len(html)},
            warnings=["beautifulsoup_not_available"],
        )

    parser = "lxml"
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        parser = "html.parser"
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            return HTMLExtractionResult(
                visible_text=_basic_strip_tags(html),
                features={"parser": "regex_fallback", "html_length": len(html)},
                warnings=[f"html_parse_failed:{type(exc).__name__}"],
            )

    title = soup.title.get_text(" ", strip=True) if soup.title else None
    form_count = len(soup.find_all("form"))
    password_inputs = soup.find_all("input", attrs={"type": lambda value: value and value.lower() == "password"})
    hidden_inputs = soup.find_all("input", attrs={"type": lambda value: value and value.lower() == "hidden"})
    iframe_count = len(soup.find_all("iframe"))
    script_count = len(soup.find_all("script"))
    meta_refresh_present = any(
        (meta.get("http-equiv") or "").strip().lower() == "refresh" for meta in soup.find_all("meta")
    )
    action_texts = _extract_action_texts(soup, limit=max_action_texts)
    external_link_count = _external_link_count(soup, base_url=base_url)
    body = soup.body or soup

    for tag_name in VISIBLE_TEXT_EXCLUDED_TAGS:
        for tag in body.find_all(tag_name):
            tag.decompose()
    for comment in body.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()

    visible_text = _normalize_visible_text(body.get_text(separator=" ", strip=True))
    features = {
        "parser": parser,
        "html_length": len(html),
        "title": title,
        "title_length": len(title) if title else 0,
        "form_count": form_count,
        "password_input_count": len(password_inputs),
        "has_password_input": bool(password_inputs),
        "iframe_count": iframe_count,
        "script_count": script_count,
        "external_link_count": external_link_count,
        "hidden_input_count": len(hidden_inputs),
        "meta_refresh_present": meta_refresh_present,
        "action_texts": action_texts,
        "action_word_hits": [word for word in ACTION_WORDS if any(word in text.lower() for text in action_texts)],
        "visible_text_length": len(visible_text) if visible_text else 0,
    }
    return HTMLExtractionResult(
        visible_text=visible_text or None,
        features=features,
        warnings=warnings,
    )


def _basic_strip_tags(html: str) -> str:
    return _normalize_visible_text(re.sub(r"<[^>]+>", " ", html))


def _normalize_visible_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_action_texts(soup: Any, *, limit: int) -> list[str]:
    texts: list[str] = []
    for tag in soup.find_all(["button", "a", "input"]):
        if tag.name == "input":
            text = (tag.get("value") or tag.get("aria-label") or "").strip()
        else:
            text = tag.get_text(" ", strip=True)
        if not text:
            continue
        texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _external_link_count(soup: Any, *, base_url: str | None) -> int:
    base_host = urlsplit(base_url).hostname.lower() if base_url and urlsplit(base_url).hostname else None
    count = 0
    for tag in soup.find_all(["a", "link"], href=True):
        href = tag.get("href")
        if not href:
            continue
        absolute = urljoin(base_url or "", href)
        host = urlsplit(absolute).hostname
        if host and (base_host is None or host.lower() != base_host):
            count += 1
    return count
