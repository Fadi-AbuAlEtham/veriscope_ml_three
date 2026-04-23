from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any


URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.\w+\b")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")
URGENCY_TERMS = ("urgent", "verify", "confirm", "suspend", "login", "password", "reset", "security")


@dataclass(frozen=True)
class TextNormalizationConfig:
    unicode_normalization: str | None = "NFKC"
    lowercase: bool = False
    collapse_whitespace: bool = True
    strip_control_characters: bool = True
    preserve_urls: bool = True
    preserve_emails: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "TextNormalizationConfig":
        mapping = mapping or {}
        return cls(
            unicode_normalization=mapping.get("unicode_normalization", "NFKC"),
            lowercase=bool(mapping.get("lowercase", False)),
            collapse_whitespace=bool(mapping.get("collapse_whitespace", True)),
            strip_control_characters=bool(mapping.get("strip_control_characters", True)),
            preserve_urls=bool(mapping.get("preserve_urls", True)),
            preserve_emails=bool(mapping.get("preserve_emails", True)),
        )


@dataclass
class TextProcessingResult:
    original_text: str | None
    normalized_text: str | None
    features: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(
    value: str | None,
    *,
    config: TextNormalizationConfig | None = None,
) -> TextProcessingResult:
    text = value.strip() if isinstance(value, str) else None
    if not text:
        return TextProcessingResult(original_text=None, normalized_text=None, warnings=["missing_text"])

    config = config or TextNormalizationConfig()
    normalized = text
    warnings: list[str] = []
    if config.unicode_normalization:
        try:
            normalized = unicodedata.normalize(config.unicode_normalization, normalized)
        except ValueError:
            warnings.append("invalid_unicode_normalization_form")
    if config.strip_control_characters:
        normalized = CONTROL_CHARS.sub(" ", normalized)
    if config.collapse_whitespace:
        normalized = WHITESPACE.sub(" ", normalized).strip()
    if config.lowercase:
        normalized = normalized.lower()

    features = {
        "original_length": len(text),
        "normalized_length": len(normalized),
        "token_count": len(normalized.split()),
        "url_count": len(URL_PATTERN.findall(text)),
        "email_count": len(EMAIL_PATTERN.findall(text)),
        "urgency_terms": [term for term in URGENCY_TERMS if term in normalized.lower()],
        "config": asdict(config),
    }
    return TextProcessingResult(
        original_text=text,
        normalized_text=normalized or None,
        features=features,
        warnings=warnings,
    )
