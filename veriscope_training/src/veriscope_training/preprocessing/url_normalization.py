from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit


DEFAULT_SUSPICIOUS_KEYWORDS = (
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "signin",
    "confirm",
    "password",
    "wallet",
    "bank",
    "auth",
)


@dataclass
class URLNormalizationResult:
    original_url: str | None
    normalized_url: str | None
    features: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_url(
    value: str | None,
    *,
    suspicious_keywords: tuple[str, ...] = DEFAULT_SUSPICIOUS_KEYWORDS,
) -> URLNormalizationResult:
    original_url = value.strip() if isinstance(value, str) else None
    if not original_url:
        return URLNormalizationResult(original_url=None, normalized_url=None, warnings=["missing_url"])

    warnings: list[str] = []
    parsed = urlsplit(original_url)
    if not parsed.netloc and "://" not in original_url:
        parsed = urlsplit(f"//{original_url}")
        warnings.append("scheme_missing_assumed_network_location")

    scheme = parsed.scheme.lower() if parsed.scheme else ""
    hostname = parsed.hostname.lower() if parsed.hostname else None
    if hostname is None:
        return URLNormalizationResult(
            original_url=original_url,
            normalized_url=original_url,
            features={
                "scheme": scheme or None,
                "hostname": None,
                "registered_domain": None,
                "subdomain": None,
                "suffix": None,
                "is_ip_address": False,
                "has_punycode": False,
                "has_percent_encoding": "%" in original_url,
                "url_length": len(original_url),
                "path_length": len(parsed.path or ""),
                "query_length": len(parsed.query or ""),
                "num_subdomains": 0,
                "suspicious_keyword_hits": _keyword_hits(original_url, suspicious_keywords),
                "digit_count": _digit_count(original_url),
                "digit_ratio": _digit_ratio(original_url),
            },
            warnings=warnings + ["hostname_missing"],
        )

    domain_parts = _extract_domain_parts(hostname)
    normalized_url = urlunsplit(
        SplitResult(
            scheme=scheme,
            netloc=_build_normalized_netloc(parsed, hostname),
            path=parsed.path or "",
            query=parsed.query or "",
            fragment=parsed.fragment or "",
        )
    )
    suspicious_hits = _keyword_hits(original_url, suspicious_keywords)
    features = {
        "scheme": scheme or None,
        "hostname": hostname,
        "registered_domain": domain_parts["registered_domain"],
        "subdomain": domain_parts["subdomain"],
        "suffix": domain_parts["suffix"],
        "port": parsed.port,
        "is_ip_address": _is_ip_address(hostname),
        "has_punycode": "xn--" in hostname,
        "has_percent_encoding": bool(re.search(r"%[0-9a-fA-F]{2}", original_url)),
        "url_length": len(original_url),
        "path_length": len(parsed.path or ""),
        "query_length": len(parsed.query or ""),
        "num_subdomains": _subdomain_count(domain_parts["subdomain"]),
        "path_depth": len([part for part in (parsed.path or "").split("/") if part]),
        "query_param_count": (parsed.query.count("&") + 1) if parsed.query else 0,
        "suspicious_keyword_hits": suspicious_hits,
        "suspicious_keyword_count": len(suspicious_hits),
        "digit_count": _digit_count(original_url),
        "digit_ratio": _digit_ratio(original_url),
        "hostname_digit_count": _digit_count(hostname),
        "contains_at_symbol": "@" in original_url,
        "contains_double_slash_path": "//" in (parsed.path or ""),
        "contains_hyphenated_hostname": "-" in hostname,
    }
    return URLNormalizationResult(
        original_url=original_url,
        normalized_url=normalized_url or original_url,
        features=features,
        warnings=warnings,
    )


def _build_normalized_netloc(parsed: SplitResult, hostname: str) -> str:
    credentials = ""
    if parsed.username:
        credentials = parsed.username
        if parsed.password:
            credentials = f"{credentials}:{parsed.password}"
        credentials = f"{credentials}@"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{credentials}{hostname}{port}"


def _extract_domain_parts(hostname: str) -> dict[str, str | None]:
    try:
        import tldextract

        extractor = tldextract.TLDExtract(suffix_list_urls=None)
        extracted = extractor(hostname)
        registered_domain = ".".join(part for part in (extracted.domain, extracted.suffix) if part) or None
        subdomain = extracted.subdomain or None
        suffix = extracted.suffix or None
    except Exception:
        labels = [part for part in hostname.split(".") if part]
        if len(labels) >= 2:
            registered_domain = ".".join(labels[-2:])
            subdomain = ".".join(labels[:-2]) or None
            suffix = labels[-1]
        else:
            registered_domain = hostname
            subdomain = None
            suffix = None
    return {
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "suffix": suffix,
    }


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return False
    return True


def _keyword_hits(value: str, keywords: tuple[str, ...]) -> list[str]:
    lowered = value.lower()
    return sorted({keyword for keyword in keywords if keyword in lowered})


def _digit_count(value: str) -> int:
    return sum(character.isdigit() for character in value)


def _digit_ratio(value: str) -> float:
    if not value:
        return 0.0
    return round(_digit_count(value) / len(value), 6)


def _subdomain_count(subdomain: str | None) -> int:
    if not subdomain:
        return 0
    return len([part for part in subdomain.split(".") if part])
