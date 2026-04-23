from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veriscope_training.adaptive.feedback_schema import load_feedback_records
from veriscope_training.utils.hashing import make_sample_id
from veriscope_training.utils.io import read_records_file, write_json, read_jsonl


TOKEN_RE = re.compile(r"\b[\w\-]{4,}\b", re.UNICODE)


def generate_heuristic_proposals(
    *,
    processed_path: str | Path,
    output_path: str | Path,
    feedback_path: str | Path | None = None,
    top_k: int = 25,
) -> dict[str, Any]:
    rows = list(read_records_file(processed_path))
    feedback_rows = load_feedback_records(feedback_path) if feedback_path else []
    feedback_by_sample = {row.get("sample_id"): row for row in feedback_rows if row.get("sample_id")}
    phishing_rows = [row for row in rows if _is_positive(row, feedback_by_sample)]
    benign_rows = [row for row in rows if _is_negative(row, feedback_by_sample)]

    proposals = []
    proposals.extend(_phrase_proposals(phishing_rows, benign_rows, top_k=top_k))
    proposals.extend(_cta_proposals(phishing_rows, benign_rows, top_k=max(10, top_k // 2)))
    proposals.extend(_url_style_proposals(phishing_rows, benign_rows))
    proposals.extend(_html_pattern_proposals(phishing_rows, benign_rows))
    proposals.sort(key=lambda item: (item["supporting_count"], item.get("estimated_precision_proxy", 0.0)), reverse=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processed_path": str(processed_path),
        "feedback_path": str(feedback_path) if feedback_path else None,
        "proposal_count": len(proposals),
        "proposals": proposals[:top_k],
    }
    write_json(output_path, payload)
    return payload


def generate_proposals_from_errors(
    *,
    error_analysis_dir: str | Path,
    output_path: str | Path,
    top_k: int = 25,
) -> dict[str, Any]:
    error_dir = Path(error_analysis_dir)
    false_positives = list(read_jsonl(error_dir / "false_positives.jsonl"))
    false_negatives = list(read_jsonl(error_dir / "false_negatives.jsonl"))

    proposals = []

    # 1. phishing_signal from false negatives (missed phishing)
    fn_tokens = _extract_recurring_tokens(false_negatives)
    for token, count in fn_tokens.most_common(top_k):
        proposals.append(
            _proposal(
                rule_type="phishing_signal",
                candidate_pattern=token,
                rationale=f"Recurring token '{token}' appears in missed phishing samples.",
                supporting_examples=_get_examples_for_token(false_negatives, token),
                supporting_count=count,
                estimated_precision_proxy=1.0, # Derived from known phishing
                languages=_languages(false_negatives),
                source_support=_source_support_from_rows(false_negatives),
                recommended_action="Consider as a new phishing detection heuristic.",
                derived_from_errors=True,
            )
        )

    # 2. benign_suppression_signal from false positives (wrongly flagged benign)
    fp_tokens = _extract_recurring_tokens(false_positives)
    for token, count in fp_tokens.most_common(top_k):
        proposals.append(
            _proposal(
                rule_type="benign_suppression_signal",
                candidate_pattern=token,
                rationale=f"Recurring token '{token}' appears in benign pages wrongly flagged as phishing.",
                supporting_examples=_get_examples_for_token(false_positives, token),
                supporting_count=count,
                estimated_precision_proxy=0.0, # Derived from known benign
                languages=_languages(false_positives),
                source_support=_source_support_from_rows(false_positives),
                recommended_action="Consider as a benign-suppression guardrail.",
                derived_from_errors=True,
            )
        )

    # 3. escalation_signal: general error patterns or domain families
    domain_counter = Counter(row.get("registered_domain") for row in false_negatives if row.get("registered_domain"))
    for domain, count in domain_counter.most_common(5):
        proposals.append(
            _proposal(
                rule_type="escalation_signal",
                candidate_pattern=domain,
                rationale=f"Domain family '{domain}' consistently missed by the model.",
                supporting_examples=[_example(row) for row in false_negatives if row.get("registered_domain") == domain][:3],
                supporting_count=count,
                estimated_precision_proxy=0.5,
                languages=_languages(false_negatives),
                source_support=_source_support_from_rows(false_negatives),
                recommended_action="Escalate for manual domain-family profiling.",
                derived_from_errors=True,
            )
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error_analysis_dir": str(error_analysis_dir),
        "proposal_count": len(proposals),
        "proposals": proposals[:top_k],
    }
    write_json(output_path, payload)
    return payload


def _extract_recurring_tokens(rows: list[dict[str, Any]]) -> Counter:
    counter = Counter()
    for row in rows:
        text = row.get("text_snippet") or ""
        tokens = TOKEN_RE.findall(text.lower())
        counter.update(t for t in tokens if len(t) >= 5)
    return counter


def _get_examples_for_token(rows: list[dict[str, Any]], token: str, limit: int = 3) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        text = row.get("text_snippet") or ""
        if token.lower() in text.lower():
            examples.append(_example(row))
            if len(examples) >= limit:
                break
    return examples


def _phrase_proposals(phishing_rows: list[dict[str, Any]], benign_rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    phishing_counter = Counter()
    benign_counter = Counter()
    examples = defaultdict(list)
    source_support = defaultdict(Counter)
    for row in phishing_rows:
        tokens = _ngrams(row.get("normalized_text") or row.get("extracted_text") or "")
        phishing_counter.update(tokens)
        for token in set(tokens):
            source_support[token][row.get("source_dataset") or "unknown"] += 1
            if len(examples[token]) < 3:
                examples[token].append(_example(row))
    for row in benign_rows:
        benign_counter.update(_ngrams(row.get("normalized_text") or row.get("extracted_text") or ""))
    proposals = []
    for phrase, count in phishing_counter.most_common(top_k * 3):
        if count < 3:
            continue
        benign_count = benign_counter.get(phrase, 0)
        precision_proxy = count / max(1, count + benign_count)
        if precision_proxy < 0.7:
            continue
        proposals.append(
            _proposal(
                rule_type="text_phrase",
                candidate_pattern=phrase,
                rationale=f"Recurring phrase appears {count} times in phishing-oriented samples with limited benign overlap.",
                supporting_examples=examples[phrase],
                supporting_count=count,
                estimated_precision_proxy=round(precision_proxy, 4),
                languages=_languages_for_examples(examples[phrase]),
                source_support=dict(source_support[phrase]),
                recommended_action="Review as a candidate phishing CTA or urgency heuristic.",
            )
        )
    return proposals


def _cta_proposals(phishing_rows: list[dict[str, Any]], benign_rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    phishing_counter = Counter()
    benign_counter = Counter()
    examples = defaultdict(list)
    source_support = defaultdict(Counter)
    for row in phishing_rows:
        for action_text in (row.get("html_features") or {}).get("action_texts", []):
            normalized = action_text.strip().lower()
            if not normalized:
                continue
            phishing_counter[normalized] += 1
            source_support[normalized][row.get("source_dataset") or "unknown"] += 1
            if len(examples[normalized]) < 3:
                examples[normalized].append(_example(row))
    for row in benign_rows:
        for action_text in (row.get("html_features") or {}).get("action_texts", []):
            normalized = action_text.strip().lower()
            if normalized:
                benign_counter[normalized] += 1
    proposals = []
    for phrase, count in phishing_counter.most_common(top_k * 2):
        if count < 2:
            continue
        precision_proxy = count / max(1, count + benign_counter.get(phrase, 0))
        if precision_proxy < 0.65:
            continue
        proposals.append(
            _proposal(
                rule_type="cta_pattern",
                candidate_pattern=phrase,
                rationale="Repeated call-to-action text appears disproportionately in phishing-oriented pages.",
                supporting_examples=examples[phrase],
                supporting_count=count,
                estimated_precision_proxy=round(precision_proxy, 4),
                languages=_languages_for_examples(examples[phrase]),
                source_support=dict(source_support[phrase]),
                recommended_action="Review as a CTA heuristic candidate.",
            )
        )
    return proposals


def _url_style_proposals(phishing_rows: list[dict[str, Any]], benign_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_names = [
        "is_ip_address",
        "has_punycode",
        "has_percent_encoding",
        "contains_at_symbol",
        "contains_double_slash_path",
    ]
    proposals = []
    for name in feature_names:
        phishing_count = sum(1 for row in phishing_rows if (row.get("url_features") or {}).get(name))
        benign_count = sum(1 for row in benign_rows if (row.get("url_features") or {}).get(name))
        if phishing_count < 2:
            continue
        precision_proxy = phishing_count / max(1, phishing_count + benign_count)
        proposals.append(
            _proposal(
                rule_type="url_obfuscation",
                candidate_pattern=name,
                rationale="Repeated URL obfuscation style is present in phishing-oriented samples.",
                supporting_examples=[_example(row) for row in phishing_rows if (row.get("url_features") or {}).get(name)][:3],
                supporting_count=phishing_count,
                estimated_precision_proxy=round(precision_proxy, 4),
                languages=_languages(phishing_rows),
                source_support=_source_support_from_rows(
                    [row for row in phishing_rows if (row.get("url_features") or {}).get(name)]
                ),
                recommended_action="Review as a URL-structure heuristic candidate.",
            )
        )
    return proposals


def _html_pattern_proposals(phishing_rows: list[dict[str, Any]], benign_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {
        "meta_refresh_present": "Meta refresh redirection",
        "has_password_input": "Password field present",
    }
    proposals = []
    for feature, description in definitions.items():
        phishing_count = sum(1 for row in phishing_rows if (row.get("html_features") or {}).get(feature))
        benign_count = sum(1 for row in benign_rows if (row.get("html_features") or {}).get(feature))
        if phishing_count < 2:
            continue
        precision_proxy = phishing_count / max(1, phishing_count + benign_count)
        proposals.append(
            _proposal(
                rule_type="html_pattern",
                candidate_pattern=feature,
                rationale=f"{description} recurs across phishing-oriented pages.",
                supporting_examples=[_example(row) for row in phishing_rows if (row.get("html_features") or {}).get(feature)][:3],
                supporting_count=phishing_count,
                estimated_precision_proxy=round(precision_proxy, 4),
                languages=_languages(phishing_rows),
                source_support=_source_support_from_rows(
                    [row for row in phishing_rows if (row.get("html_features") or {}).get(feature)]
                ),
                recommended_action="Review as an HTML/DOM heuristic candidate.",
            )
        )
    return proposals


def _proposal(
    *,
    rule_type: str,
    candidate_pattern: str,
    rationale: str,
    supporting_examples: list[dict[str, Any]],
    supporting_count: int,
    estimated_precision_proxy: float,
    languages: list[str],
    source_support: dict[str, int],
    recommended_action: str,
    derived_from_errors: bool = False,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proposal_id = make_sample_id(source="heuristic_proposal", original_id=f"{rule_type}:{candidate_pattern}:{created_at}")
    return {
        "proposal_id": proposal_id,
        "rule_type": rule_type,
        "candidate_pattern": candidate_pattern,
        "rationale": rationale,
        "supporting_examples": supporting_examples,
        "supporting_count": supporting_count,
        "estimated_precision_proxy": estimated_precision_proxy,
        "languages": sorted(set(language for language in languages if language)),
        "source_support": source_support,
        "status": "proposed",
        "recommended_action": recommended_action,
        "created_at": created_at,
        "derived_from_errors": derived_from_errors,
    }


def _is_positive(row: dict[str, Any], feedback_by_sample: dict[str, dict[str, Any]]) -> bool:
    feedback = feedback_by_sample.get(row.get("sample_id"))
    if feedback and feedback.get("final_reviewed_label") in (0, 1):
        return feedback["final_reviewed_label"] == 1
    return row.get("normalized_label") == 1


def _is_negative(row: dict[str, Any], feedback_by_sample: dict[str, dict[str, Any]]) -> bool:
    feedback = feedback_by_sample.get(row.get("sample_id"))
    if feedback and feedback.get("final_reviewed_label") in (0, 1):
        return feedback["final_reviewed_label"] == 0
    return row.get("normalized_label") == 0


def _ngrams(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())[:80]
    grams = []
    for n in (1, 2, 3):
        for index in range(0, max(0, len(tokens) - n + 1)):
            grams.append(" ".join(tokens[index : index + n]))
    return grams


def _example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "source_dataset": row.get("source_dataset"),
        "language": row.get("language"),
        "url": row.get("normalized_url") or row.get("original_url"),
        "text_snippet": (row.get("normalized_text") or row.get("extracted_text") or "")[:240],
    }


def _languages(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({row.get("language") for row in rows if row.get("language")})


def _languages_for_examples(examples: list[dict[str, Any]]) -> list[str]:
    return sorted({example.get("language") for example in examples if example.get("language")})


def _source_support_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(row.get("source_dataset") or "unknown" for row in rows)
    return dict(counter)
