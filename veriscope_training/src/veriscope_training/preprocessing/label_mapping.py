from __future__ import annotations

from dataclasses import asdict, dataclass

from veriscope_training.datasets.loaders import normalize_binary_label


@dataclass
class LabelNormalizationResult:
    source_label: str | int | None
    normalized_label: int | None
    label_name: str
    mapping_reason: str
    is_auxiliary: bool
    is_supervised: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_record_label(
    *,
    source_dataset: str,
    source_label: str | int | None,
    adapter_normalized_label: int | None,
    metadata: dict[str, object] | None = None,
) -> LabelNormalizationResult:
    metadata = metadata or {}
    if source_dataset == "oscar_aux":
        return LabelNormalizationResult(
            source_label=source_label,
            normalized_label=None,
            label_name="auxiliary_unlabeled",
            mapping_reason="auxiliary_text_corpus",
            is_auxiliary=True,
            is_supervised=False,
        )
    if source_dataset == "openphish":
        return LabelNormalizationResult(
            source_label=source_label or "phishing",
            normalized_label=1,
            label_name="phishing",
            mapping_reason="phishing_only_feed",
            is_auxiliary=False,
            is_supervised=True,
        )
    if source_dataset == "phishtank":
        if adapter_normalized_label == 1:
            return LabelNormalizationResult(
                source_label=source_label,
                normalized_label=1,
                label_name="phishing",
                mapping_reason="confirmed_phishtank_entry",
                is_auxiliary=False,
                is_supervised=True,
            )
        return LabelNormalizationResult(
            source_label=source_label,
            normalized_label=None,
            label_name="unknown",
            mapping_reason="unconfirmed_or_missing_verification",
            is_auxiliary=False,
            is_supervised=False,
        )
    if adapter_normalized_label in (0, 1):
        label_name = "phishing" if adapter_normalized_label == 1 else "benign"
        return LabelNormalizationResult(
            source_label=source_label,
            normalized_label=adapter_normalized_label,
            label_name=label_name,
            mapping_reason="adapter_supplied_binary_label",
            is_auxiliary=False,
            is_supervised=True,
        )

    fallback = normalize_binary_label(
        source_label,
        phishing_values={"phish", "phishing", "malicious", "fraudulent", "-1"},
        benign_values={"benign", "legitimate", "legit", "safe", "1"},
        uncertain_values={"suspicious", "unknown", "uncertain", "0", "unlabeled"},
    )
    if fallback in (0, 1):
        label_name = "phishing" if fallback == 1 else "benign"
        return LabelNormalizationResult(
            source_label=source_label,
            normalized_label=fallback,
            label_name=label_name,
            mapping_reason="central_fallback_mapping",
            is_auxiliary=False,
            is_supervised=True,
        )
    return LabelNormalizationResult(
        source_label=source_label,
        normalized_label=None,
        label_name="unknown",
        mapping_reason="unmapped_or_unlabeled",
        is_auxiliary=False,
        is_supervised=False,
    )
