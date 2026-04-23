from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
from xml.etree.ElementTree import Element

from veriscope_training.datasets.base import DatasetAdapter, DatasetRecord
from veriscope_training.datasets.loaders import (
    clean_text,
    exclude_fields,
    iter_json_rows,
    iter_structured_rows,
    iter_xml_elements,
    normalize_binary_label,
    open_text,
    pick_first,
    primary_suffix,
)
from veriscope_training.datasets.registry import register_dataset


PHISHTANK_CSV_FIELDS = (
    "phish_id",
    "url",
    "phish_detail_url",
    "submission_time",
    "verified",
    "verification_time",
    "online",
    "target",
)


@register_dataset("phishtank")
class PhishTankAdapter(DatasetAdapter):
    supported_formats = ("csv", "json", "xml")
    source_type = "local_snapshot"

    def validation_report(self) -> dict[str, Any]:
        payload = super().validation_report()
        supported_paths = self._supported_paths()
        payload["supported_file_count"] = len(supported_paths)
        if not supported_paths:
            payload["issues"].append(
                "Expected one or more PhishTank CSV/XML/JSON snapshots, including compressed variants."
            )
        return payload

    def iterate_records(self) -> Iterator[DatasetRecord]:
        supported_paths = self._supported_paths()
        if not supported_paths:
            raise FileNotFoundError(
                f"PhishTank adapter could not find supported snapshots under {self.snapshot_root}."
            )
        for path in supported_paths:
            suffix = primary_suffix(path)
            if suffix == ".xml":
                yield from self._iterate_xml(path)
            elif suffix == ".json":
                yield from self._iterate_json(path)
            else:
                yield from self._iterate_csv_like(path)

    def _supported_paths(self) -> list[Path]:
        return [path for path in self.snapshot_paths() if primary_suffix(path) in {".csv", ".xml", ".json"}]

    def _iterate_csv_like(self, path: Path) -> Iterator[DatasetRecord]:
        for row_index, row in enumerate(iter_structured_rows(path), start=1):
            record = self._record_from_mapping(row, path=path, row_index=row_index)
            if record is not None:
                yield record

    def _iterate_json(self, path: Path) -> Iterator[DatasetRecord]:
        for row_index, row in enumerate(iter_json_rows(path), start=1):
            record = self._record_from_mapping(row, path=path, row_index=row_index)
            if record is not None:
                yield record

    def _iterate_xml(self, path: Path) -> Iterator[DatasetRecord]:
        for row_index, entry in enumerate(iter_xml_elements(path, "entry"), start=1):
            row = self._mapping_from_xml_entry(entry)
            record = self._record_from_mapping(row, path=path, row_index=row_index)
            if record is not None:
                yield record

    def _mapping_from_xml_entry(self, entry: Element) -> dict[str, Any]:
        details = []
        for detail in entry.findall("./details/detail"):
            details.append(
                {
                    "ip_address": clean_text(detail.findtext("ip_address")),
                    "cidr_block": clean_text(detail.findtext("cidr_block")),
                    "announcing_network": clean_text(detail.findtext("announcing_network")),
                    "rir": clean_text(detail.findtext("rir")),
                    "detail_time": clean_text(detail.findtext("detail_time")),
                }
            )
        return {
            "url": clean_text(entry.findtext("url")),
            "phish_id": clean_text(entry.findtext("phish_id")),
            "phish_detail_url": clean_text(entry.findtext("phish_detail_url")),
            "submission_time": clean_text(entry.findtext("./submission/submission_time")),
            "verified": clean_text(entry.findtext("./verification/verified")),
            "verification_time": clean_text(entry.findtext("./verification/verification_time")),
            "online": clean_text(entry.findtext("./status/online")),
            "target": clean_text(entry.findtext("target")),
            "details": details,
        }

    def _record_from_mapping(
        self,
        row: dict[str, Any],
        *,
        path: Path,
        row_index: int,
    ) -> DatasetRecord | None:
        url = clean_text(row.get("url"))
        if url is None:
            return None
        verified = clean_text(row.get("verified"))
        normalized_label = self._normalize_label(verified=verified, path=path)
        original_id = clean_text(row.get("phish_id")) or f"{path.name}:{row_index}"
        timestamp = clean_text(row.get("verification_time")) or clean_text(row.get("submission_time"))
        metadata = exclude_fields(row, excluded=("url", "phish_id"))
        metadata["source_path"] = str(path)
        metadata["source_format"] = primary_suffix(path).lstrip(".")
        metadata["phishing_only_source"] = True
        metadata["label_mapping_status"] = "mapped" if normalized_label == 1 else "unmapped"
        return DatasetRecord(
            source=self.name,
            original_id=original_id,
            original_label=verified or "phishing_confirmed",
            normalized_label=normalized_label,
            url=url,
            timestamp=timestamp,
            metadata=metadata,
        )

    def _normalize_label(self, *, verified: str | None, path: Path) -> int | None:
        if verified is not None:
            return normalize_binary_label(
                verified,
                phishing_values={"yes", "true", "valid", "verified", "1"},
                benign_values={"no", "false", "0"},
                uncertain_values={"unknown", "unverified", "suspicious"},
            )
        if "online-valid" in path.name.lower():
            return 1
        return None
