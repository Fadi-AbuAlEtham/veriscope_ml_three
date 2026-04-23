from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from veriscope_training.acquisition.base import AcquisitionError, DatasetFetcher, FetchResult
from veriscope_training.utils.io import write_json


class UCIPhishingFetcher(DatasetFetcher):
    dataset_name = "uci_phishing"

    def fetch(self, *, force: bool = False) -> FetchResult:
        mode = self.fetch_mode()
        if mode != "auto":
            raise AcquisitionError(f"Unsupported fetch mode '{mode}' for uci_phishing.")
        remote = self.fetch_config.get("remote_source") or {}
        dataset_id = int(remote.get("dataset_id", 327))
        csv_path = self.output_dir / str(remote.get("expected_primary_file", "uci_phishing.csv"))
        if csv_path.exists() and not force:
            result = FetchResult(
                dataset_name=self.dataset_name,
                fetch_mode_used="auto",
                output_dir=str(self.output_dir),
                source_reference=f"ucimlrepo:{dataset_id}",
                files_written=[str(csv_path)],
                validation_status="ready_existing",
            )
            self.write_fetch_metadata(result)
            return result
        files_written: list[str]
        source_reference: str
        metadata_note: dict[str, str] = {}
        try:
            from ucimlrepo import fetch_ucirepo

            dataset = fetch_ucirepo(id=dataset_id)
            features = dataset.data.features.copy()
            targets = dataset.data.targets.copy()
            if len(targets.columns) == 1:
                features[targets.columns[0]] = targets.iloc[:, 0]
            else:
                for column in targets.columns:
                    features[column] = targets[column]
            features.to_csv(csv_path, index=False)
            metadata_path = write_json(self.output_dir / "uci_metadata.json", dataset.metadata)
            variables_path = self.output_dir / "uci_variables.csv"
            dataset.variables.to_csv(variables_path, index=False)
            files_written = [str(csv_path), str(metadata_path), str(variables_path)]
            source_reference = f"ucimlrepo:{dataset_id}"
        except Exception:
            files_written = self._fetch_official_zip_fallback(force=force)
            source_reference = "https://archive.ics.uci.edu/static/public/327/phishing%2Bwebsites.zip"
            metadata_note = {"fallback": "official_uci_zip_download"}
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="auto",
            output_dir=str(self.output_dir),
            source_reference=source_reference,
            files_written=files_written,
            validation_status="ready",
            metadata=metadata_note,
        )
        self.write_fetch_metadata(result)
        return result

    def _fetch_official_zip_fallback(self, *, force: bool) -> list[str]:
        zip_url = "https://archive.ics.uci.edu/static/public/327/phishing%2Bwebsites.zip"
        zip_path = self.output_dir / "phishing_websites.zip"
        if force or not zip_path.exists():
            response = requests.get(zip_url, timeout=(20, 300))
            response.raise_for_status()
            zip_path.write_bytes(response.content)
        written = [str(zip_path)]
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(self.output_dir)
            for member in archive.namelist():
                written.append(str(self.output_dir / member))
        return written
