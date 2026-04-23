from __future__ import annotations

from pathlib import Path

from veriscope_training.acquisition.base import AcquisitionError, DatasetFetcher, FetchResult
from veriscope_training.utils.io import write_jsonl


class PhreshPhishFetcher(DatasetFetcher):
    dataset_name = "phreshphish"

    def fetch(self, *, force: bool = False) -> FetchResult:
        mode = self.fetch_mode()
        if mode == "disabled":
            raise AcquisitionError("Fetching is disabled for phreshphish in configs/datasets.yaml.")
        if mode == "streaming":
            return self._streaming_preview()
        if mode != "auto":
            raise AcquisitionError(f"Unsupported fetch mode '{mode}' for phreshphish.")

        remote = self.fetch_config.get("remote_source") or {}
        dataset_id = remote.get("dataset_id", "phreshphish/phreshphish")
        local_dir = self.output_dir / "hf_snapshot"
        if local_dir.exists() and any(local_dir.rglob("*.parquet")) and not force:
            result = FetchResult(
                dataset_name=self.dataset_name,
                fetch_mode_used=mode,
                output_dir=str(self.output_dir),
                source_reference=dataset_id,
                files_written=[str(path) for path in sorted(local_dir.rglob("*")) if path.is_file()],
                validation_status="ready_existing",
            )
            self.write_fetch_metadata(result)
            return result
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise AcquisitionError(
                "Fetching phreshphish from Hugging Face requires huggingface_hub. "
                "Install project requirements on the remote server."
            ) from exc

        token = self.env(remote.get("hf_token_env"))
        snapshot_download(
            repo_id=dataset_id,
            repo_type=str(remote.get("repo_type", "dataset")),
            local_dir=str(local_dir),
            allow_patterns=remote.get("allow_patterns"),
            token=token,
            resume_download=True,
        )
        files = [str(path) for path in sorted(local_dir.rglob("*")) if path.is_file()]
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used=mode,
            output_dir=str(self.output_dir),
            source_reference=dataset_id,
            files_written=files,
            validation_status="ready",
        )
        self.write_fetch_metadata(result)
        return result

    def _streaming_preview(self) -> FetchResult:
        hf_cfg = self.source_config.extra.get("huggingface") or {}
        dataset_id = hf_cfg.get("dataset_id", "phreshphish/phreshphish")
        split = (hf_cfg.get("splits") or ["train"])[0]
        preview_cfg = self.fetch_config.get("streaming_preview") or {}
        limit = int(preview_cfg.get("sample_limit", 100))
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise AcquisitionError(
                "Streaming preview for phreshphish requires the datasets package."
            ) from exc
        rows = load_dataset(dataset_id, split=split, streaming=True)
        preview_rows = []
        for index, row in enumerate(rows):
            preview_rows.append(row)
            if index + 1 >= limit:
                break
        preview_path = self.output_dir / "streaming_preview.jsonl"
        write_jsonl(preview_path, preview_rows)
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="streaming",
            output_dir=str(self.output_dir),
            source_reference=f"{dataset_id}:{split}",
            files_written=[str(preview_path)],
            validation_status="preview_only",
            warnings=["Streaming mode writes a preview artifact, not a full local training snapshot."],
        )
        self.write_fetch_metadata(result)
        return result
