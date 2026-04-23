from __future__ import annotations

from pathlib import Path

from veriscope_training.acquisition.base import AcquisitionError, CredentialRequired, DatasetFetcher, FetchResult
from veriscope_training.utils.io import write_jsonl


class OscarAuxFetcher(DatasetFetcher):
    dataset_name = "oscar_aux"

    def fetch(self, *, force: bool = False) -> FetchResult:
        mode = self.fetch_mode()
        if mode == "disabled":
            raise AcquisitionError("Fetching is disabled for oscar_aux in configs/datasets.yaml.")
        if mode == "streaming":
            return self._streaming_preview()
        if mode != "auto":
            raise AcquisitionError(f"Unsupported fetch mode '{mode}' for oscar_aux.")

        remote = self.fetch_config.get("remote_source") or {}
        dataset_id = remote.get("dataset_id", "oscar-corpus/oscar")
        config_name = remote.get("config_name")
        split = remote.get("split", "train")
        output_file = self.output_dir / "oscar_aux.jsonl"
        if output_file.exists() and not force:
            result = FetchResult(
                dataset_name=self.dataset_name,
                fetch_mode_used="auto",
                output_dir=str(self.output_dir),
                source_reference=f"{dataset_id}:{config_name}:{split}",
                files_written=[str(output_file)],
                validation_status="ready_existing",
            )
            self.write_fetch_metadata(result)
            return result
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise AcquisitionError(
                "Fetching oscar_aux requires the datasets package. Install project requirements on the server."
            ) from exc
        token = self.env(remote.get("hf_token_env"))
        if token is None:
            raise CredentialRequired(
                "OSCAR access may require a Hugging Face token and accepted gated-access terms. "
                "Set HF_TOKEN on the server or place an approved local snapshot directly under "
                f"{self.output_dir}."
            )
        try:
            dataset = load_dataset(dataset_id, name=config_name, split=split, token=token, streaming=False)
        except Exception as exc:
            raise CredentialRequired(
                "Failed to fetch OSCAR from Hugging Face. Ensure the server-side HF_TOKEN is set and that "
                "the account has accepted any gated-access terms, or place a snapshot manually under "
                f"{self.output_dir}."
            ) from exc

        text_field = remote.get("text_field", "text")
        id_field = remote.get("id_field", "id")
        max_records = int(remote.get("max_records", 50000))
        rows = []
        for index, row in enumerate(dataset):
            rows.append({
                id_field: row.get(id_field),
                text_field: row.get(text_field),
            })
            if index + 1 >= max_records:
                break
        write_jsonl(output_file, rows)
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="auto",
            output_dir=str(self.output_dir),
            source_reference=f"{dataset_id}:{config_name}:{split}",
            files_written=[str(output_file)],
            validation_status="ready",
            warnings=["OSCAR remains unlabeled auxiliary text only."],
            metadata={"records_written": len(rows)},
        )
        self.write_fetch_metadata(result)
        return result

    def _streaming_preview(self) -> FetchResult:
        remote = self.fetch_config.get("remote_source") or {}
        dataset_id = remote.get("dataset_id", "oscar-corpus/oscar")
        config_name = remote.get("config_name")
        split = remote.get("split", "train")
        token = self.env(remote.get("hf_token_env"))
        if token is None:
            raise CredentialRequired(
                "Streaming OSCAR preview requires HF_TOKEN on the server and any required gated-access approval."
            )
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise AcquisitionError("Streaming OSCAR preview requires the datasets package.") from exc
        rows = load_dataset(dataset_id, name=config_name, split=split, token=token, streaming=True)
        limit = int((self.fetch_config.get("streaming_preview") or {}).get("sample_limit", 100))
        preview = []
        for index, row in enumerate(rows):
            preview.append(row)
            if index + 1 >= limit:
                break
        preview_path = self.output_dir / "streaming_preview.jsonl"
        write_jsonl(preview_path, preview)
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="streaming",
            output_dir=str(self.output_dir),
            source_reference=f"{dataset_id}:{config_name}:{split}",
            files_written=[str(preview_path)],
            validation_status="preview_only",
            warnings=["Streaming mode writes a preview artifact, not a full local training snapshot."],
        )
        self.write_fetch_metadata(result)
        return result
