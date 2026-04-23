from __future__ import annotations

from veriscope_training.acquisition.base import DatasetFetcher, FetchResult


class MendeleyFetcher(DatasetFetcher):
    dataset_name = "mendeley"

    def fetch(self, *, force: bool = False) -> FetchResult:
        instructions = (
            "# Mendeley Server-Side Snapshot Preparation\n\n"
            "Direct scripted downloading is not enabled because Mendeley mirrors and access patterns vary.\n\n"
            f"Place the dataset files directly on the server under `{self.output_dir}`.\n\n"
            "Supported layouts:\n"
            "- structured files such as csv/tsv/jsonl/json/parquet\n"
            "- html/htm/txt files grouped under phishing/ and benign/\n\n"
            "Recommended fields for structured files:\n"
            "- url\n- html\n- text/content\n- label\n- optional split, language, timestamp\n"
        )
        instructions_path = self.write_manual_instructions(instructions)
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="manual_snapshot",
            output_dir=str(self.output_dir),
            source_reference="manual_server_snapshot",
            files_written=[instructions_path],
            validation_status="manual_action_required",
            warnings=["Manual server-side placement is required for Mendeley snapshots."],
        )
        self.write_fetch_metadata(result)
        return result
