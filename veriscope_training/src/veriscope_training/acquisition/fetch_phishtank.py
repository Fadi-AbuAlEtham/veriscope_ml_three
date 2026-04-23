from __future__ import annotations

from pathlib import Path

from veriscope_training.acquisition.base import AcquisitionError, DatasetFetcher, FetchResult


class PhishTankFetcher(DatasetFetcher):
    dataset_name = "phishtank"

    def fetch(self, *, force: bool = False) -> FetchResult:
        mode = self.fetch_mode()
        if mode == "disabled":
            raise AcquisitionError("Fetching is disabled for phishtank in configs/datasets.yaml.")
        if mode == "manual_snapshot":
            return self._prepare_manual_snapshot()
        if mode != "auto":
            raise AcquisitionError(f"Unsupported fetch mode '{mode}' for phishtank.")

        remote = self.fetch_config.get("remote_source") or {}
        app_key = self.env(remote.get("app_key_env"))
        user_agent = self.env(remote.get("user_agent_env")) or "veriscope-training/server-fetch"
        if app_key and remote.get("app_key_url_template"):
            url = str(remote["app_key_url_template"]).format(app_key=app_key)
        else:
            url = remote.get("public_url")
        if not url:
            raise AcquisitionError("PhishTank fetch URL is not configured.")
        target = self.output_dir / (Path(url).name or "online-valid.json.bz2")
        file_path = self.download_to_path(
            url=url,
            destination=target,
            headers={"User-Agent": user_agent},
            force=force,
        )
        warnings = []
        if not app_key:
            warnings.append("PHISHTANK_APP_KEY was not set; public downloads may be rate limited.")
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="auto",
            output_dir=str(self.output_dir),
            source_reference=url,
            files_written=[file_path],
            validation_status="ready",
            warnings=warnings,
        )
        self.write_fetch_metadata(result)
        return result

    def _prepare_manual_snapshot(self) -> FetchResult:
        instructions = (
            "# PhishTank Manual Server-Side Snapshot Placement\n\n"
            f"Place official PhishTank bulk snapshot files directly into `{self.output_dir}` on the server.\n\n"
            "Supported formats:\n"
            "- csv, xml, json\n"
            "- csv.gz, xml.gz, json.gz\n"
            "- csv.bz2, xml.bz2, json.bz2\n\n"
            "Examples:\n"
            f"- `{self.output_dir}/online-valid.csv`\n"
            f"- `{self.output_dir}/online-valid.json.bz2`\n"
        )
        instructions_path = self.write_manual_instructions(instructions)
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="manual_snapshot",
            output_dir=str(self.output_dir),
            source_reference="manual_server_snapshot",
            files_written=[instructions_path],
            validation_status="manual_action_required",
        )
        self.write_fetch_metadata(result)
        return result
