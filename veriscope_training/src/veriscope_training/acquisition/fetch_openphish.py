from __future__ import annotations

from pathlib import Path

from veriscope_training.acquisition.base import AcquisitionError, CredentialRequired, DatasetFetcher, FetchResult


class OpenPhishFetcher(DatasetFetcher):
    dataset_name = "openphish"

    def fetch(self, *, force: bool = False) -> FetchResult:
        mode = self.fetch_mode()
        if mode == "disabled":
            raise AcquisitionError("Fetching is disabled for openphish in configs/datasets.yaml.")
        if mode == "manual_snapshot":
            return self._prepare_manual_snapshot()
        if mode != "auto":
            raise AcquisitionError(f"Unsupported fetch mode '{mode}' for openphish.")

        remote = self.fetch_config.get("remote_source") or {}
        variant = str(remote.get("variant", "community"))
        if variant == "community":
            url = remote.get("community_url")
            if not url:
                raise AcquisitionError("OpenPhish community feed URL is not configured.")
            target = self.output_dir / "feed.txt"
            file_path = self.download_to_path(url=url, destination=target, force=force)
            result = FetchResult(
                dataset_name=self.dataset_name,
                fetch_mode_used="auto",
                output_dir=str(self.output_dir),
                source_reference=url,
                files_written=[file_path],
                validation_status="ready",
                warnings=["Downloaded the public community feed directly on the server."],
            )
            self.write_fetch_metadata(result)
            return result

        fetch_url = self.env(remote.get("academic_or_premium_url_env"))
        token = self.env(remote.get("token_env"))
        username = self.env(remote.get("username_env"))
        password = self.env(remote.get("password_env"))
        if not fetch_url:
            raise CredentialRequired(
                "OpenPhish academic/premium fetching requires a server-side fetch URL in OPENPHISH_FETCH_URL. "
                f"If you do not have approved access, place your snapshot directly under {self.output_dir}."
            )
        auth = (username, password) if username and password else None
        target_name = Path(fetch_url).name or "openphish_feed.txt"
        file_path = self.download_to_path(
            url=fetch_url,
            destination=self.output_dir / target_name,
            auth=auth,
            token=token,
            force=force,
        )
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="auto",
            output_dir=str(self.output_dir),
            source_reference=fetch_url,
            files_written=[file_path],
            validation_status="ready",
        )
        self.write_fetch_metadata(result)
        return result

    def _prepare_manual_snapshot(self) -> FetchResult:
        instructions = (
            "# OpenPhish Manual Server-Side Snapshot Placement\n\n"
            f"Place one or more approved OpenPhish snapshot files directly into `{self.output_dir}` on the server.\n\n"
            "Supported formats:\n"
            "- txt\n- csv\n- tsv\n- json\n- jsonl\n- parquet\n\n"
            "Examples:\n"
            f"- `{self.output_dir}/feed.txt`\n"
            f"- `{self.output_dir}/openphish_archive.csv`\n\n"
            "If you have approved academic or premium access, you may also set `OPENPHISH_FETCH_URL` and optional "
            "`OPENPHISH_USERNAME`/`OPENPHISH_PASSWORD` or `OPENPHISH_TOKEN` and switch fetch mode to `auto`.\n"
        )
        instructions_path = self.write_manual_instructions(instructions)
        result = FetchResult(
            dataset_name=self.dataset_name,
            fetch_mode_used="manual_snapshot",
            output_dir=str(self.output_dir),
            source_reference="manual_server_snapshot",
            files_written=[instructions_path],
            validation_status="manual_action_required",
            warnings=["Server-side manual snapshot placement is required for this mode."],
        )
        self.write_fetch_metadata(result)
        return result
