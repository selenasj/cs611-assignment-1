import json
from datetime import datetime, timezone
from pathlib import Path

from utils.contracts import contracts_as_dict


class PipelineManifest:
    """Collect reproducible run evidence and save it beside the generated datamart."""

    def __init__(self) -> None:
        self.payload = {
            "pipeline": "cs611-assignment-1",
            "started_at_utc": self._now(),
            "finished_at_utc": None,
            "status": "running",
            "contracts": contracts_as_dict(),
            "stages": {},
            "error": None,
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_stage(self, stage_name: str, result: dict) -> None:
        self.payload["stages"][stage_name] = result

    def complete(self) -> None:
        self.payload["status"] = "passed"
        self.payload["finished_at_utc"] = self._now()

    def fail(self, error: Exception) -> None:
        self.payload["status"] = "failed"
        self.payload["error"] = f"{type(error).__name__}: {error}"
        self.payload["finished_at_utc"] = self._now()

    def write(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path
