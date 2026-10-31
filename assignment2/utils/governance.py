"""Explicit model approval, deployment history and rollback for two scopes.

Historical approvals are policy simulations. They never authorize scoring in
the production scope, where approval and promotion require separate commands.
This local registry records decisions; it does not authenticate the reviewer.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from utils.config import PROJECT_ROOT

MODEL_BANK = PROJECT_ROOT / "model_bank"
REGISTRY_DIR = MODEL_BANK / "registry"
SCOPES = ("historical", "production")


def _date(value) -> str:
    day = pd.Timestamp(value)
    if pd.isna(day):
        raise ValueError("An effective date is required")
    return day.strftime("%Y-%m-%d")


def _identity(reviewer: str, reason: str) -> None:
    if not reviewer or not reviewer.strip() or not reason or not reason.strip():
        raise ValueError("A nonempty reviewer and reason are required")


def _artifact(version: str) -> Path:
    if Path(version).name != version or not version.startswith("credit_model_"):
        raise ValueError("Expected a model version, not a file path")
    path = MODEL_BANK / f"{version}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact missing: {version}")
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _registry(scope: str):
    if scope not in SCOPES:
        raise ValueError(f"Unknown governance scope: {scope}")
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = REGISTRY_DIR / f"{scope}.json"
    with (REGISTRY_DIR / f"{scope}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {
            "scope": scope, "models": {}, "events": [],
        }
        original = json.dumps(state, sort_keys=True)
        yield state
        if json.dumps(state, sort_keys=True) != original:
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(path)


def _event(state: dict, action: str, version: str, effective: str,
           reviewer: str, reason: str, **details) -> dict:
    event = {
        "sequence": len(state["events"]) + 1,
        "action": action, "model_version": version,
        "effective_date": effective,
        "reviewer": reviewer, "reason": reason,
        "decision_mode": "policy_simulation" if state["scope"] == "historical" else "manual_record",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    state["events"].append(event)
    return event


def _registered(state: dict, version: str) -> dict:
    record = state["models"].get(version)
    if record is None:
        raise RuntimeError(f"Model {version} is not registered in {state['scope']}")
    if record["artifact_sha256"] != _hash(_artifact(version)):
        raise RuntimeError("Artifact changed after registration; use a new version")
    return record


def register_candidate(version: str, scope: str = "historical") -> dict:
    path = _artifact(version)
    metadata = joblib.load(path)["metadata"]
    digest = _hash(path)
    with _registry(scope) as state:
        existing = state["models"].get(version)
        if existing:
            _registered(state, version)
            return dict(existing)
        record = {
            "model_version": version, "status": "candidate",
            "trained_as_of_date": metadata["as_of_date"],
            "artifact_sha256": digest,
            "approval_effective_date": None,
        }
        state["models"][version] = record
        _event(state, "register_candidate", version, metadata["as_of_date"],
               "training_pipeline", "Trained artifact registered; not yet approved")
        return dict(record)


def assess_candidate(version: str, as_of: str) -> dict:
    # Lazy import avoids a cycle between training and governance.
    from utils.model_pipeline import assess_for_promotion
    return assess_for_promotion(version, as_of)


def approve_model(version: str, as_of: str, reviewer: str, reason: str,
                  scope: str = "production") -> dict:
    _identity(reviewer, reason)
    effective = _date(as_of)
    register_candidate(version, scope)
    assessment = assess_candidate(version, effective)
    if not assessment["promotion_eligible"]:
        raise RuntimeError("Approval blocked by validation/data gates: " +
                           ", ".join(assessment["blocked_reasons"]))
    with _registry(scope) as state:
        record = _registered(state, version)
        if effective < record["trained_as_of_date"]:
            raise ValueError("Approval cannot precede training")
        if record["status"] == "approved":
            return dict(record)
        record.update(status="approved", approval_effective_date=effective,
                      approval_review=assessment, approved_by=reviewer,
                      approval_reason=reason)
        _event(state, "approve", version, effective, reviewer, reason,
               review_required=assessment["review_required"])
        return dict(record)


def _deployment(state: dict, as_of: str) -> dict | None:
    events = [e for e in state["events"]
              if e["action"] in {"promote", "rollback"} and e["effective_date"] <= as_of]
    return max(events, key=lambda e: (e["effective_date"], e["sequence"])) if events else None


def resolve_approved_model(as_of: str, scope: str = "production") -> str:
    effective = _date(as_of)
    with _registry(scope) as state:
        deployment = _deployment(state, effective)
        if deployment is None:
            raise RuntimeError(f"No approved model has been promoted by {effective} in {scope}")
        record = _registered(state, deployment["model_version"])
        if (record["status"] != "approved" or
                record["approval_effective_date"] > effective or
                record["trained_as_of_date"] > effective):
            raise RuntimeError("Deployment does not have a time-eligible approval")
        return deployment["model_version"]


def _deploy(version: str, as_of: str, reviewer: str, reason: str,
            scope: str, action: str) -> dict:
    _identity(reviewer, reason)
    effective = _date(as_of)
    with _registry(scope) as state:
        record = _registered(state, version)
        if record["status"] != "approved" or record["approval_effective_date"] > effective:
            raise RuntimeError("Only an approved model can be deployed")
        matches = [e for e in state["events"] if e["action"] == action
                   and e["model_version"] == version and e["effective_date"] == effective]
        if matches:
            return dict(matches[-1])
        deployments = [e for e in state["events"] if e["action"] in {"promote", "rollback"}]
        if deployments and effective < max(e["effective_date"] for e in deployments):
            raise ValueError("A new deployment cannot rewrite earlier deployment history")
        if action == "rollback":
            current = _deployment(state, effective)
            prior = [e for e in deployments if e["model_version"] == version
                     and e["effective_date"] <= effective]
            if not prior or current is None or current["model_version"] == version:
                raise RuntimeError("Rollback requires a different, previously deployed approved model")
        return dict(_event(state, action, version, effective, reviewer, reason))


def promote_model(version: str, as_of: str, reviewer: str, reason: str,
                  scope: str = "production") -> dict:
    assessment = assess_candidate(version, _date(as_of))
    if not assessment["promotion_eligible"]:
        raise RuntimeError("Promotion blocked by validation/data gates")
    return _deploy(version, as_of, reviewer, reason, scope, "promote")


def rollback_model(version: str, as_of: str, reviewer: str, reason: str,
                   scope: str = "production") -> dict:
    return _deploy(version, as_of, reviewer, reason, scope, "rollback")


def simulate_review_and_promote(as_of: str) -> dict:
    """Reproduce an explicitly simulated approval, isolated from production."""
    effective = _date(as_of)
    version = "credit_model_" + effective.replace("-", "_")
    if not (MODEL_BANK / f"{version}.joblib").exists():
        return {"status": "no_application_cohort", "as_of_date": effective}
    reviewer = "historical_policy_simulation"
    reason = ("Historical replay of validation and data gates; recorded policy review "
              "is simulated and does not authorize production")
    assessment = assess_candidate(version, effective)
    if not assessment["promotion_eligible"]:
        return {"status": "candidate_held", "as_of_date": effective,
                "blocked_reasons": assessment["blocked_reasons"]}
    approve_model(version, effective, reviewer, reason, "historical")
    event = promote_model(version, effective, reviewer, reason, "historical")
    return {"status": "simulated_approval_and_promotion", **event}


def registry_status(scope: str = "production") -> dict:
    with _registry(scope) as state:
        return json.loads(json.dumps(state))


def require_unregistered_for_rebuild(version: str) -> None:
    for scope in SCOPES:
        if version in registry_status(scope)["models"]:
            raise RuntimeError("Registered artifacts are immutable; use the full historical "
                               "backfill with --force-train, or a new production version")


def prepare_historical_rebuild() -> None:
    """Archive simulated decisions before an explicitly forced full rebuild."""
    if registry_status("production")["models"]:
        raise RuntimeError("Production-registered artifacts are immutable; use new model versions")
    with _registry("historical") as state:
        if state["models"]:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            (REGISTRY_DIR / f"historical_archive_{stamp}.json").write_text(json.dumps(state, indent=2))
        state.update(models={}, events=[])
