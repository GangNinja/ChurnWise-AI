"""Typed runtime settings for ChurnWise.

Defaults are safe for a first run. Every path and network limit can be
overridden with a ``CHURNWISE_`` environment variable, as documented in
``docs/DISCOVERY.md``. No secrets are read or stored here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_thresholds(
    name: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be comma-separated numbers, got {raw!r}"
        ) from exc
    return values


@dataclass(frozen=True)
class Settings:
    """Resolved project settings.

    ``project_root`` is only used to derive default directories; tests can pass
    temporary directories directly.
    """

    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path | None = None
    artifact_dir: Path | None = None
    report_dir: Path | None = None
    seed: int = 42
    http_timeout_s: float = 12.0
    download_timeout_s: float = 60.0
    max_retries: int = 2
    max_response_bytes: int = 2_000_000
    max_download_bytes: int = 200 * 1024 * 1024
    max_extracted_bytes: int = 300 * 1024 * 1024
    max_archive_members: int = 200
    allow_unverified_license: bool = False
    offline: bool = False
    user_agent: str = "ChurnWiseAI/0.3 (dataset discovery and acquisition)"
    risk_tiers: tuple[float, ...] = (0.30, 0.50, 0.70)

    def __post_init__(self) -> None:
        root = Path(self.project_root)
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "data_dir", Path(self.data_dir or root / "data"))
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir or root / "artifacts"))
        object.__setattr__(self, "report_dir", Path(self.report_dir or root / "reports"))

    def with_overrides(self, **kwargs: object) -> "Settings":
        """Return a copy with validated overrides (used by the CLI)."""
        return replace(self, **kwargs)

    @classmethod
    def load(cls, project_root: Path | None = None) -> "Settings":
        """Build settings from defaults plus ``CHURNWISE_*`` environment values."""
        root = Path(project_root or Path.cwd())
        return cls(
            project_root=root,
            data_dir=Path(_env_str("CHURNWISE_DATA_DIR", str(root / "data"))),
            artifact_dir=Path(_env_str("CHURNWISE_ARTIFACT_DIR", str(root / "artifacts"))),
            report_dir=Path(_env_str("CHURNWISE_REPORT_DIR", str(root / "reports"))),
            seed=_env_int("CHURNWISE_SEED", 42),
            http_timeout_s=_env_float("CHURNWISE_HTTP_TIMEOUT_S", 12.0),
            download_timeout_s=_env_float("CHURNWISE_DOWNLOAD_TIMEOUT_S", 60.0),
            max_retries=_env_int("CHURNWISE_MAX_RETRIES", 2),
            max_response_bytes=_env_int("CHURNWISE_MAX_RESPONSE_BYTES", 2_000_000),
            max_download_bytes=_env_int("CHURNWISE_MAX_DOWNLOAD_BYTES", 200 * 1024 * 1024),
            max_extracted_bytes=_env_int("CHURNWISE_MAX_EXTRACTED_BYTES", 300 * 1024 * 1024),
            max_archive_members=_env_int("CHURNWISE_MAX_ARCHIVE_MEMBERS", 200),
            allow_unverified_license=_env_bool("CHURNWISE_ALLOW_UNVERIFIED_LICENSE", False),
            offline=_env_bool("CHURNWISE_OFFLINE", False),
            user_agent=_env_str("CHURNWISE_USER_AGENT", "ChurnWiseAI/0.3 (dataset discovery and acquisition)"),
            risk_tiers=_env_thresholds(
                "CHURNWISE_RISK_TIERS", (0.30, 0.50, 0.70)
            ),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not (1.0 <= self.http_timeout_s <= 60.0):
            problems.append("http_timeout_s must be between 1 and 60 seconds")
        if not (1.0 <= self.download_timeout_s <= 600.0):
            problems.append("download_timeout_s must be between 1 and 600 seconds")
        if not (0 <= self.max_retries <= 5):
            problems.append("max_retries must be between 0 and 5")
        if self.seed < 0:
            problems.append("seed must be non-negative")
        if not (10_000 <= self.max_response_bytes <= 50_000_000):
            problems.append("max_response_bytes must be between 10 KB and 50 MB")
        if not (1_024 <= self.max_download_bytes <= 2_000_000_000):
            problems.append("max_download_bytes must be between 1 KB and 2 GB")
        if not (1_024 <= self.max_extracted_bytes <= 4_000_000_000):
            problems.append("max_extracted_bytes must be between 1 KB and 4 GB")
        if not (1 <= self.max_archive_members <= 10_000):
            problems.append("max_archive_members must be between 1 and 10,000")
        if not self.user_agent.strip():
            problems.append("user_agent must not be empty")
        tiers = tuple(self.risk_tiers)
        if len(tiers) not in (2, 3):
            problems.append("risk_tiers must contain exactly two or three boundaries")
        elif any(
            not isinstance(value, (int, float))
            or not 0.0 < float(value) < 1.0
            for value in tiers
        ):
            problems.append("risk_tiers boundaries must be strictly between 0 and 1")
        elif any(right <= left for left, right in zip(tiers, tiers[1:])):
            problems.append("risk_tiers boundaries must be strictly increasing")
        return problems
