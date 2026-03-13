"""
Configuration for the AgentReview multi-agent pipeline.

Defines pipeline options, agent toggles, and path conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Output directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    workspace_dir: Path = field(default_factory=lambda: Path(__file__).parent / "workspace")

    # Pipeline toggles (each agent can be enabled/disabled)
    enable_injection_detector: bool = True
    enable_novelty: bool = True
    enable_baselines: bool = True
    enable_code_reviewer: bool = False  # Only when code is provided
    enable_soundness: bool = True
    enable_writing_quality: bool = True
    enable_reproducibility: bool = True
    enable_ethics_limitations: bool = True

    # Venue list for baseline search
    top_venues: list[str] = field(default_factory=lambda: [
        "NeurIPS", "ICML", "ICLR", "CVPR", "ECCV", "ICCV", "ACL", "EMNLP",
        "AAAI", "IJCAI", "SIGGRAPH", "KDD", "WWW", "NAACL", "CoRL",
    ])

    # Review format
    review_format: str = "openreview"  # "openreview" or "plain"

    # Parser settings
    parser_backend: str = "mineru"  # PDF-to-markdown tool

    # File size limits
    max_upload_mb: int = 100  # Max upload size in MB

    def ensure_dirs(self):
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
