"""Evaluation harness: graders that read traces, and the runner that reports."""

from .graders import CheckResult, GRADERS, grade

__all__ = ["CheckResult", "GRADERS", "grade"]
