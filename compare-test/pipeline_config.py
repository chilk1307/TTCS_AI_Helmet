"""Paths for pipeline queue vs no-queue test."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

VIDEO_PATH = os.path.join(ROOT, "IMG_3115 - Trim.mp4")
RESULTS_DIR = os.path.join(ROOT, "results", "pipeline_test")

# Isolated storage for test runs (do not touch main storage)
EVIDENCE_DIR = os.path.join(RESULTS_DIR, "evidence")
VIOLATIONS_DIR = os.path.join(RESULTS_DIR, "violations")
VIOLATIONS_FILE = os.path.join(VIOLATIONS_DIR, "violations.xlsx")
