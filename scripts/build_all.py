"""Regenerate diagram, evaluation doc, PDF report and presentation from current artifacts.

    python scripts/build_all.py --author "Your Name" --institution "Your College"
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_docs, build_presentation, build_report, make_architecture_diagram
from config import settings

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="[Your Name]")
    ap.add_argument("--institution", default="[Your Institution]")
    a = ap.parse_args()
    if not settings.METADATA_PATH.exists():
        raise SystemExit("Train first: python -m src.train")
    make_architecture_diagram.main()
    build_docs.main()
    print("Wrote", build_report.build(a.author, a.institution,
                                      settings.REPORTS_DIR / "ChurnGuard_AI_Project_Report.pdf"))
    print("Wrote", build_presentation.build(a.author, settings.REPORTS_DIR / "ChurnGuard_AI_Presentation.pptx",
                                            ROOT / "docs" / "Presentation_Speaking_Script.md"))
