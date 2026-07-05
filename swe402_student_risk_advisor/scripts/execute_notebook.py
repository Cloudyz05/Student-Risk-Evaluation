from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
VENV_SCRIPTS = PROJECT_ROOT / ".venv" / "Scripts"
os.environ["PATH"] = f"{VENV_SCRIPTS}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ["VIRTUAL_ENV"] = str(PROJECT_ROOT / ".venv")

INPUT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "student_risk_advisor.ipynb"
OUTPUT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "student_risk_advisor_executed.ipynb"


def main() -> None:
    notebook = nbformat.read(INPUT_NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, OUTPUT_NOTEBOOK)
    print(f"Executed notebook written to: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
