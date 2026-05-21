import json
from pathlib import Path


def test_research_notebooks_are_valid_and_output_free() -> None:
    notebook_paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert len(notebook_paths) == 5

    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs", []) == []
                assert cell.get("execution_count") is None
