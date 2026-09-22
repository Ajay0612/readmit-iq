"""Execute the data-understanding notebook using the current virtual environment."""

import os
import sys
from pathlib import Path

import nbformat
from ipykernel.kernelspec import install
from jupyter_client import KernelManager
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(root / ".cache/ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".cache/jupyter"))
path = root / "notebooks/01_data_understanding.ipynb"
notebook = nbformat.read(path, as_version=4)
# Register inside .venv so interactive Jupyter selects the same interpreter.
install(prefix=sys.prefix, kernel_name="readmit-iq", display_name="Python 3 (ReadmitIQ .venv)")
notebook.metadata.kernelspec = {
    "display_name": "Python 3 (ReadmitIQ .venv)",
    "language": "python",
    "name": "readmit-iq",
}
manager = KernelManager(kernel_name="readmit-iq")
manager.kernel_spec.argv = [
    sys.executable,
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}",
]
client = NotebookClient(
    notebook,
    timeout=180,
    kernel_name="readmit-iq",
    km=manager,
    resources={"metadata": {"path": str(root)}},
)
client.execute()
nbformat.write(notebook, path)
print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells")
