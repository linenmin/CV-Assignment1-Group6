import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True, help="Path to the notebook file")
    parser.add_argument("--until-cell", type=int, required=True, help="Last cell index to execute")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save notebook after every N executed code cells",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook).resolve()
    workdir = notebook_path.parent

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(workdir)}},
        allow_errors=False,
    )

    executed_code_cells = 0
    important = {54, 55, 56, 57, 58, 59, 60}

    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if index > args.until_cell:
                break
            if cell.cell_type != "code":
                continue

            first_line = "".join(cell.source).strip().splitlines()
            preview = first_line[0][:120] if first_line else ""
            print(f"[cell {index}] executing: {preview}", flush=True)
            client.execute_cell(cell, index)
            executed_code_cells += 1

            if index in important:
                print(f"[cell {index}] completed", flush=True)

            if executed_code_cells % max(args.save_every, 1) == 0:
                with notebook_path.open("w", encoding="utf-8") as f:
                    nbformat.write(nb, f)
                print(
                    f"[save] wrote notebook after {executed_code_cells} executed code cells",
                    flush=True,
                )

    with notebook_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[done] notebook saved to {notebook_path}", flush=True)


if __name__ == "__main__":
    main()
