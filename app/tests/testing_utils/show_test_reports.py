#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Utility function to display test report."""


from os import terminal_size
from pathlib import Path
from shutil import get_terminal_size
from typing import Tuple


def show_test_outputs(module_dir: str = "None"):
    """Show test code coverage annotated in files."""
    term_dims = get_terminal_size((80, 20))  # type: terminal_size
    n_dashes, n_rem = divmod(term_dims[0], 2)  # type: Tuple[int, int]
    smsg = ""  # type: str
    file_print_divider = smsg.join(
        ["=" * (n_dashes + int(n_rem / 2))] * 2
    )  # type: str

    PROJECT_DIR = Path(__file__).parents[1]  # type: Path

    with open(PROJECT_DIR / "test-logs" / "report.md") as f:
        print(f"{f.read()}{file_print_divider}")

    if module_dir != "None":
        for fn in Path(module_dir).glob("**/*.py"):
            with open(fn) as f:
                print(f"{f.read()}{file_print_divider}")


if __name__ == "__main__":
    show_test_outputs(module_dir="None")
