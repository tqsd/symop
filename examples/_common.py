from __future__ import annotations

from pathlib import Path
from typing import Any


def finalize_figure(
    fig: Any,
    *,
    file: str | None,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """
    Common end-of-script behavior for examples.

    - Saves a PNG next to the example script when file is available.
    - Shows the figure interactively when possible.

    Under Sphinx-Gallery, file may be None and show should be avoided by
    guarding the call with `if __name__ == "__main__":`.
    """
    if save and file:
        out = Path(file).with_suffix(".png")
        try:
            fig.savefig(out, dpi=dpi)
            print("saved figure to:", out)
        except Exception:
            pass

    if show:
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except Exception:
            pass
