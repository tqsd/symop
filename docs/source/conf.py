from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# For examples
sys.path.insert(0, str(ROOT / "examples"))

project = "symop"
author = "Simon Sekavčnik"
copyright = f"{date.today().year}, {author}"
release = ""

sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "examples"),
    "gallery_dirs": "examples",
    "filename_pattern": r"plot_.*\.py$",
    "ignore_pattern": r"(^|/)_",
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "jupyter_sphinx",
]
extensions += ["sphinx_gallery.gen_gallery"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}


autosummary_generate = True

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
    "inherited-members": False,
    "private-members": True,
}
autodoc_mock_imports = ["matplotlib"]


add_module_names = False
toc_object_entries_show_parents = "hide"
python_use_unqualified_type_names = True
modindex_common_prefix = ["symop_proto."]

mathjax3_config = {
    "tex": {
        "packages": ["base", "ams"],
    }
}

jupyter_execute_notebooks = "off"
jupyter_execute_timeout = 120
jupyter_execute_data_priority = [
    "text/html",
    "text/markdown",
    "image/svg+xml",
    "image/png",
    "application/json",
    "text/plain",
]
typehints_use_signature = True
typehints_use_signature_return = True
typehints_document_rtype = False
always_document_param_types = False
typehints_fully_qualified = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_custom_sections = [
    ("Mathematics", "Admonition"),
    ("Numerical Note", "Admonition"),
]

pygments_style = "sphinx"
pygments_dark_style = "native"

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path = ["_static"]

nitpicky = True
nitpick_ignore = [
    ("py:class", "SignatureProto"),
    ("py:class", "RCArray"),
    ("py:class", "ndarray"),
    ("py:class", "numpy.ndarray"),
]

html_css_files = ["custom.css"]
