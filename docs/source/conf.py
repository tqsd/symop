from __future__ import annotations
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "symop_proto"
author = "Simon Sekavčnik"
copyright = f"{date.today().year}, {author}"
release = ""

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
extensions += ["sphinx_autodoc_typehints"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autosummary_generate = True

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
napoleon_custom_sections = [("Mathematics", "Admonition")]

pygments_style = "sphinx"
pygments_dark_style = "native"

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path = ["_static"]

nitpicky = True
nitpick_ignore = []

html_css_files = ["custom.css"]
