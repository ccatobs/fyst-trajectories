"""Sphinx configuration for fyst-trajectories documentation."""

import os
import sys
import warnings
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath("../src"))

# Suppress the upstream sphinx_autodoc_typehints deprecation warning about
# _RstSnippetParser.set_application being removed in Sphinx 10.
warnings.filterwarnings(
    "ignore",
    message=".*set_application.*is deprecated.*",
    category=DeprecationWarning,
)

project = "fyst-trajectories"
copyright = "[TBD]"
author = "Graham Gibson"

release = _pkg_version("fyst-trajectories")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

html_theme = "sphinx_rtd_theme"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["matplotlib"]
