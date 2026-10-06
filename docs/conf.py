"""Sphinx configuration for fyst-trajectories documentation."""

import os
import sys
import typing
import warnings

sys.path.insert(0, os.path.abspath("../src"))

# Suppress the upstream sphinx_autodoc_typehints deprecation warning about
# _RstSnippetParser.set_application being removed in Sphinx 10.
warnings.filterwarnings(
    "ignore",
    message=".*set_application.*is deprecated.*",
    category=DeprecationWarning,
)

project = "fyst-trajectories"
copyright = "2026, Graham Gibson"
author = "Graham Gibson"

from fyst_trajectories import __version__ as release  # noqa: E402

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
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# custom.css lets table cells wrap; the theme keeps each cell on one line.
html_css_files = ["custom.css"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# Optional dependencies absent from the docs build environment. matplotlib
# backs the visualization subpackage; sun_avoidance is the shared
# ccatobs/sun-avoidance library (CCAT-internal) behind
# make_sun_safe("cone"/"cad") and the TYPE_CHECKING-guarded AvoidanceData
# annotations in sun_models (installed from a git clone, never from PyPI,
# so CI docs builds do not have it).
autodoc_mock_imports = ["matplotlib", "sun_avoidance"]


def _hide_generic_bases(app, name, obj, options, bases):
    """Drop ``Generic[...]`` from the bases a class entry lists.

    A generic class such as ``ScanBlock`` would otherwise show its private
    type variable (``Bases: Generic[...]``); without it the entry shows
    ``Bases: object``, like the package's other dataclasses.
    """
    bases[:] = [b for b in bases if typing.get_origin(b) is not typing.Generic] or [object]


def _format_private_type_variable(annotation, config=None):
    """Render a private bounded type variable as its bound.

    ``ScanBlock.computed_params`` is annotated with the class's private type
    variable; its entry shows the bound, the union of the computed-parameter
    schemas, instead. Anything else keeps the default formatting.
    """
    from sphinx_autodoc_typehints import format_annotation

    if (
        config is not None
        and isinstance(annotation, typing.TypeVar)
        and annotation.__name__.startswith("_")
        and annotation.__bound__ is not None
    ):
        return format_annotation(annotation.__bound__, config)
    return None


typehints_formatter = _format_private_type_variable
# Sphinx warns that it cannot cache a function-valued setting between builds;
# the only cost is that an incremental build re-reads every page.
suppress_warnings = ["config.cache"]


def setup(app):
    """Connect the autodoc hooks of this build."""
    app.connect("autodoc-process-bases", _hide_generic_bases)
