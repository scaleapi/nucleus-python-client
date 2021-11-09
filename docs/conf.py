# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "Nucleus"
copyright = "2021, Scale"
author = "Scale"

# The full version, including alpha/beta/rc tags
from nucleus import __version__  # noqa: E402

release = "v" + str(__version__)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "autoapi.extension",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# autodoc_typehints = "description"
# autodoc_member_order = "groupwise"
# autosummary_generate = True
# autosummary_imported_members = True


def autoapi_prepare_jinja_env(jinja_env):
    # HACK: have to define dummy "underline" filter that autoapi references
    jinja_env.filters["underline"] = lambda value: value.lower()


autoapi_type = "python"
autoapi_dirs = ["../nucleus"]
autoapi_options = [
    "members",
    "undoc-members",
    "inherited-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_template_dir = "_templates"
autoapi_root = "api"

SKIP_KEYWORDS = [
    "ModelRun",
    "Category",
]


def handle_skip(app, what, name, obj, skip, options):
    # skip by keyword
    if what in ("module", "class") and any(
        word in name for word in SKIP_KEYWORDS
    ):
        return True

    # skip where objname is all caps (globals)
    objname = name.split(".")[-1]
    if objname == objname.upper():
        return True

    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", handle_skip)
