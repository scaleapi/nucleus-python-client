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

import pkg_resources

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "Nucleus"
copyright = "2021, Scale"
author = "Scale"


# The full version, including alpha/beta/rc tags
release = "v" + str(pkg_resources.get_distribution("scale-nucleus").version)


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
html_title = "Nucleus API Reference"
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_favicon = "favicon.ico"

html_logo = "nucleus-logo.svg"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
}


# -- autogen configuration ---------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../nucleus"]
autoapi_options = [
    "members",
    "no-undoc-members",
    "inherited-members",
    "show-module-summary",
    "imported-members",
]
autoapi_template_dir = "_templates"
autoapi_root = "api"
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autodoc_typehints = "description"
autoapi_add_toctree_entry = False
napoleon_include_init_with_doc = True
