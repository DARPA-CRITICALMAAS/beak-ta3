# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, '/beak-ta3/src')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'beak-ta3'
copyright = '2024, Michigan Tech Research Institute'
author = 'Michigan Tech Research Institute'

version = '1.0.0'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

# enabling dollar math, get back to this for others
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    #"linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# we need this for documentation of __call__ methods to show up
autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__,__call__',
    'undoc-members': True
}

# suppress annoying warnings about footnotes
suppress_warnings = ["ref.footnote"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration


intersphinx_mapping = {

    # The entries in this list are modified from github.com/bskinn/intersphinx-gist.
    # Please feel free to post an issue at that repo if any of these mappings don't work for you,
    # or if you're having trouble constructing a mapping for a project not listed here.

    "python": ('https://docs.python.org/3/', None),
    "h5py": ('https://docs.h5py.org/en/latest/', None),
    "matplotlib": ('https://matplotlib.org/stable/', None),
    "numpy": ('https://numpy.org/doc/stable/', None),
    "pandas": ('https://pandas.pydata.org/docs/', None),
    "scikit-learn": ('https://scikit-learn.org/stable/', None),
    "scipy": ('https://docs.scipy.org/doc/scipy/', None),
    "sphinx": ('https://www.sphinx-doc.org/en/master/', None)

}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
