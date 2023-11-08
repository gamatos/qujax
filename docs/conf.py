import os
import sys
import importlib
import inspect
import pathlib

sys.path.insert(0, os.path.abspath(".."))  # pylint: disable=wrong-import-position

from qujax.version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qujax"
project_copyright = "2023, The qujax authors"
author = "Sam Duffield, Gabriel Matos, Melf Johannsen"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

autodoc_typehints = "description"

autodoc_type_aliases = {
    "jnp.ndarray": "ndarray",
    "random.PRNGKeyArray": "jax.random.PRNGKeyArray",
    "UnionCallableOptionalArray": "Union[Callable[[ndarray, Optional[ndarray]], ndarray], "
    "Callable[[Optional[ndarray]], ndarray]]",
}

latex_engine = "pdflatex"

titles_only = True

rst_prolog = """
.. role:: python(code)
   :language: python
"""

html_logo = "logo.svg"

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "collapse_navigation": False,
    "prev_next_buttons_location": "None",
}



def linkcode_resolve(domain, info):
    github_url = f"https://github.com/CQCL/qujax/tree/develop/qujax"

    if domain != "py":
        return

    module = importlib.import_module(info["module"])
    object = getattr(module, info["fullname"])

    try:
        path = pathlib.Path(inspect.getsourcefile(object))
        file_name = path.name
        lines = inspect.getsourcelines(object)
    except TypeError:
        return None
    
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{github_url}/{file_name}#L{start}-L{end}"