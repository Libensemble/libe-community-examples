#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import os
import sys
from datetime import datetime

from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
            return MagicMock()

MOCK_MODULES = [
        'deap',
        'vtmop.vtmop',
        'consensus.gens',
        'dragonfly',
        'dragonfly.opt',
        'dragonfly.opt.gp_bandit',
        'dragonfly.exd',
        'dragonfly.exd.cp_domain_utils',
        'dragonfly.exd.domains',
        'dragonfly.exd.experiment_caller'
        ]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

sys.path.append(os.path.abspath('../ax-multitask'))
sys.path.append(os.path.abspath('../consensus'))
sys.path.append(os.path.abspath('../consensus/gens'))
sys.path.append(os.path.abspath('../deap'))
sys.path.append(os.path.abspath('../vtmop'))
sys.path.append(os.path.abspath('../heffte_ytopt/'))
sys.path.append(os.path.abspath('../icesheet/gen_funcs'))
sys.path.append(os.path.abspath('../parmoo-emittance'))
sys.path.append(os.path.abspath('../gp_dragonfly'))
sys.path.append(os.path.abspath('../warpx'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '2.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.imgconverter',
              'sphinx.ext.mathjax']

intersphinx_mapping = {
    'main': ('https://libensemble.readthedocs.io/en/main/', None)
}

extlinks = {'duref': ('http://docutils.sourceforge.net/docs/ref/rst/'
                      'restructuredtext.html#%s', ''),
            'durole': ('http://docutils.sourceforge.net/docs/ref/rst/'
                       'roles.html#%s', ''),
            'dudir': ('http://docutils.sourceforge.net/docs/ref/rst/'
                      'directives.html#%s', '')}


# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The latex toctree document.
# General information about the project.
project = 'libEnsemble'
copyright = str(datetime.now().year) + ' Argonne National Laboratory'
author = 'Jeffrey Larson, Stephen Hudson, Stefan M. Wild, David Bindel and John-Luke Navarro'
today_fmt = '%B %-d, %Y'

#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {'navigation_depth': 3,
                      'collapse_navigation': False}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
html_static_path = []

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'libEnsembledoc'
