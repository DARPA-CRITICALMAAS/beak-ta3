#!/bin/bash
# Installs and builds Sphinx with necessary extensions. If run more than once, will search for updates in codebase to add to html  
# 05/2022 Aidan Sorensen

author="Michigan Tech Research Institute"
release=1
version=1.0.0
currentdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" # Find current directory

if [ ! -d docs/source ]; then
    mkdir -p docs/source
fi
cp -r $currentdir/staticdocs docs/source

# remove pre-existing docs before rebuilding
rm -rf docs/source/*.rst
rm -rf docs/source/conf.py
rm -rf docs/build/*


proj_name=$(basename $currentdir) # Project name generated from the parent folder name

# subshell
(
    cd docs/
    # # sphinx-apidoc is a command that uses autodoc to import/find docstrings in all packages/subpackages from parent dir
    sphinx-apidoc \
    --templatedir=templates/apidoc \
     -F `# -F forces the generation of conf.py` \
     -a `# -a append module path to sys path` \
     -e `# -e generates a separate rst file for each module, so they show up on separate pages` \
     -M `# -M shows modules first, then submodules. (less redundancy in the TOC)` \
     -A "$author" `# -A Author name var 'author' set at top of script` \
     -H $proj_name `# -H Project name from parent dir` \
     -V $version `# -V Version of the project, set to 1` \
     -R $release `# -R Release number, set to 1` \
     -P `# -P forces documentation of private modules that start with an underscore` \
     -o source/ `# -o source/ tells apidoc to output to the given path (source/)` \
      ../src/ \
      --extensions sphinx.ext.napoleon,sphinx.ext.mathjax,sphinx.ext.intersphinx,myst_parser \

    # # Now that the conf.py file has been generated, we'll modify it to do what we want it to do:
    # sphinx-build is a command used to build the .rst files into html, put them in the build dir
    # -D modifies conf.py at runtime to include these options. Does not save these changes in conf.py for later use
    sphinx-build -b html `# -b build in this format:html` \
     -D html_theme="sphinx_rtd_theme" `# Sets theme to readthedocs, looks significantly better than alabaster` \
     -D html_theme_options.sticky_navigation=True `# Pins navigation so it stays on page when scrolling` \
     -D html_theme_options.collapse_navigation=False `# Allows expandable branch button in TOC when navigating ` \
     `#-D numfig=False -D math_numfig=False -D math_number_all=False # numfig, math_numfig, math_number_all work together to enable equation labeling` \
     source/ build/html/
     
)
#to open static site after build: firefox docs/build/html/index.html (any .html in this dir will work)

