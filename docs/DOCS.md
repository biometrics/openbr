# OpenBR Documentation Guide

This is a quick guide for generating the OpenBR documentation on your local machine. The documentation is available online at www.openbiometrics.org.

OpenBR's documentation is built with MkDocs. Please see their website for additional information at www.mkdocs.org. Installing mkdocs with pip is super easy-

    $ pip install mkdocs

Please note that you need Python 2.7 and above already installed. Once mkdocs is installed you can run build_docs.sh to build static html pages at openbr/docs/site. However, for viewing we recommend serving the docs to your browser using a simple http server. Run

    $ mkdocs serve

from openbr/docs and then go to http://127.0.0.1:8000 in your internet browser of choice to view the docs locally.
