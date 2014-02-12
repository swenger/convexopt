To build documentation, go to the root directory (the one that contains
convexopt/ and doc/), and run:

  sphinx-apidoc -o doc convexopt

Then, go into doc/ and run:

  make html

The generated HTML documentation is located at:

  _build/html/index.html

