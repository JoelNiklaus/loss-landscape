#!/bin/sh

# Check if version number has been incremented!

rm -rf dist # Clean dist folder
python3 setup.py sdist bdist_wheel # Build package
gpg --detach-sign -a dist/* # Sign package
twine upload dist/* # Upload package and signature