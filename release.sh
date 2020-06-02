#!/bin/sh

# Check if version number has been incremented!
# Make sure you have created a gpg key (gpg --full-generate-key)
# Make sure you have a pypi account

set -e # Make the script exit immediately when a command exited with a non-zero exit code

rm -rf dist # Clean dist folder
python setup.py sdist bdist_wheel # Build package
twine check dist/* # Check if the README.md is ok
gpg --detach-sign -a dist/* # Sign package
twine upload dist/* # Upload package and signature
