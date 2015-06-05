# Build the OpenBR documentation as static html
#
# NOTE: This requires mkdocs to be installed. Please see DOCS.md file for
# instructions

cd scripts
python generate_plugin_docs.py
cd ..

mkdocs build --clean
