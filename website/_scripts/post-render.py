"""Quarto post-render script: copy the downloadable CSV, JSON and SVG files into the built site."""
from pollofpolls.report.render import publish_files

publish_files()
