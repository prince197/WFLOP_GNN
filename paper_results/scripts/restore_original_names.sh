#!/bin/bash
# The analysis scripts in this folder read the files under their original campaign names.
# This creates those names as symbolic links next to the scripts, using file_name_map.tsv,
# so that e.g. "python reanalyze.py new" can be run from this folder unchanged.
cd "$(dirname "$0")"
while IFS=$'\t' read -r orig desc; do
  [ -n "$orig" ] && ln -sf "../$desc" "$orig"
done < file_name_map.tsv
echo "Linked $(grep -c . file_name_map.tsv) files. Run the scripts from $(pwd)."
