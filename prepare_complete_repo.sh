#!/usr/bin/env bash

prepare_complete_repo() {
  project_source="${1:-/home/zervou/research/spt_forecast_then_test}"
  repository_root="$(cd "$(dirname "$0")" && pwd)"

  if [ ! -d "$project_source" ]; then
    printf 'Project directory not found: %s\n' "$project_source"
    return 1
  fi

  if [ ! -d "$project_source/spt" ]; then
    printf 'Missing source package: %s/spt\n' "$project_source"
    return 1
  fi

  mkdir -p "$repository_root/spt"
  cp -R "$project_source/spt/." "$repository_root/spt/"

  if [ -f "$project_source/power_decomposition.py" ]; then
    cp "$project_source/power_decomposition.py" "$repository_root/power_decomposition.py"
  else
    printf 'Warning: power_decomposition.py was not found. ESA decomposition code will remain incomplete.\n'
  fi

  for dependency_file in requirements.txt pyproject.toml; do
    if [ -f "$project_source/$dependency_file" ]; then
      cp "$project_source/$dependency_file" "$repository_root/$dependency_file"
    fi
  done

  find "$repository_root" -type d -name __pycache__ -prune -exec rm -r {} + 2>/dev/null
  find "$repository_root" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null

  printf 'Copied the shared source package into %s\n' "$repository_root"
  printf 'Data, checkpoints, caches, outputs, and virtual environments were not copied.\n'
}

prepare_complete_repo "$@"
