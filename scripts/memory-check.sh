#!/usr/bin/env bash
set -euo pipefail

memory_file="${MEMORY_FILE:-.agents/MEMORY.md}"

for section in PLANS DECISIONS PROGRESS DISCOVERIES OUTCOMES; do
  if ! grep -Fxq "## [$section]" "$memory_file"; then
    echo "missing section: $section" >&2
    exit 1
  fi
done

if grep -E '^- ' "$memory_file" | grep -Ev '^- [0-9]{4}-[0-9]{2}-[0-9]{2}T[^ ]+ \[(USER|CODE|TOOL|ASSUMPTION)\] .+'; then
  echo "invalid memory entry format" >&2
  exit 1
fi

echo "memory format valid"
