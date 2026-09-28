#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 SECTION PROVENANCE NOTE" >&2
  exit 2
fi

section="${1^^}"
provenance="${2^^}"
shift 2
note="$*"
memory_file="${MEMORY_FILE:-.agents/MEMORY.md}"

case "$section" in
  PLANS|DECISIONS|PROGRESS|DISCOVERIES|OUTCOMES) ;;
  *) echo "invalid section: $section" >&2; exit 2 ;;
esac

case "$provenance" in
  USER|CODE|TOOL|ASSUMPTION) ;;
  *) echo "invalid provenance: $provenance" >&2; exit 2 ;;
esac

timestamp="$(date -Is)"
entry="- ${timestamp} [${provenance}] ${note}"
temp_file="$(mktemp)"
awk -v header="## [$section]" -v entry="$entry" '
  { print }
  $0 == header { print ""; print entry }
' "$memory_file" > "$temp_file"
mv "$temp_file" "$memory_file"
