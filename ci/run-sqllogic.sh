#!/usr/bin/env bash
# Run sqllogictest files through the published haybarn-unittest with the signed
# community `vgi` extension -- the same setup as the vgi-starter templates, no local
# DuckDB/vgi builds. Drop-in for `unittest --test-dir DIR PATTERN`; environment
# variables the tests read (require-env ...) pass straight through.
#
#   ci/run-sqllogic.sh --test-dir . "test/sql/*"
#
# HAYBARN_VERSION (default 2.0.0rc11) picks the runner; it is installed into .tools/.
set -euo pipefail
HAYBARN_VERSION="${HAYBARN_VERSION:-2.0.0rc11}"
TOOLS="${TOOLS:-.tools}"
HAYBARN_UNITTEST="${HAYBARN_UNITTEST:-$TOOLS/bin/haybarn-unittest}"
TEST_DIR=.
PATTERN="test/sql/*"
while [ $# -gt 0 ]; do
  case "$1" in
    --test-dir) TEST_DIR="$2"; shift 2 ;;
    *) PATTERN="$1"; shift ;;
  esac
done
if [ ! -x "$HAYBARN_UNITTEST" ]; then
  uv venv -q "$TOOLS" && uv pip install -q -p "$TOOLS" "haybarn-unittest==$HAYBARN_VERSION"
fi
HAYBARN_UNITTEST="$(cd "$(dirname "$HAYBARN_UNITTEST")" && pwd)/$(basename "$HAYBARN_UNITTEST")"
SRC="$(cd "$TEST_DIR" && pwd)"
# Stage inside the test dir: the runner changes into --test-dir, so relative worker
# commands (uv run foo_worker.py) and data paths keep resolving from the repo root.
STAGE_REL="test/_sqllogic_stage"
STAGE="$SRC/$STAGE_REL"
rm -rf "$STAGE"; mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT

# The standalone runner links no extensions: turn `require <ext>` into a LOAD.
shopt -s nullglob
files=( "$SRC"/$PATTERN )
[ ${#files[@]} -gt 0 ] || { echo "no test files match $SRC/$PATTERN" >&2; exit 1; }
for f in "${files[@]}"; do
  rel="${f#"$SRC"/}"
  mkdir -p "$STAGE/$(dirname "$rel")"
  awk '/^require[ \t]+(vgi|httpfs|json)[ \t]*$/ { print "statement ok"; print "LOAD " $2 ";"; print ""; next } { print }' \
    "$f" > "$STAGE/$rel"
done

# Install the extensions once (refreshing any stale vgi), bounded so a stalled
# download can't hang the run.
cat > "$STAGE/_warm.test" <<'SQL'
statement ok
FORCE INSTALL vgi FROM community;

statement ok
INSTALL httpfs FROM core;

statement ok
INSTALL json FROM core;
SQL
timeout "${WARMUP_TIMEOUT:-300}" "$HAYBARN_UNITTEST" --keep-home --test-dir "$SRC" "$STAGE_REL/_warm.test" >/dev/null 2>&1 \
  || echo "warning: extension warm-up did not fully succeed" >&2
rm -f "$STAGE/_warm.test"

cd "$SRC"
"$HAYBARN_UNITTEST" --keep-home --test-dir "$SRC" "$STAGE_REL/$PATTERN"
