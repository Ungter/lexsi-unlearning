# Helpers shared by scripts/vast/remote_run.sh and custom job scripts (source it; needs S3_PREFIX).
#   stage MSG              log a stage marker (shown by launch.py status / the run log)
#   fail MSG               print and exit 1 (remote_run.sh adds: sync results, mark FAILED)
#   sync_up                mirror results/ and run.log to $S3_PREFIX
#   fetch_model REF DIR    s3://... prefix -> download into DIR and print DIR; anything else (HF id, local path) -> print REF
#   upload_model DIR NAME  upload DIR to s3://<bucket>/models/$RUN_ID/NAME
export PYTHONPATH=/root/lexsi/src PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 PIP_BREAK_SYSTEM_PACKAGES=1
PY=${PY:-python3}
BUCKET_URI=$(echo "$S3_PREFIX" | cut -d/ -f1-3)
stamp() { date -u +%FT%TZ; }
stage() { echo "== $(stamp) $1"; }
fail() { echo "== $(stamp) FAILED: $1"; exit 1; }
sync_up() {
  [ -d results ] && $PY -m lfm3u.storage upload results "$S3_PREFIX/results" >/dev/null 2>&1
  $PY -m lfm3u.storage put run.log "$S3_PREFIX/run.log" >/dev/null 2>&1 || true
}
fetch_model() {
  local ref=$1 dir=$2
  if [[ $ref == s3://* ]]; then
    if [ ! -f "$dir/config.json" ]; then
      $PY -m lfm3u.storage download "$ref" "$dir" >&2 || return 1
    fi
    echo "$dir"
  else
    echo "$ref"
  fi
}
upload_model() { $PY -m lfm3u.storage upload "$1" "$BUCKET_URI/models/$RUN_ID/$2"; }
