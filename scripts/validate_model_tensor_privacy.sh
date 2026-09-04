#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${EMEL_PRIVACY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/model-tensor-privacy}"
CARGO="${CARGO:-cargo}"

for required_tool in date mkdir mktemp mv rm; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    printf 'error: required tool not found: %s\n' "$required_tool" >&2
    exit 1
  fi
done
if ! command -v "$CARGO" >/dev/null 2>&1; then
  printf 'error: configured CARGO command not found: %s\n' "$CARGO" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$ARTIFACT_DIR"

RUN_TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
SUMMARY="$ARTIFACT_DIR/${RUN_ID}-summary.txt"
summary_tmp=""
cleanup() {
  if [[ -n "$summary_tmp" ]]; then
    rm -f "$summary_tmp"
  fi
}
trap cleanup EXIT

source_repository='https://github.com/stateforward/emel.rs'
source_commit='unavailable'
source_worktree='unavailable'
if command -v git >/dev/null 2>&1; then
  configured_repository="$(git config --get remote.origin.url 2>/dev/null || true)"
  if [[ -n "$configured_repository" ]]; then
    source_repository="$configured_repository"
  fi
  source_commit="$(git rev-parse HEAD 2>/dev/null || printf 'unavailable')"
  if [[ "$(git status --porcelain=v1 --untracked-files=all 2>/dev/null)" ]]; then
    source_worktree='dirty'
  else
    source_worktree='clean'
  fi
fi

test_names=()
test_commands=()
test_statuses=()
test_logs=()
overall_status=0

run_test() {
  local name="$1"
  shift
  local log_file="$ARTIFACT_DIR/${RUN_ID}-${name}.log"
  local command_display
  printf -v command_display '%q ' "$CARGO" "$@"
  command_display="${command_display% }"

  printf '%s\n' "$command_display" >"$log_file"
  printf 'started_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$log_file"
  set +e
  "$CARGO" "$@" >>"$log_file" 2>&1
  local status=$?
  set -e
  printf 'exit_status=%d\n' "$status" >>"$log_file"

  test_names+=("$name")
  test_commands+=("$command_display")
  test_statuses+=("$status")
  test_logs+=("${log_file##*/}")
  if (( status != 0 )); then
    overall_status=1
  fi
}

# Run the maintained tests by exact names: model-owned byte views, then the
# tensor-window ownership/lifecycle paths in the emel-model library target.
run_test 'data_construction_copies_resident_storage_and_exposes_checked_views' \
  test --offline --locked -p emel-model --test model_data_privacy \
  data_construction_copies_resident_storage_and_exposes_checked_views
run_test 'fitting_bind_acquire_is_allocation_free_and_unbinds' \
  test --offline --locked -p emel-model --lib \
  tensor::window::actor::tests::fitting_bind_acquire_is_allocation_free_and_unbinds
run_test 'resident_acquire_reuses_slot_without_second_copy' \
  test --offline --locked -p emel-model --lib \
  tensor::window::actor::tests::resident_acquire_reuses_slot_without_second_copy
run_test 'unbind_clears_residency_and_allows_clean_rebind' \
  test --offline --locked -p emel-model --lib \
  tensor::window::actor::tests::unbind_clears_residency_and_allows_clean_rebind
run_test 'failed_slot_retries_and_commits_on_next_acquire' \
  test --offline --locked -p emel-model --lib \
  tensor::window::actor::tests::failed_slot_retries_and_commits_on_next_acquire

summary_tmp="$(mktemp "$ARTIFACT_DIR/.${RUN_ID}-summary.XXXXXX")"
{
  printf 'schema_version=1\n'
  printf 'validator=model-tensor-privacy\n'
  if (( overall_status == 0 )); then
    printf 'status=PASS\n'
  else
    printf 'status=FAIL\n'
  fi
  printf 'timestamp_utc=%s\n' "$RUN_TIMESTAMP"
  printf 'source_repository=%s\n' "$source_repository"
  printf 'source_commit=%s\n' "$source_commit"
  printf 'source_worktree=%s\n' "$source_worktree"
  printf 'fixture=model_data_privacy resident byte-view ownership; tensor::window::actor ownership and lifecycle\n'
  printf 'configuration.cargo=%s\n' "$CARGO"
  printf 'configuration.working_directory=repository_root\n'
  printf 'configuration.cargo_flags=--offline --locked\n'
  printf 'test_count=%d\n' "${#test_names[@]}"
  for ((index = 0; index < ${#test_names[@]}; index++)); do
    number=$((index + 1))
    printf 'test_%d_name=%s\n' "$number" "${test_names[$index]}"
    printf 'test_%d_command=%s\n' "$number" "${test_commands[$index]}"
    printf 'test_%d_exit_status=%s\n' "$number" "${test_statuses[$index]}"
    printf 'test_%d_log=%s\n' "$number" "${test_logs[$index]}"
  done
} >"$summary_tmp"
mv "$summary_tmp" "$SUMMARY"
summary_tmp=''

if (( overall_status != 0 )); then
  printf 'Model tensor privacy validation failed; see %s\n' "${SUMMARY##*/}" >&2
  exit "$overall_status"
fi
printf 'Model tensor privacy validation passed; see %s\n' "${SUMMARY##*/}"
