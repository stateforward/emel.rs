#!/usr/bin/env bash
set -euo pipefail

# Source inventory for the pinned generic kernel domain. The generated artifact
# belongs under target/ by default so it remains persistent between commands
# without becoming a source snapshot or relying on /tmp.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
OUTPUT="${EMEL_KERNEL_SOURCE_INVENTORY_OUTPUT:-$ROOT_DIR/target/kernel-source-inventory/source-inventory.tsv}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SOURCE_RELATIVE=src/emel/kernel

EXPECTED_FILES=(
  src/emel/kernel/actions.hpp
  src/emel/kernel/any.hpp
  src/emel/kernel/context.hpp
  src/emel/kernel/detail.hpp
  src/emel/kernel/errors.hpp
  src/emel/kernel/events.hpp
  src/emel/kernel/guards.hpp
  src/emel/kernel/sm.hpp
  src/emel/kernel/aarch64/actions.hpp
  src/emel/kernel/aarch64/context.hpp
  src/emel/kernel/aarch64/detail.hpp
  src/emel/kernel/aarch64/errors.hpp
  src/emel/kernel/aarch64/events.hpp
  src/emel/kernel/aarch64/guards.hpp
  src/emel/kernel/aarch64/sm.hpp
  src/emel/kernel/x86_64/actions.hpp
  src/emel/kernel/x86_64/context.hpp
  src/emel/kernel/x86_64/detail.hpp
  src/emel/kernel/x86_64/errors.hpp
  src/emel/kernel/x86_64/events.hpp
  src/emel/kernel/x86_64/guards.hpp
  src/emel/kernel/x86_64/sm.hpp
)

fail() {
  printf 'error: %s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is missing: $1"
}

for command in git awk diff sort wc tr grep rg; do
  require_command "$command"
done

if [[ $# -gt 1 || ("${1:-}" != "" && "${1:-}" != "--validate") ]]; then
  printf 'usage: scripts/kernel-source-inventory.sh [--validate]\n' >&2
  exit 2
fi

[[ -d "$EMEL_CPP_SOURCE" ]] || fail "reference repository is missing: $EMEL_CPP_SOURCE"
git -C "$EMEL_CPP_SOURCE" rev-parse --git-dir >/dev/null 2>&1 ||
  fail "reference path is not a git repository: $EMEL_CPP_SOURCE"
git -C "$EMEL_CPP_SOURCE" cat-file -e "$SOURCE_COMMIT^{commit}" 2>/dev/null ||
  fail "pinned reference commit is missing: $SOURCE_COMMIT"
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] ||
  fail "reference checkout is not at pinned commit: $SOURCE_COMMIT"
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain)" ]] ||
  fail "reference checkout must be clean: $EMEL_CPP_SOURCE"

validate_maintained_rust() {
  local rust_root="$ROOT_DIR/crates/emel-kernels/src"
  [[ -f "$ROOT_DIR/crates/emel-kernels/src/lib.rs" ]] ||
    fail "maintained kernel crate root is missing"
  rg -q '^#!\[forbid\(unsafe_code\)\]' \
    "$ROOT_DIR/crates/emel-kernels/src/lib.rs" ||
    fail "maintained kernel crate is not safe-Rust by construction"
  if rg -n 'TODO|FIXME|todo!|unimplemented!|scaffold' "$rust_root"; then
    fail "maintained kernel source still contains scaffold or TODO markers"
  fi
  [[ "$(rg -c 'guard_generic_[a-z0-9_]+_valid\]' \
    "$rust_root/x86_64/sm.rs")" == "95" ]] ||
    fail "x86_64 generic SML route count is not the pinned 95"
  [[ "$(rg -c 'guard_generic_[a-z0-9_]+_invalid\]' \
    "$rust_root/x86_64/sm.rs")" == "95" ]] ||
    fail "x86_64 generic invalid SML route count is not the pinned 95"
  [[ "$(rg -c 'guard_generic_[a-z0-9_]+_valid\]' \
    "$rust_root/aarch64/sm.rs")" == "95" ]] ||
    fail "AArch64 generic SML route count is not the pinned 95"
  [[ "$(rg -c 'guard_generic_[a-z0-9_]+_invalid\]' \
    "$rust_root/aarch64/sm.rs")" == "95" ]] ||
    fail "AArch64 generic invalid SML route count is not the pinned 95"
  [[ "$(rg -c 'fn op_' "$rust_root/any/actor.rs")" -ge 95 ]] ||
    fail "portable actor does not expose the complete maintained operation set"

  local x86_sm="$rust_root/x86_64/sm.rs"
  local required_x86_route
  for required_x86_route in \
    Matmul MatmulQ4_0 MatmulQ4_1 MatmulQ5_0 MatmulQ8_0 MatmulQ2K MatmulQ3K MatmulQ4K MatmulQ6K \
    MatmulArgmax MatmulArgmaxQ4_0 MatmulArgmaxQ4_1 MatmulArgmaxQ5_0 MatmulArgmaxQ8_0 \
    MatmulArgmaxQ2K MatmulArgmaxQ3K MatmulArgmaxQ4K MatmulArgmaxQ6K FlashAttn; do
    rg -q "^[[:space:]]*\\\"ready\\\"_s <= \\\"ready\\\"_s \\+ ${required_x86_route}\\(" "$x86_sm" ||
      fail "x86_64 target machine is missing explicit route: $required_x86_route"
  done
  rg -q '^[[:space:]]*\"ready\"_s <= \"ready\"_s \+ F16Matmul\(' "$x86_sm" ||
    fail "x86_64 target machine is missing explicit route: F16Matmul"

  local aarch64_sm="$rust_root/aarch64/sm.rs"
  rg -q '^[[:space:]]*\"ready\"_s <= \"ready\"_s \+ F16Matmul\(' "$aarch64_sm" ||
    fail "AArch64 target machine is missing explicit route: F16Matmul"
}

validate_maintained_rust

expected_listing="$(printf '%s\n' "${EXPECTED_FILES[@]}" | LC_ALL=C sort)"
actual_listing="$(git -C "$EMEL_CPP_SOURCE" ls-tree -r --name-only "$SOURCE_COMMIT" -- "$SOURCE_RELATIVE")"
if [[ "$actual_listing" != "$expected_listing" ]]; then
  printf '%s\n' 'error: pinned kernel source has missing or unclassified files' >&2
  diff -u <(printf '%s\n' "$expected_listing") <(printf '%s\n' "$actual_listing") >&2 || true
  exit 1
fi

source_class() {
  case "$1" in
    src/emel/kernel/aarch64/*.hpp) printf 'aarch64' ;;
    src/emel/kernel/x86_64/*.hpp) printf 'x86_64' ;;
    src/emel/kernel/*.hpp) printf 'core' ;;
    *) fail "unclassified kernel file: $1" ;;
  esac
}

rust_counterpart() {
  case "$1" in
    src/emel/kernel/any.hpp|src/emel/kernel/actions.hpp|\
    src/emel/kernel/context.hpp|src/emel/kernel/errors.hpp)
      printf 'crates/emel-kernels/src/any/actor.rs' ;;
    src/emel/kernel/events.hpp) printf 'crates/emel-kernels/src/any/event.rs' ;;
    src/emel/kernel/guards.hpp|src/emel/kernel/sm.hpp)
      printf 'crates/emel-kernels/src/any/sm.rs' ;;
    src/emel/kernel/detail.hpp) printf 'crates/emel-kernels/src/detail/mod.rs' ;;
    src/emel/kernel/aarch64/sm.hpp)
      printf 'crates/emel-kernels/src/aarch64/sm.rs' ;;
    src/emel/kernel/aarch64/events.hpp)
      printf 'crates/emel-kernels/src/aarch64/event.rs' ;;
    src/emel/kernel/aarch64/actions.hpp|src/emel/kernel/aarch64/context.hpp|\
    src/emel/kernel/aarch64/errors.hpp|src/emel/kernel/aarch64/guards.hpp|\
    src/emel/kernel/aarch64/detail.hpp)
      printf 'crates/emel-kernels/src/aarch64/actor.rs' ;;
    src/emel/kernel/x86_64/sm.hpp)
      printf 'crates/emel-kernels/src/x86_64/sm.rs' ;;
    src/emel/kernel/x86_64/events.hpp)
      printf 'crates/emel-kernels/src/x86_64/event.rs' ;;
    src/emel/kernel/x86_64/actions.hpp|src/emel/kernel/x86_64/context.hpp|\
    src/emel/kernel/x86_64/errors.hpp|src/emel/kernel/x86_64/guards.hpp|\
    src/emel/kernel/x86_64/detail.hpp)
      printf 'crates/emel-kernels/src/x86_64/actor.rs' ;;
    *) fail "unclassified Rust counterpart mapping: $1" ;;
  esac
}

rust_status() {
  local reference="$1"
  local counterpart="$2"
  if [[ "$counterpart" == '-' || ! -e "$ROOT_DIR/$counterpart" ]]; then
    printf 'missing'
    return
  fi
  # Completeness is established by validate_maintained_rust before rows are
  # emitted. A present counterpart is therefore a maintained counterpart.
  printf 'present'
}

source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT^{tree}")"
kernel_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$SOURCE_RELATIVE")"
mkdir -p "$(dirname "$OUTPUT")"

{
  printf 'kernel-source-inventory/v1\n'
  printf 'source_repository=stateforward/emel.cpp\n'
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'source_tree=%s\n' "$source_tree"
  printf 'source_kernel_tree=%s\n' "$kernel_tree"
  printf 'source_relative=%s\n' "$SOURCE_RELATIVE"
  printf 'source_file_count=%s\n' "${#EXPECTED_FILES[@]}"
  printf 'columns=file\tclass\tblob\tline_count\trust_counterpart\trust_status\n'
  for file in "${EXPECTED_FILES[@]}"; do
    class="$(source_class "$file")"
    blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$file")"
    line_count="$(git -C "$EMEL_CPP_SOURCE" show "$SOURCE_COMMIT:$file" | wc -l | tr -d '[:space:]')"
    counterpart="$(rust_counterpart "$file")"
    status="$(rust_status "$file" "$counterpart")"
    printf 'file=%s\tclass=%s\tblob=%s\tline_count=%s\trust_counterpart=%s\trust_status=%s\n' \
      "$file" "$class" "$blob" "$line_count" "$counterpart" "$status"
  done
  printf 'result=valid\n'
} >"$OUTPUT"

awk -F '\t' '
  NR == 1 && $0 == "kernel-source-inventory/v1" { header = 1; next }
  index($0, "columns=file\tclass\tblob\tline_count\trust_counterpart\trust_status") == 1 {
    columns = 1
    next
  }
  /^file=/ {
    rows += 1
    if (NF != 6 || $1 !~ /^file=src\/emel\/kernel\/.+\.hpp$/ ||
        $2 !~ /^class=(core|aarch64|x86_64)$/ ||
        $3 !~ /^blob=[0-9a-f]{40}$/ || $4 !~ /^line_count=[1-9][0-9]*$/ ||
        $5 !~ /^rust_counterpart=(crates\/emel-kernels\/src\/.+\.rs|-)$/ ||
        $6 !~ /^rust_status=(missing|partial|scaffold|present)$/) {
      invalid = 1
    }
    next
  }
  /^result=valid$/ { result = 1; next }
  /^source_file_count=22$/ { count = 1; next }
  END {
    exit !(header && columns && count && rows == 22 && result && !invalid)
  }
' "$OUTPUT" || fail "generated kernel source inventory failed schema validation: $OUTPUT"

printf 'Kernel source inventory validated: %s\n' "$OUTPUT"
