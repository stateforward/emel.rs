#!/usr/bin/env bash
set -euo pipefail

# The kernel domain has many intentionally narrow observers because the
# reference exposes both generic and target-specialized operation families.
# Keep their registry explicit so adding a route cannot silently omit it from
# the aggregate gate.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=all

for argument in "$@"; do
  case "$argument" in
    --snapshot-only) MODE=snapshot ;;
    --live-only) MODE=live ;;
    --update-only) MODE=update ;;
    --update) MODE=all ;;
    --no-update) MODE=verify ;;
    --help|-h)
      cat <<'USAGE'
usage: scripts/kernel-parity-suite.sh [--snapshot-only|--live-only|--update-only]
                                      [--update|--no-update]

The default runs every registered live observer and refreshes its checked-in
snapshot only after the observer comparison succeeds. Snapshot-only verifies
the checked-in evidence without starting C++ observers. Live-only preserves
snapshots. Update-only reruns live observers and refreshes snapshots.
USAGE
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

CORE_SCRIPTS=(
  kernel-aarch64-silu-parity.sh
  kernel-aarch64-dup-f32-parity.sh
  kernel-aarch64-f16-matmul-parity.sh
  kernel-activation-parity.sh
  kernel-binary-parity.sh
  kernel-broadcast-parity.sh
  kernel-conv-transpose-1d-parity.sh
  kernel-elementwise-parity.sh
  kernel-f16-matmul-parity.sh
  kernel-flash-attn-parity.sh
  kernel-get-rows-parity.sh
  kernel-im2col-parity.sh
  kernel-matmul-parity.sh
  kernel-normalization-parity.sh
  kernel-portable-router-parity.sh
  kernel-power-parity.sh
  kernel-quant-more-parity.sh
  kernel-quantized-parity.sh
  kernel-rope-parity.sh
  kernel-sequence-parity.sh
  kernel-softmax-parity.sh
  kernel-unary-parity.sh
  argmax-parity.sh
)

TARGET_SCRIPTS=(
  "$ROOT_DIR"/scripts/kernel-target-aarch64-*-parity.sh
  "$ROOT_DIR"/scripts/kernel-target-get-rows-parity.sh
  "$ROOT_DIR"/scripts/kernel-target-get-rows-packed-parity.sh
  "$ROOT_DIR"/scripts/kernel-target-im2col-parity.sh
  "$ROOT_DIR"/scripts/kernel-target-normalization-parity.sh
  "$ROOT_DIR"/scripts/kernel-target-rope-parity.sh
  "$ROOT_DIR"/scripts/kernel-target-scalar-trig-parity.sh
  "$ROOT_DIR"/scripts/kernel-target-scalar-unary-parity.sh
  "$ROOT_DIR"/scripts/kernel-target-softmax-parity.sh
)

for relative in "${CORE_SCRIPTS[@]}"; do
  [[ -x "$ROOT_DIR/scripts/$relative" ]] || {
    printf 'error: registered kernel parity observer is missing: scripts/%s\n' "$relative" >&2
    exit 1
  }
done
for script in "${TARGET_SCRIPTS[@]}"; do
  [[ -x "$script" ]] || {
    printf 'error: registered target parity observer is missing: %s\n' "$script" >&2
    exit 1
  }
done

"$ROOT_DIR/scripts/kernel-source-inventory.sh" --validate >/dev/null

run_supported() {
  local script="$1"
  case "$MODE" in
    all) "$script" --update-only; "$script" --snapshot-only ;;
    snapshot) "$script" --snapshot-only ;;
    live) "$script" --live-only ;;
    update) "$script" --update-only ;;
    verify) "$script" --no-update ;;
  esac
}

run_special() {
  local script="$1"
  case "$MODE" in
    all)
      case "$script" in
        */argmax-parity.sh)
          "$script" --live --update
          "$script" --snapshot --no-live
          ;;
        */kernel-portable-router-parity.sh)
          "$script"
          "$script" --snapshot-only
          ;;
        */kernel-get-rows-parity.sh|*/kernel-matmul-parity.sh)
          "$script"
          "$script" --snapshot-only
          ;;
        */kernel-softmax-parity.sh)
          "$script" --update
          "$script" --snapshot-only
          ;;
        *)
          "$script" --update-only
          "$script" --snapshot-only
          ;;
      esac
      ;;
    snapshot) "$script" --snapshot-only ;;
    live) "$script" --live-only ;;
    update) "$script" --update-only ;;
    verify) "$script" --no-update ;;
  esac
}

for relative in "${CORE_SCRIPTS[@]}"; do
  case "$relative" in
    kernel-quant-more-parity.sh|kernel-portable-router-parity.sh|argmax-parity.sh|\
    kernel-get-rows-parity.sh|kernel-matmul-parity.sh|kernel-softmax-parity.sh)
      run_special "$ROOT_DIR/scripts/$relative" ;;
    *)
      run_supported "$ROOT_DIR/scripts/$relative" ;;
  esac
done

target_arch="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
if [[ "$target_arch" == aarch64 ]]; then
  for script in "${TARGET_SCRIPTS[@]}"; do
    run_special "$script"
  done
else
  printf 'kernel target parity: skipped native AArch64 observers on target_arch=%s; cross-target compile is checked below\n' "$target_arch"
fi

# The x86 implementation is compile-checked even when the host cannot execute
# its target-specific observers. The target event bridge is still required to
# accept every public portable event at compile time.
cargo check --locked --package emel-kernels --all-targets --target x86_64-apple-darwin \
  >/dev/null

printf 'Kernel parity suite passed: %s core observers, %s target observers, source inventory validated\n' \
  "${#CORE_SCRIPTS[@]}" "${#TARGET_SCRIPTS[@]}"
