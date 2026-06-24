#!/usr/bin/env bash
set -u -o pipefail

usage() {
  cat <<'EOF'
Reproduce the backend-vs-regime evidence with the deterministic harness.

Usage:
  scripts/reproduce_backend_regime_evidence.sh [options]

Options:
  --out-dir DIR          Output directory for this reproduction run.
                         Default: results/YYYY-MM-DD_backend-regime-repro
  --python PYTHON        Python executable. Default: python
  --dry-run              Validate specs and compute spec_hash only; do not launch servers.
  --include-crash-repro  Also run sglang-cutlass-fp8-patched.yaml, which is expected
                         to reproduce the fp8 + flashinfer_cutlass startup failure.
  --continue-on-failure  Continue after hard failures. Default: stop after hard failure.
  --no-summary           Do not print the final comparison summary.
  -h, --help             Show this help.

Expected environment:
  - Run from this repository on an H200 machine.
  - conda env from the specs exists (usually sglang-dev).
  - Models referenced by configs/moe_qwen3_30b*.yaml are present.
  - For sglang-cutlass-bf16-patched.yaml, use sglang main with PR #26496 or
    apply patches/sglang_cutlass_autotune_allowlist.diff to older sglang.

Harness exit codes:
  0 = ok, 1 = hard fail, 2 = quality gate failed, 3 = unreliable stddev.
  This script treats 0 and 3 as completed runs, because summary.json is still
  useful evidence when a regime is explicitly marked reliable:false.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="$repo_root/results/$(date +%Y-%m-%d)_backend-regime-repro"
python_bin="${PYTHON:-python}"
dry_run=0
include_crash_repro=0
continue_on_failure=0
print_summary=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --include-crash-repro)
      include_crash_repro=1
      shift
      ;;
    --continue-on-failure)
      continue_on_failure=1
      shift
      ;;
    --no-summary)
      print_summary=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

specs=(
  "bench-specs/sglang-triton-bf16-baseline.yaml"
  "bench-specs/sglang-cutlass-bf16-patched.yaml"
  "bench-specs/sglang-triton-fp8-baseline.yaml"
  "bench-specs/sglang-native-cutlass-fp8.yaml"
)

if [[ "$include_crash_repro" -eq 1 ]]; then
  specs+=("bench-specs/sglang-cutlass-fp8-patched.yaml")
fi

mkdir -p "$out_dir"
cd "$repo_root" || exit 1

echo "Repository: $repo_root"
echo "Output:     $out_dir"
echo "Python:     $python_bin"
echo

overall_status=0

for spec in "${specs[@]}"; do
  name="$(basename "$spec" .yaml)"
  spec_out="$out_dir/$name"
  mkdir -p "$spec_out"

  echo "==> Running $spec"
  cmd=("$python_bin" "harness/run_bench.py" "--spec" "$spec" "--out-dir" "$spec_out")
  if [[ "$dry_run" -eq 1 ]]; then
    cmd+=("--dry-run")
  fi

  "${cmd[@]}"
  status=$?

  case "$status" in
    0)
      echo "OK: $name"
      ;;
    3)
      echo "COMPLETED_WITH_UNRELIABLE_REGIME: $name (inspect summary.json warnings)"
      ;;
    *)
      echo "FAILED: $name exited with $status" >&2
      overall_status="$status"
      if [[ "$continue_on_failure" -eq 0 ]]; then
        break
      fi
      ;;
  esac
  echo
done

if [[ "$print_summary" -eq 1 && "$dry_run" -eq 0 ]]; then
  echo "==> Summary"
  "$python_bin" "scripts/summarize_backend_regime_evidence.py" "$out_dir" || overall_status=$?
fi

echo
echo "Artifacts written under: $out_dir"
exit "$overall_status"
