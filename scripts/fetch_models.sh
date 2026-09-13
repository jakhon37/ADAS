#!/usr/bin/env bash
# Download every ADAS ONNX listed in models/MANIFEST.json and verify its
# pinned SHA-256. Nothing is installed; no Python packages are touched.
#
# The manifest is the single source of truth for url / sha256 / filename, so
# this script never carries its own copy of a hash.
#
# Usage:
#   bash scripts/fetch_models.sh              # fetch everything that is missing
#   bash scripts/fetch_models.sh yolop midas_v21_small
#   FORCE=1 bash scripts/fetch_models.sh      # re-download even if present
#
# Failure behaviour: a hash mismatch deletes the partial file and aborts with
# exit 1 - a model whose bytes do not match the pin is never left on disk where
# scripts/build_engines.py could pick it up. A network failure aborts the same
# way. Already-correct files are left untouched and reported as "ok (cached)".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/models"
MANIFEST="${DEST}/MANIFEST.json"
FORCE="${FORCE:-0}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing ${MANIFEST}" >&2
  exit 1
fi
mkdir -p "${DEST}"

# Emit "name<TAB>file<TAB>sha256<TAB>url<TAB>archive_member" for the requested
# models (or all of them). archive_member is "-" for a direct download.
manifest_rows() {
  python3 - "$MANIFEST" "$@" <<'PY'
import json, sys
manifest_path = sys.argv[1]
wanted = sys.argv[2:]
with open(manifest_path) as fh:
    models = json.load(fh)["models"]
if wanted:
    missing = [w for w in wanted if w not in models]
    if missing:
        sys.stderr.write("unknown model(s): %s\nknown: %s\n"
                         % (", ".join(missing), ", ".join(models)))
        raise SystemExit(2)
    names = wanted
else:
    names = list(models)
for name in names:
    m = models[name]
    print("\t".join([
        name,
        m["onnx_file"],
        m.get("onnx_sha256") or "-",
        m["url"],
        m.get("archive_member") or "-",
    ]))
PY
}

check_sha() {   # check_sha <file> <expected>  -> 0 match, 1 mismatch
  local got
  got="$(sha256sum "$1" | awk '{print $1}')"
  [[ "${got}" == "$2" ]]
}

report_sha() { sha256sum "$1" | awk '{print $1}'; }

fetch_direct() {  # fetch_direct <url> <out>
  local url="$1" out="$2"
  echo "  GET ${url}"
  curl -L --fail --retry 3 --retry-delay 5 -# -o "${out}.part" "${url}"
  mv "${out}.part" "${out}"
}

fetch_from_archive() {  # fetch_from_archive <url> <member> <out>
  # PINTO_model_zoo ships one 2.9 GB resources.tar.gz per model entry with no
  # per-file endpoint, so we stream it through tar and keep only the member we
  # need. Nothing but <member> ever lands on disk. Took ~8 min at ~4 MB/s from
  # this board to the Wasabi ap-northeast-2 endpoint.
  local url="$1" member="$2" out="$3"
  local work
  work="$(mktemp -d "${DEST}/.fetch.XXXXXX")"
  echo "  STREAM ${url}"
  echo "  extracting only: ${member}  (multi-GB archive, expect several minutes)"
  if ! curl -L --fail --retry 3 --retry-delay 5 "${url}" \
       | tar -xzf - -C "${work}" --wildcards "${member}"; then
    rm -rf "${work}"
    echo "  archive stream failed" >&2
    return 1
  fi
  local found
  found="$(find "${work}" -name "$(basename "${member}")" -type f | head -n1)"
  if [[ -z "${found}" ]]; then
    rm -rf "${work}"
    echo "  member ${member} not found in archive" >&2
    return 1
  fi
  mv "${found}" "${out}"
  rm -rf "${work}"
}

# Resolve the work list up front. A process substitution would hide a non-zero
# exit from manifest_rows (an unknown model name), so the rows go through a
# temp file whose producer status we can actually check.
ROWS="$(mktemp "${TMPDIR:-/tmp}/adas-fetch-rows.XXXXXX")"
cleanup() { rm -f "${ROWS}"; }
trap cleanup EXIT

if ! manifest_rows "$@" > "${ROWS}"; then
  echo "could not resolve the model list from ${MANIFEST}" >&2
  exit 2
fi
if [[ ! -s "${ROWS}" ]]; then
  echo "no models selected" >&2
  exit 2
fi

rc=0
while IFS=$'\t' read -r name file sha url member; do
  [[ -z "${name}" ]] && continue
  out="${DEST}/${file}"
  echo "== ${name} -> models/${file}"

  if [[ -f "${out}" && "${FORCE}" != "1" ]]; then
    if [[ "${sha}" == "-" ]]; then
      echo "  ok (cached, no pin recorded) $(report_sha "${out}")"
      continue
    fi
    if check_sha "${out}" "${sha}"; then
      echo "  ok (cached) ${sha}"
      continue
    fi
    echo "  cached file does not match the pin; re-downloading" >&2
    rm -f "${out}"
  fi

  if [[ "${member}" == "-" ]]; then
    if ! fetch_direct "${url}" "${out}"; then
      echo "  DOWNLOAD FAILED ${name}" >&2
      rm -f "${out}.part"
      rc=1
      continue
    fi
  else
    if ! fetch_from_archive "${url}" "${member}" "${out}"; then
      echo "  DOWNLOAD FAILED ${name}" >&2
      rc=1
      continue
    fi
  fi

  if [[ "${sha}" == "-" ]]; then
    echo "  WARNING: no sha256 pinned for ${name}; got $(report_sha "${out}")" >&2
    continue
  fi
  if ! check_sha "${out}" "${sha}"; then
    echo "  HASH MISMATCH ${file}: expected ${sha} got $(report_sha "${out}")" >&2
    rm -f "${out}"
    rc=1
    continue
  fi
  echo "  ok ${sha}"
done < "${ROWS}"

if [[ "${rc}" -ne 0 ]]; then
  echo "one or more models failed to fetch" >&2
  exit "${rc}"
fi

echo
echo "ONNX under ${DEST} (see models/MANIFEST.json)"
echo "next: python3 scripts/build_engines.py"
