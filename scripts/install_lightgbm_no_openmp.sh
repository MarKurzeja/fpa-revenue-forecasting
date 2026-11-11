#!/bin/sh
set -eu

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This installer is for macOS only." >&2
  exit 1
fi

python -m pip install \
  --force-reinstall \
  --no-deps \
  --no-binary=lightgbm \
  --config-settings=cmake.define.USE_OPENMP=OFF \
  "lightgbm==4.5.0"

python - <<'PY'
import subprocess

from lightgbm.libpath import find_lib_path

library = find_lib_path()[0]
dependencies = subprocess.run(
    ["otool", "-L", library], check=True, capture_output=True, text=True
).stdout
if "libomp.dylib" in dependencies:
    raise SystemExit(f"LightGBM still links libomp.dylib: {library}")
print(f"LightGBM built without libomp.dylib: {library}")
PY
