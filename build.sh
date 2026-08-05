#!/usr/bin/env bash
set -e

NAME="agent"

echo "Starting Nuitka compilation..."

uv run nuitka \
    --standalone \
    --onefile \
    --onefile-no-compression \
    --include-data-dir=prompts=prompts \
    --include-data-dir=src/system_prompts=src/system_prompts \
    --include-data-files=config.toml=config.toml \
    --output-filename="$NAME" \
    main.py

echo "Build complete. Binary generated at ./${NAME}"
