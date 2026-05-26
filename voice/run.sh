#!/usr/bin/env sh

cd "$(dirname "$0")"

python3 src/app.py \
    --uri 'tcp://127.0.0.1:10500' \
    --hass-api 'http://localhost:8123/api' \
    --hass-token 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4ZThlZWE1NDQ4ZDY0NGJjYjIzZDJlZmVkNjZmZDAyMyIsImlhdCI6MTY5NTMyMjk5MywiZXhwIjoyMDEwNjgyOTkzfQ.t9C8P1HT4xQleyXv8-SQbM_hkZMiIt8HTx0MA6wzIvY' \
    --tools 'tools.yaml' \
    --llama-state 'local/llama_state.bin' \
    --default-area-id 'office' \
    --debug
