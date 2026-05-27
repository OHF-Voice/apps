#!/usr/bin/env sh

cd "$(dirname "$0")"

    # --hass-api 'http://localhost:8123/api' \
    # --hass-token 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4ZThlZWE1NDQ4ZDY0NGJjYjIzZDJlZmVkNjZmZDAyMyIsImlhdCI6MTY5NTMyMjk5MywiZXhwIjoyMDEwNjgyOTkzfQ.t9C8P1HT4xQleyXv8-SQbM_hkZMiIt8HTx0MA6wzIvY' \

python3 src/app.py \
    --uri 'tcp://127.0.0.1:10500' \
    --hass-api 'http://homeassistant.local:8123/api' \
    --hass-token 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1N2U3YjlkY2U5MTY0YjhjOGQ5OGQ2NjVjZWYzYjVmYyIsImlhdCI6MTcyNzU2NTY5MCwiZXhwIjoyMDQyOTI1NjkwfQ.vGExzcuHUlvZ66ufZDkWxKictuXVfwaxHVHMb4tvTHY' \
    --tools 'tools.yaml' \
    --llama-state 'local/llama_state.bin' \
    --include-names-in-tools \
    --debug "$@"
