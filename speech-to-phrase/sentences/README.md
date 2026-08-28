# Bundled sentence overrides

Files in this directory patch sentence templates from the installed
`home-assistant-intents` package. Use the same layout and YAML format as the
upstream `sentences` directory:

```text
sentences/<language>/<Intent>/<slot_combination>.yaml
```

For example, `sentences/en/HassMedia/default.yaml` overrides
`en/HassMedia.default`.

Only `data[].sentences` is replaced. Slot, context, response, and other metadata
continues to come from the installed package. When a file contains
`speech_to_phrase: true` blocks, only those blocks are loaded; otherwise all
blocks are loaded.

Top-level and `data[]`-level `expansion_rules` are supported. Data-block rules
override top-level rules with the same name, and references to either are
recursively substituted into that block's sentences while loading. References
not declared in the override remain available for resolution from the installed
package.

Overrides are read once when the app process starts.
