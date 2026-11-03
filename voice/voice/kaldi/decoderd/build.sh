#!/usr/bin/env bash
# Build verify_decoderd against a rhasspy-speech style Kaldi dev tree.
#   ./build.sh /path/to/tools-dir [out-path]
# where tools-dir contains kaldi/{include,lib} and openfst/{include,lib}
# (kaldi include may be nested as include/kaldi and include/openfst).
set -euo pipefail

TOOLS_DIR="${1:?usage: build.sh <tools-dir> [out]}"
OUT="${2:-$(dirname "$0")/verify_decoderd}"
SRC="$(dirname "$0")/verify_decoderd.cc"

KALDI="$TOOLS_DIR/kaldi"
OPENFST="$TOOLS_DIR/openfst"

# The kaldi-exported OpenFST headers predate unique_ptr-correct assignment
# (fst.h FstImpl::operator= assigns raw pointers into unique_ptr members) and
# no longer compile under modern g++. The libs match those headers, so patch
# a SHADOW COPY of just fst/fst.h (pure compile fix, no layout change) and
# put it first on the include path.
SHADOW="$(dirname "$OUT")/.fst_shadow_include"
if [ -f "$KALDI/include/openfst/fst/fst.h" ]; then
  mkdir -p "$SHADOW/fst"
  sed -E 's/^(\s*)([io]symbols_) = (impl\.[io]symbols_ \? impl\.[io]symbols_->Copy\(\) : nullptr);/\1\2.reset(\3);/' \
    "$KALDI/include/openfst/fst/fst.h" > "$SHADOW/fst/fst.h"
fi

INCLUDES=("-I$SHADOW")
for d in "$KALDI/include/kaldi" "$KALDI/include/openfst" "$KALDI/include" \
         "$OPENFST/include"; do
  [ -d "$d" ] && INCLUDES+=("-I$d")
done

g++ -O2 -std=c++17 -DHAVE_OPENBLAS=1 \
  "${INCLUDES[@]}" \
  "$SRC" \
  -L"$KALDI/lib" -L"$OPENFST/lib" \
  -lkaldi-online2 -lkaldi-ivector -lkaldi-nnet3 -lkaldi-chain \
  -lkaldi-nnet2 -lkaldi-cudamatrix -lkaldi-decoder -lkaldi-lat \
  -lkaldi-fstext -lkaldi-hmm -lkaldi-feat -lkaldi-transform \
  -lkaldi-gmm -lkaldi-tree -lkaldi-util -lkaldi-matrix -lkaldi-base \
  -lfst \
  -Wl,-rpath,"$KALDI/lib" -Wl,-rpath,"$OPENFST/lib" \
  -o "$OUT"

echo "built: $OUT"
