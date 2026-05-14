#!/bin/bash
# Install the transformers-5.x → vLLM 0.11 compat shim into the active venv.
#
# transformers 5.x dropped PreTrainedTokenizerBase.all_special_tokens_extended;
# vLLM 0.11.0's tokenizer init (in both the parent process and the spawn'd
# EngineCore_DP0 subprocess) reads that attribute. We install a .pth file
# that imports a shim at every Python startup so both processes see the
# restored attribute.
#
# Re-run after any venv rebuild.
set -euo pipefail

PY="${PYTHON:-python3}"
SITE=$($PY -c "import site; sp = site.getsitepackages(); print(sp[0])")
HERE="$(cd "$(dirname "$0")"/../src/lora && pwd)"

cp "$HERE/_vllm_compat.py" "$SITE/vllm_compat_shim.py"
printf 'import vllm_compat_shim\n' > "$SITE/vllm_compat.pth"

echo "installed shim:"
echo "  $SITE/vllm_compat_shim.py"
echo "  $SITE/vllm_compat.pth"

$PY -c "from transformers import PreTrainedTokenizerBase as P; assert hasattr(P, 'all_special_tokens_extended'); print('verified: PreTrainedTokenizerBase.all_special_tokens_extended is restored')"
