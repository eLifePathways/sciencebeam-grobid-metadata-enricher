"""Loaded at Python startup via vllm_compat.pth.

transformers 5.x removed PreTrainedTokenizerBase.all_special_tokens_extended.
vLLM 0.11.0's tokenizer init (both in the parent LLM and in the spawn'd
EngineCore_DP0 subprocess) reads that attribute. Restore it as a property
that returns plain string special tokens — sufficient for vLLM's caching
path."""
try:
    from transformers import PreTrainedTokenizerBase as _PTB
    if not hasattr(_PTB, "all_special_tokens_extended"):
        _PTB.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
except Exception:
    pass
