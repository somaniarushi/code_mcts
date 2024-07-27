from __future__ import annotations

from data_processing.clean_jsonl import clean_jsonl
from run_eval import run_eval

run_eval(
    outpath='outputs',
    n_samples=64,
)

# clean_jsonl(
#     "outputs/samples_llama_pretrain_10.jsonl",
#     "outputs/cleaned_samples_llama_pretrain_10.jsonl"
# )
