from run_eval import run_eval
from data_processing.clean_jsonl import clean_jsonl

run_eval(
    outpath="outputs",
    n_samples=64,
)

# clean_jsonl(
#     "outputs/samples_llama_pretrain_10.jsonl",
#     "outputs/cleaned_samples_llama_pretrain_10.jsonl"
# )