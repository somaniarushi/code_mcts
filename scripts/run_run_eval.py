from run_eval import run_eval
from data_processing.clean_jsonl import clean_jsonl

# run_eval(
#     outpath="outputs",
#     n_samples=1
# )

clean_jsonl(
    "outputs/samples_llama_1.jsonl",
    "outputs/cleaned_samples_llama_1.jsonl"
)