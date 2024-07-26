import json
from typing import Any, Dict

def clean_completion(completion: str) -> str:
    # Get the code between ```python or ``` and ending at ```
    # Find the first occurence of ``` and last occurence of ```
    opening_backtick = completion.find("```")
    closing_backtick = completion.rfind("```")

    if opening_backtick == -1 or closing_backtick == -1:
        return completion
    else:
        code = completion[opening_backtick + 3:closing_backtick]

    # If the code has a section that looks like if __name__ == "__main__":, remove it and everything after it
    # main_start = code.find("if __name__ == '__main__':")
    # if main_start != -1:
    #     code = code[:main_start]
    # main_start = code.find('if __name__ == "__main__":')
    # if main_start != -1:
    #     code = code[:main_start]
    return code.strip()

def clean_data(data_point: Dict[str, Any]) -> Dict[str, Any]:
    completion = data_point["completion"]
    code = clean_completion(completion)
    data_point["completion"] = code
    return data_point

def clean_jsonl(raw_jsonl_path: str, cleaned_json_path: str) -> None:
    # Load the raw JSONL file
    with open(raw_jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Clean the data
            cleaned_data = clean_data(data)
            # Save the cleaned data to a new JSONL file
            with open(cleaned_json_path, "a") as f:
                json.dump(cleaned_data, f)
                f.write("\n")