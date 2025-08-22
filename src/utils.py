import re

# Process raw data corpus
def process_data(input_file = "./data/hemingway.txt", output_file = "./data/hemingway_processed.txt"):
    with open(input_file, "r", encoding="utf-8") as f_in:
        text = f_in.read()

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(text)

# BPE tokenizer