import re
# import unicodedata

# Process raw data corpus
def process_data(input_file = "./data/hemingway.txt", output_file = "./data/hemingway_processed.txt"):
    with open(input_file, "r", encoding="utf-8") as f_in:
        data = f_in.read()

    # ?
    # data = unicodedata.normalize("NFKC", data)
    # data = ''.join(c for c in data if not unicodedata.category(c).startswith('C'))

    data = re.sub(r'\s+', ' ', data)
    data = data.strip()

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(data)

# BPE tokenizer