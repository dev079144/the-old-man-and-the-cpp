import json
import regex as re
import numpy as np

# Needed for alternative processing approach
# import re
# import unicodedata

# Process raw data corpus
def process_data(input_path = "./data/hemingway.txt", output_path = "./data/hemingway_processed.txt"):
    with open(input_path, "r", encoding="utf-8") as f_in:
        data = f_in.read()

    # ?
    # data = unicodedata.normalize("NFKC", data)
    # data = ''.join(c for c in data if not unicodedata.category(c).startswith('C'))

    data = re.sub(r'\s+', ' ', data)
    data = data.strip()

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(data)

# BPE Tokenizer
def tokenize(dataset, vocab_size, vocabulary_output_path='../data/vocabulary.json', corpus_output_path='../data/tokenized_corpus_ids.npy'):
    # Tokenize

    # Process dataset
    vocab = sorted(list(set(dataset)))

    processing_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    processed_tokens = re.findall(processing_pattern, dataset)

    tokens = [list(pt) for pt in processed_tokens]

    # Alterative processing method
    # raw_tokens = re.findall(r'\w+|[^\w\s]', dataset)

    # processed_tokens = [raw_tokens[0]]

    # for raw_token in raw_tokens[1:]:
    #     if raw_token.isalnum():
    #         processed_tokens.append('Ġ' + raw_token)
    #     else:
    #         processed_tokens.append(raw_token)

    # tokens = [list(pt) for pt in processed_tokens]

    # TODO: Verify further that processing is working as intended
    # print(tokens[:1000])

    new_tokens = []

    while len(vocab) < vocab_size:
        # Rank pairs
        pairs = {}

        for token in tokens:
            for pair in zip(token, token[1:]):  # Possible edge case at last character?
                pairs[pair] = pairs.get(pair, 0) + 1
        
        ranked_pairs = sorted(pairs.items(), key=lambda item: item[1], reverse=True)

        # Merge
        freq_thresh = 10
    
        new_tokens = list(tokens)
        for ranked_pair in ranked_pairs:
            if ranked_pair[1] > freq_thresh and len(vocab) < vocab_size:
                pair = list(ranked_pair[0])
                merged_token = ''.join(pair)
                for idx, token in enumerate(tokens):
                    occs = [i for i in range(len(token) - len(pair) + 1) if token[i:i+len(pair)] == pair]
                    for occ in occs:
                        new_tokens[idx][occ:occ+len(pair)] = [merged_token]
                
                vocab.append(merged_token)
            else:
                break
    
    # Save vocabulary and tokenized corpus ids
    token_to_id = {token: idx for idx, token in enumerate(vocab)}

    with open(vocabulary_output_path, 'w') as f:
        json.dump(token_to_id, f)

    tokenized_corpus_flattened = [token for sublist in new_tokens for token in sublist]
    tokenized_corpus_ids = [token_to_id[token] for token in tokenized_corpus_flattened]
    tokenized_corpus_ids = np.array(tokenized_corpus_ids, dtype=np.int16)

    np.save(corpus_output_path, tokenized_corpus_ids)

    return new_tokens, vocab

