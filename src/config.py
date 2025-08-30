# Hyperparameters

vocab_size = 11000  # Number of tokens in vocabulary post-BPE
seq_len = 256  # Number of tokens in a single training sequence (can be increased to 512)
batch_size = 8  # Number of training examples per batch (should be within range 8-16)
d_model = 256  # Vector dimension for token embeddings
feedforward_dim = 1024  # Matrix dimension for feedforward layers
n_layers = 6  # Number of transformer blocks (should be within range 4-8)
n_heads = 4  # Number of attention heads (can be increased to 8)
# d_k = d_model / n_heads (?)

# Attention mechanism: Masked multi-head self-attention
# Loss function: Cross-entropy
# Positional encodings: Sinusoidal

# TODO: Learn about optimizer and learning rates - Adam optimizer, 1e-4 learning rate, warmup, cosine annealing schedule, linear warmup, cosine decay ?
# TODO: Learn CUDA and implement GPU acceleration ?