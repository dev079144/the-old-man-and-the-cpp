# Hyperparameters

vocab_size = 8000  # Number of tokens in vocabulary post-BPE
seq_len = 128  # Number of tokens in a single training sequence (can be increased to 256)
batch_size = 16  # Number of training examples per batch (should be within range 8-16)
d_model = 128  # Vector dimension for token embeddings (can be increased to 256)
feedforward_dim = 512  # Matrix dimension for feedforward layers (can be increased to 1024)
n_layers = 3  # Number of transformer blocks (should be within range 2-8)
n_heads = 4  # Number of attention heads (can be increased to 8)
# d_k = d_model / n_heads (?)

# Attention mechanism: Masked multi-head self-attention
# Loss function: Cross-entropy
# Positional encodings: Sinusoidal

# TODO: Learn about optimizer and learning rates - Adam optimizer, 1e-4 learning rate, warmup, cosine annealing schedule, linear warmup, cosine decay ?
# TODO: Learn CUDA and implement GPU acceleration ?