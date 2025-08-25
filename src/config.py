# Hyperparameters

# What's the optimal vocabulary size?
vocab_size = 11000  # Number of tokens in vocabulary post-BPE
context_size = 512  # Number of tokens in a single training sequence (can be reduced to 256)
batch_size = 8  # Number of training examples per batch (should be within range 8-16)
embedding_dim = 256  # Vector dimension for token embeddings (d_model)
feedforward_dim = 1024  # Matrix dimension for feedforward layers
n_layers = 6  # Number of transformer blocks (should be within range 4-8)
n_heads = 8  # Number of attention heads (can be reduced to 4)

# Loss function: Cross-entropy
# Positional encodings: Sinusoidal

# TODO: Learn about optimizer and learning rates - Adam optimizer, 1e-4 learning rate, warmup, cosine annealing schedule, linear warmup, cosine decay ?