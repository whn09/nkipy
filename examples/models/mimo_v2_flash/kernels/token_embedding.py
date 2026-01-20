"""Token embedding kernel for MiMo-V2-Flash"""

import numpy as np


def token_embedding(tok_embedding, token_ids):
    """
    Token embedding lookup.

    Args:
        tok_embedding: Embedding table [vocab_size, hidden_size]
        token_ids: Token indices [batch_size, seq_len]

    Returns:
        Embedded tokens [batch_size, seq_len, hidden_size]
    """
    # Flatten token IDs for indexing
    ids_1d = token_ids.reshape(-1)

    # Lookup embeddings
    hidden = tok_embedding[ids_1d, :]

    # Reshape back to [batch_size, seq_len, hidden_size]
    hidden = hidden.reshape(*token_ids.shape, -1)

    return hidden
