import numpy as np

# Sampling
def binary_sampler(p, rows, cols):
    return np.random.binomial(1, p, size=(rows, cols)).astype(np.float32)

def sample_batch_index(total, batch_size):
    return np.random.permutation(total)[:batch_size]
    
