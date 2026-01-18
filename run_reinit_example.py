
import torch
import sys
import os

# Modify path to include current directory so we can import reinit_vocab_vectors
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinit_vocab_vectors import reinitialize_vocab_embedding

# Mock classes to simulate Megatron structure
class MockEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.word_embeddings.vocab_start_index = 0
        self.word_embeddings.vocab_end_index = vocab_size

class MockGPTModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = MockEmbedding(vocab_size, hidden_size)
        # Shared weights logic
        self.output_layer = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_layer.weight = self.embedding.word_embeddings.weight

def test_reinit():
    # Setup mocks
    vocab_size = 100
    hidden_size = 16
    target_vocab_id = 42

    # Mock distributed environment (single process for demo)
    # We need to monkeypath parallel_state for the script to run without actual mpirun
    from megatron.core import parallel_state
    
    # Mocking parallel_state methods to return single-process values
    parallel_state.get_tensor_model_parallel_group = lambda: None
    parallel_state.get_tensor_model_parallel_rank = lambda: 0
    parallel_state.get_tensor_model_parallel_world_size = lambda: 1
    
    # Mock torch.distributed for the single process test
    if not torch.distributed.is_initialized():
        # A dummy init just to satisfy some checks if needed, 
        # but our script avoids actual dist calls by mocking logic if we were strict.
        # However, our script USES torch.distributed.all_reduce. 
        # So we must fake it or initialize it.
        # Simplest is to init a trivial process group.
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)

    print("Creating mock model...")
    model = MockGPTModel(vocab_size, hidden_size)
    
    # Intentionally set the target vector to something distinct (e.g., all zeros) to show change
    with torch.no_grad():
        model.embedding.word_embeddings.weight[target_vocab_id].fill_(0.0)
    
    print(f"\nBefore reinit, vector at {target_vocab_id} (first 5 elements):")
    print(model.embedding.word_embeddings.weight[target_vocab_id][:5])
    
    # Calculate initial stats manually for verification
    all_weights = model.embedding.word_embeddings.weight
    # Exclude the zeroed one for fair "global" stats expectation? 
    # The script uses ALL weights including the target.
    global_mean = all_weights.mean()
    global_std = all_weights.std()
    print(f"Global stats: Mean={global_mean:.4f}, Std={global_std:.4f}")

    print(f"\nrunning reinitialize_vocab_embedding for vocab_id={target_vocab_id}...")
    reinitialize_vocab_embedding(model, target_vocab_id)
    
    print(f"\nAfter reinit, vector at {target_vocab_id} (first 5 elements):")
    print(model.embedding.word_embeddings.weight[target_vocab_id][:5])
    
    # Verify it's not zero anymore
    if not torch.allclose(model.embedding.word_embeddings.weight[target_vocab_id], torch.zeros(hidden_size)):
        print("\nSUCCESS: Vector was updated!")
    else:
        print("\nFAILURE: Vector is still zero.")

if __name__ == "__main__":
    test_reinit()
