import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.utils import VocabUtility

def get_distributed_mean_std(tensor):
    """
    Calculates the mean and standard deviation of a tensor distributed across 
    the tensor model parallel group.
    """
    # Calculate local stats
    local_tensor = tensor.float()
    local_sum = local_tensor.sum()
    local_sum_sq = local_tensor.pow(2).sum()
    local_count = torch.tensor(local_tensor.numel(), device=tensor.device, dtype=torch.float)

    # All-reduce to get global stats
    group = parallel_state.get_tensor_model_parallel_group()
    if group is not None:
        torch.distributed.all_reduce(local_sum, group=group)
        torch.distributed.all_reduce(local_sum_sq, group=group)
        torch.distributed.all_reduce(local_count, group=group)

    # Calculate global mean and std
    mean = local_sum / local_count
    std = torch.sqrt(local_sum_sq / local_count - mean.pow(2))
    
    return mean, std

def reinitialize_vocab_embedding(model, vocab_id):
    """
    Re-initializes the embedding vector for a specific vocab_id across 
    word_embeddings and output_layer (if separate).
    
    Args:
        model: The Megatron-LM model (or list of model chunks).
        vocab_id: The vocab ID to re-initialize.
    """
    if isinstance(model, list):
        models = model
    else:
        models = [model]

    for model_chunk in models:
        # Access global vocab size if available
        vocab_size = getattr(model_chunk, 'vocab_size', None)
        
        # Check if weights are shared
        share_embeddings_and_output_weights = getattr(model_chunk, 'share_embeddings_and_output_weights', True)
        # If attribute is missing, default to True (safer for consistency in ambiguous cases, or False? 
        # Usually Megatron models have this. If not, assuming True ensures sync in PP which is critical.
        
        # 1. Word Embeddings
        word_embeddings = None
        if hasattr(model_chunk, 'embedding') and hasattr(model_chunk.embedding, 'word_embeddings'):
            word_embeddings = model_chunk.embedding.word_embeddings
        elif hasattr(model_chunk, 'language_model') and hasattr(model_chunk.language_model, 'embedding'):
             word_embeddings = model_chunk.language_model.embedding.word_embeddings

        if word_embeddings is not None:
            _reinit_layer_vector(word_embeddings.weight, vocab_id, vocab_size, 
                                 layer_type='word_embeddings', 
                                 vocab_start_index=getattr(word_embeddings, 'vocab_start_index', None),
                                 vocab_end_index=getattr(word_embeddings, 'vocab_end_index', None),
                                 seed_salt=0)

        # 2. Output Layer
        output_layer = None
        if hasattr(model_chunk, 'output_layer'):
            output_layer = model_chunk.output_layer
        elif hasattr(model_chunk, 'language_model') and hasattr(model_chunk.language_model, 'output_layer'):
             output_layer = model_chunk.language_model.output_layer

        if output_layer is not None and hasattr(output_layer, 'weight'):
            # Double check identity just in case (for local sharing)
            is_shared_memory = (word_embeddings is not None and output_layer.weight is word_embeddings.weight)
            
            # Reinit if:
            # a) It's not shared memory (could be Untied, or Tied-but-PP)
            # b) If it IS shared memory, we already did it in step 1? 
            #    Wait, if is_shared_memory is True, step 1 updated 'word_embeddings.weight' which IS 'output_layer.weight'.
            #    So we don't need to do anything.
            
            if not is_shared_memory:
                 # Determine salt. 
                 # If shared=True (but PP), we want SAME seed as embedding -> salt=0
                 # If shared=False (Untied), we want DIFFERENT seed -> salt=1
                 
                 salt = 0 if share_embeddings_and_output_weights else 1
                 
                 _reinit_layer_vector(output_layer.weight, vocab_id, vocab_size, 
                                      layer_type='output_layer',
                                      seed_salt=salt)

def _reinit_layer_vector(weight, vocab_id, global_vocab_size, layer_type, vocab_start_index=None, vocab_end_index=None, seed_salt=0):
    # Calculate stats
    mean, std = get_distributed_mean_std(weight)
    
    # Determine the vocab range for this rank
    if vocab_start_index is None or vocab_end_index is None:
        rank = parallel_state.get_tensor_model_parallel_rank()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        
        if global_vocab_size is not None:
             vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                global_vocab_size, rank, world_size
            )
        else:
             # Fallback if vocab size is missing (less reliable if padded)
             local_size = weight.shape[0]
             vocab_start_index = rank * local_size
             vocab_end_index = vocab_start_index + local_size

    # Check if vocab_id is in this rank's partition
    if vocab_start_index <= vocab_id < vocab_end_index:
        local_idx = vocab_id - vocab_start_index
        
        # We need to ensure that if this operation runs on different pipeline stages (Rank 0 and Rank N)
        # for shared weights, they generate the EXACT same random values.
        # We achieve this by using a Generator seeded with the vocab_id.
        gen = torch.Generator(device=weight.device)
        # Add salt to seed to differentiable untied layers
        gen.manual_seed(vocab_id + seed_salt * 1000000) 
        
        with torch.no_grad():
            weight[local_idx].normal_(mean.item(), std.item(), generator=gen)
        
        # Optional: Print confirmation
        rank = parallel_state.get_tensor_model_parallel_rank()
        print(f"[Rank {rank}] Reinitialized {layer_type} for vocab_id {vocab_id} (local idx {local_idx}) with mean={mean.item():.6f}, std={std.item():.6f}")

def add_reinit_args(parser):
    """
    Adds re-initialization related arguments to the parser.
    Args:
        parser: The argparse parser object.
    """
    group = parser.add_argument_group(title='reinit-vocab-embeddings')
    group.add_argument('--reinit-vocab-ids', type=int, nargs='+', default=None,
                       help='List of vocab IDs to re-initialize with distributed statistics.')
    return parser

def reinit_from_args(model, args):
    """
    Helper function to perform re-initialization if arguments are present.
    Args:
        model: The model to operate on.
        args: The parsed arguments object.
    """
    if hasattr(args, 'reinit_vocab_ids') and args.reinit_vocab_ids is not None:
        vocab_ids = args.reinit_vocab_ids
        if len(vocab_ids) > 0:
            rank = parallel_state.get_tensor_model_parallel_rank()
            if rank == 0:
                print(f"Re-initializing embeddings for {len(vocab_ids)} vocab IDs specified in args.")
            
            for vocab_id in vocab_ids:
                reinitialize_vocab_embedding(model, vocab_id)

