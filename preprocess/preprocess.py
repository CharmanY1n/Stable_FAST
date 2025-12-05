from scipy.fft import dct
import numpy as np
from tqdm import tqdm
import pickle
def cal_information(chunks, scale=10,ndim=3):
    """
    Calculate DCT non-zero information for chunks.
    
    Args:
        chunks: Input chunks np.ndarray
        scale: Only used for calculating DCT_NON_ZERO, not for tokenizer
        ndim: Manually set, use ndim=3 for chunks, ndim=2 for single chunk
    
    Returns:
        DCT_NON_ZERO np.ndarray
    """
    info_list = []
    if ndim == 3:
        for chunk in chunks:
            dct_coeff = dct(chunk, axis=0, norm='ortho') 
            dct_coeff = np.around(dct_coeff * scale)
            non_zero = np.count_nonzero(dct_coeff)
            info_list.append(non_zero)
    elif ndim == 2:
        dct_coeff = dct(chunks, axis=0, norm='ortho')  
        dct_coeff = np.around(dct_coeff * scale)
        non_zero = np.count_nonzero(dct_coeff)
        info_list.append(non_zero)
        
    return np.array(info_list)




def adaptive_chunking_rev4(chunks, scale=10):
    """
    Re-chunks based on information content - Revision 4

    Args:
        chunks (np.ndarray): Original chunks, typically from a fixed-frequency split. 
                             Shape: [num_episodes, episode_len, action_dim].
        scale (int): The DCT scaling factor γ.

    Returns:
        list[np.ndarray]: A list of NumPy arrays, where each array is a chunk of
                          variable size.
    """
    
    result_chunks = []
    chunk_sizes_per_episode = []
    initial_size = 10
    
    mean_info = 11.4242 # target mean information for chunks of size 10 with scale 10
    print(f"Using hardcoded mean_info: {mean_info}")
    
    total_steps = sum(len(episode.reshape(-1, episode.shape[-1])) for episode in chunks)

    with tqdm(total=total_steps, desc="Overall Progress") as pbar:
        for episode_idx, episode in enumerate(chunks):
            episode_flatten = episode.reshape(-1, episode.shape[-1])
            episode_len = len(episode_flatten)
            
            if episode_len == 0:
                continue

            ptr = 0
            max_size_so_far = initial_size
            episode_chunk_sizes = []
            while ptr < episode_len:
                
                def get_chunk(size):
                    end = ptr + size
                    chunk = episode_flatten[ptr:end]
                    
                    if len(chunk) < size:
                        padding_needed = size - len(chunk)
                        last_action = chunk[-1] if len(chunk) > 0 else np.zeros(episode_flatten.shape[1])
                        padding = np.tile(last_action, (padding_needed, 1))
                        chunk = np.concatenate([chunk, padding], axis=0)
                    return chunk

                best_size = initial_size
                
                current_chunk = get_chunk(initial_size)
                current_info = sum(cal_information(current_chunk, scale=scale, ndim=2))
                best_diff = abs(current_info - mean_info)
                
                if current_info > mean_info:
                    for test_size in range(initial_size - 1, 0, -1):
                        test_chunk = get_chunk(test_size)
                        test_info = sum(cal_information(test_chunk, scale=scale, ndim=2))
                        test_diff = abs(test_info - mean_info)
                        
                        if test_diff < best_diff:
                            best_diff = test_diff
                            best_size = test_size
                        
                        if test_info < mean_info:
                            break
                else: 
                    
                    test_size = initial_size + 1
                    while True:  
                        test_chunk = get_chunk(test_size)
                        test_info = sum(cal_information(test_chunk, scale=scale, ndim=2))
                        test_diff = abs(test_info - mean_info)

                        if test_diff < best_diff:
                            best_diff = test_diff
                            best_size = test_size
                        
                        # Stop if information exceeds mean or if we reach the end of episode
                        if test_info > mean_info or (ptr + test_size) >= episode_len:
                            break
                            
                        test_size += 1

                final_chunk = get_chunk(best_size)
                result_chunks.append(final_chunk)
                episode_chunk_sizes.append(best_size)

                max_size_so_far = max(max_size_so_far, best_size)
                
                ptr += 1
                pbar.update(1)
            chunk_sizes_per_episode.append(np.array(episode_chunk_sizes))

    return result_chunks, chunk_sizes_per_episode




import os
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

def rechunk_and_save(parquet_dir, output_dir, chunk_sizes_per_episode):
    """
    Re-partition the parquet file according to chunk_sizes_per_episode and save as episode-specific folders.

    Args:
        parquet_dir (str): Directory containing parquet files
        output_dir (str): Output directory
        chunk_sizes_per_episode (list[np.ndarray]): Chunk size sequence for each episode
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all parquet files
    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
    assert len(parquet_files) == len(chunk_sizes_per_episode), \
        f"chunk_sizes_per_episode length ({len(chunk_sizes_per_episode)}) != parquet file count ({len(parquet_files)})"

    for epi_idx, (parquet_file, chunk_sizes) in enumerate(
        tqdm(zip(parquet_files, chunk_sizes_per_episode), 
             total=len(parquet_files), desc="Processing Episodes")
    ):
        file_path = os.path.join(parquet_dir, parquet_file)

        # Read parquet file (convert to pandas DataFrame)
        df = pq.read_table(file_path).to_pandas()
        episode_len = len(df)

        # Create a separate folder for each episode
        episode_dir = os.path.join(output_dir, f"episode_{epi_idx:06d}")
        os.makedirs(episode_dir, exist_ok=True)

        ptr = 0
        frame_idx = 0
        for csize in chunk_sizes:
            if ptr + csize > episode_len:
                break  # Avoid out-of-bounds

            # Save all fields for each chunk
            chunk_dict = {}
            for col in df.columns:
                values = df[col].to_numpy()

                if isinstance(values[0], (np.ndarray, list)):
                    values = np.stack(values)

                chunk_dict[col] = values[ptr:ptr+csize]

            # Save npz file
            save_path = os.path.join(
                episode_dir,
                f"episode_{epi_idx:06d}_frame_{frame_idx:06d}.npz"
            )
            np.savez_compressed(save_path, **chunk_dict)

            frame_idx += 1
            ptr += 1  # Step size is 1



