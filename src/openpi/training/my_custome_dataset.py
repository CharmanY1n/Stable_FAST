# import it in data_loader.py
# npz files are in your directory

import os
import numpy as np
import torch 

class MyNPZDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for loading pre-chunked data directly from .npz files.
    """
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        
        self.file_paths = []
        print(f"Scanning for all .npz files in directory {self.root_dir}...")
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.endswith(".npz"):
                    self.file_paths.append(os.path.join(root, f))
        
        self.file_paths.sort() 
        if not self.file_paths:
            raise FileNotFoundError(f"No .npz files found in directory {self.root_dir}.")
        print(f"Scanning completed, found {len(self.file_paths)} sample files.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.file_paths[idx]
        
        try:
            npz_data = np.load(file_path)
            sample = {key: npz_data[key] for key in npz_data.keys()}
            
            if 'Action_is_pad' in sample:
                 sample['action_is_pad'] = sample.pop('Action_is_pad')

            return sample
        except Exception as e:
            print(f"Error: Failed to load or process file {file_path}: {e}")
            return {}