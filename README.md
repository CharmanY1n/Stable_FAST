# README

## Environment Setup

This project uses a same environment configuration process as [openpi](https://github.com/Physical-Intelligence/openpi).

```bash
git clone --recurse-submodules git@github.com:your-username/your-repo.git #TODO 写入我们的仓库
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0}—FAST$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain FIVE steps:
1. Convert your data to a LeRobot dataset (which we use for training)
1. Compute the new chunk size for rechunking and fit the tokenizer
1. Convert the LeRobot dataset to NPZ file
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.\

### 2. Compute the new chunk size for rechunking and fit the tokenizer

This project includes adaptive chunking preprocessing for trajectory data. The main preprocessing functionality is in `preprocess.py`.

```python
processed_chunks, chunk_sizes = adaptive_chunking_rev4(
    chunks=trajectory_data, #suppose you have normalized the origin action data
    scale=10  # DCT scaling factor
)

# First, we download the tokenizer from the Hugging Face model hub
# Here, we will not use the pre-trained tokenizer weights, but only the source code
# to train a new tokenizer on our own data.
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


# Train the new tokenizer, depending on your dataset size this can take a few minutes
tokenizer = tokenizer.fit(processed_chunks)

# Save the new tokenizer, optionally push it to the Hugging Face model hub
tokenizer.save_pretrained("<your_local_path>")
tokenizer.push_to_hub("YourUsername/my_new_tokenizer")
```



### 3. Convert the LeRobot dataset to NPZ file

```python
parquet_dir = "path/to/your/parquet/files"  # Directory containing original parquet files
output_dir = "path/to/output/rechunked_data"

rechunk_and_save(
    parquet_dir=parquet_dir,
    output_dir=output_dir,
    chunk_sizes_per_episode=chunk_sizes # The chunk sizes obtained in Step 2
)
```

### 4. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.
- [`SimpleDataConfig`](src/openpi/training/config.py): We use SimpleDataConfig instead of LeRobotLiberoDataConfig to enable training from scratch.
- [`MyLocalFASTTokenizer`](src/openpi/models/tokenizer.py):A Local Tokenizer Loader inheriting from FASTTokenizer class, reuses the parent class's tokenize and decode logic.
- [`MyNPZDataset`](src/openpi/training/my_custome_dataset.py): This class creates a PyTorch dataset for loading pre-chunked data samples from .npz files in your directory.

We provide example fine-tuning configs for [π₀-FAST](src/openpi/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_adaptive
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_adaptive --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 5. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero_adaptive --policy.dir=checkpoints/pi0_fast_libero_adaptive/my_experiment/320000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).



### 
