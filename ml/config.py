from pathlib import Path
from dataclasses import dataclass

# Dynamic Path Resolution
# Assumes structure: savyar/ml/config.py
# Base dir becomes:  savyar/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

@dataclass
class MLConfig:
    # --- File Paths ---
    model_path: Path = "ml/model.pt"
    training_count_file: Path = DATA_DIR / "training_count.txt"
    
    # --- Model Architecture ---
    # Vocab size is dynamic (passed at runtime), others are static
    embed_dim: int = 64        # 64 suits current vocab size (~290) and dataset scale.
    num_layers: int = 2        # Increase to 4 once you have >2000 confirmed sentences.
    num_heads: int = 4         # Must divide embed_dim evenly.
    dropout: float = 0.1

    # --- Training Hyperparameters ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # --- Experience Replay ---
    # On every training call, the new example is mixed with `replay_k` randomly
    # sampled past examples and trained for `steps_per_update` gradient steps.
    # This prevents catastrophic forgetting and stops single-example memorisation.
    replay_buffer_size: int = 300   # Max past sequences to keep in memory.
    replay_k: int = 7               # Past examples mixed in per new one.
    steps_per_update: int = 4       # Gradient steps per training call.

    # --- Interactive/Loop Settings ---
    checkpoint_frequency: int = 10  # Save every N confirmed examples.

# Create the global config instance
config = MLConfig()