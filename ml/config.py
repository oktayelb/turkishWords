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
    embed_dim: int = 128        # Kept at 128; optimal for a 300 vocab size.
    num_layers: int = 3         # Reduced to 3 to prevent memorizing the 7,000 sequences.
    num_heads: int = 8         
    dropout: float = 0.2        # Increased to combat overfitting on the small sequence count.

    # --- Training Hyperparameters ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.05  # Increased for stronger regularization.

    # --- Experience Replay ---
    # Sized to hold the entire dataset of 7,000 sentences in memory.
    replay_buffer_size: int = 7000   
    replay_k: int = 64
    steps_per_update: int = 4       

    # --- Interactive/Loop Settings ---
    checkpoint_frequency: int = 1000  

# Create the global config instance
config = MLConfig()