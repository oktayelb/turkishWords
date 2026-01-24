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
    category_num: int = 2  # Noun/Verb
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # --- Training Hyperparameters ---
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # --- Interactive/Loop Settings ---
    checkpoint_frequency: int = 10  # Save every N examples
    patience: int = 10             # Early stopping patience
    margin: float = 0.5            # Margin for contrastive loss

# Create the global config instance
config = MLConfig()