from dataclasses import dataclass
@dataclass
class TrainingConfig:
    """Configuration for the interactive trainer"""
    model_path: str = "data/turkish_morph_model.pt"
    vocab_path: str = "data/suffix_vocab.json"
    training_count_file: str = "data/training_count.txt"
    valid_decompositions_file: str = "data/valid_decompositions.jsonl"
    checkpoint_frequency: int = 10  # Save every N examples
    
    # Model hyperparameters
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    learning_rate: float = 1e-4
