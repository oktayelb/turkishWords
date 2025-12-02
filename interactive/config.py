class TrainingConfig:
    """Configuration for the interactive trainer"""
    # Save every N examples
    checkpoint_frequency: int = 10  
    # Model hyperparameters
    category_num: int = 2
    batch_size: int = 16
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    learning_rate: float = 1e-4

