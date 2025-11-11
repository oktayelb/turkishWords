"""
Machine Learning Ranking Model for Turkish Morphological Decomposition

This module provides a neural network model that learns to rank different
morphological decompositions of Turkish words, helping identify the most
linguistically correct analysis.

Architecture:
    - Transformer-based encoder for suffix sequences
    - Contrastive learning with preference ranking
    - Gradient accumulation for stable training
"""

# torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F

# list etc imports
from typing import List, Tuple

# in project imports
import util.suffixes as sfx
from data.config import TrainingConfig

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class Ranker(nn.Module):
    """
    Transformer-based neural network that scores morphological decompositions.
    
    The model uses three types of embeddings:
        1. Suffix embeddings: Identity of each suffix
        2. Category embeddings: POS category (Noun/Verb) of each suffix
        3. Position embeddings: Position of suffix in the chain
    
    These are combined and processed through a Transformer encoder, then
    pooled and scored to produce a single quality score for the decomposition.
    
    Architecture:
        Input: Sequence of suffixes → Embeddings → Transformer → Pooling → Score
    """
    
    def __init__(
        self, 
        num_categories: int = 2, 
        embed_dim: int = 128, 
        num_layers: int = 4, 
        num_heads: int = 8
    ):
        """
        Initialize the decomposition ranker model.
        
        Args:
            num_categories: Number of POS categories (default: 2 for Noun/Verb)
            embed_dim: Dimensionality of embeddings and hidden states
            num_layers: Number of Transformer encoder layers
            num_heads: Number of attention heads in Transformer
        """
        super().__init__()
        self.embed_dim = TrainingConfig.embed_dim
        self.num_layers=TrainingConfig.num_layers,
        self.num_heads=TrainingConfig.num_heads
        
        # ---- Embedding Layers ----
        # +1 for padding token (ID 0)
        self.suffix_embed = nn.Embedding(
            len(sfx.ALL_SUFFIXES) + 1, 
            embed_dim, 
            padding_idx=0
        )
        self.category_embed = nn.Embedding(
            num_categories + 1, 
            embed_dim, 
            padding_idx=0
        )
        # Position embeddings (support up to 50 suffixes in a chain)
        self.pos_embed = nn.Embedding(50, embed_dim)
        
        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True  # Input shape: (batch, seq, features)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ---- Scoring Head ----
        # Multi-layer perceptron that reduces embeddings to a single score
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1)  # Final score
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:  # Only for matrices, not biases
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        suffix_ids: torch.Tensor, 
        category_ids: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass: compute scores for a batch of decompositions.
        
        Args:
            suffix_ids: (batch, seq_len) - Numeric IDs of suffixes
            category_ids: (batch, seq_len) - POS category IDs (Noun/Verb)
            mask: (batch, seq_len) - Attention mask (True = ignore padding)
            
        Returns:
            scores: (batch,) - Quality score for each decomposition
        """
        batch_size, seq_len = suffix_ids.shape
        
        # ---- Create position indices ----
        # [0, 1, 2, ..., seq_len-1] for each batch item
        pos = torch.arange(seq_len, device=suffix_ids.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)
        
        # ---- Combine all embeddings ----
        # Sum suffix identity + POS category + position information
        x = (
            self.suffix_embed(suffix_ids) +
            self.category_embed(category_ids) +
            self.pos_embed(pos)
        )
        
        # ---- Apply Transformer encoder ----
        # Process sequence with self-attention
        x = self.encoder(x, src_key_padding_mask=mask)
        
        # ---- Mean pooling over sequence ----
        # Average the embeddings, ignoring padding positions
        if mask is not None:
            # Expand mask to match embedding dimensions
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            # Zero out padding positions
            x = x.masked_fill(mask_expanded, 0.0)
            # Count non-padding positions per sequence
            seq_len = (~mask).sum(dim=1, keepdim=True).float()
            # Average over non-padding positions
            pooled = x.sum(dim=1) / seq_len.clamp(min=1)
        else:
            # Simple mean if no mask provided
            pooled = x.mean(dim=1)
        
        # ---- Compute final score ----
        score = self.scorer(pooled).squeeze(-1)
        return score


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def preference_loss(
    score_correct: torch.Tensor, 
    score_incorrect: torch.Tensor, 
    margin: float = 1.0
) -> torch.Tensor:
    """
    Pairwise ranking loss: correct decomposition should score higher.
    
    This loss encourages the model to score the correct decomposition at least
    'margin' points higher than incorrect ones. Uses margin ranking loss for
    stable training gradients.
    
    Args:
        score_correct: Score(s) of correct decomposition
        score_incorrect: Score(s) of incorrect decomposition
        margin: Minimum desired score difference (default: 1.0)
        
    Returns:
        Loss value (scalar)
        
    Formula:
        loss = max(0, margin - (score_correct - score_incorrect))
    """
    # Target = 1 means we want score_correct > score_incorrect
    target = torch.ones_like(score_correct)
    loss = F.margin_ranking_loss(score_correct, score_incorrect, target, margin=margin)
    return loss


def contrastive_loss(
    score_correct: torch.Tensor, 
    scores_incorrect: torch.Tensor, 
    temperature: float = 0.5
) -> torch.Tensor:
    """
    Contrastive loss: treat correct as positive, all others as negatives.
    
    This loss is more suitable when you have multiple incorrect candidates.
    It maximizes the probability of the correct decomposition relative to
    all incorrect ones simultaneously.
    
    Args:
        score_correct: Score of the correct decomposition
        scores_incorrect: Scores of all incorrect decompositions
        temperature: Controls softmax sharpness (default: 0.5)
            - Higher (0.5-1.0): Smoother gradients, better for similar candidates
            - Lower (0.1): Sharp distinctions, risk of vanishing gradients
            
    Returns:
        Loss value (scalar)
        
    Formula:
        loss = -log(exp(score_correct/T) / sum(exp(all_scores/T)))
    """
    # Concatenate correct and incorrect scores
    # Shape: (batch, num_candidates)
    all_scores = torch.cat([score_correct.unsqueeze(1), scores_incorrect], dim=1)
    
    # Apply softmax with temperature scaling
    probs = F.softmax(all_scores / temperature, dim=1)
    
    # Loss is negative log probability of correct decomposition (index 0)
    # Add small epsilon to prevent log(0)
    loss = -torch.log(probs[:, 0] + 1e-8).mean()
    return loss


# ============================================================================
# TRAINING MANAGER
# ============================================================================

class Trainer:
    """
    Manages the training process for the DecompositionRanker model.
    
    Features:
        - Gradient accumulation for stable training on small batches
        - Learning rate warmup for better convergence
        - Adaptive learning rate scheduling
        - Checkpoint saving/loading with full training state
        - Training history tracking
    """
    
    def __init__(
        self, 
        model: Ranker = Ranker(), 
        lr: float = 1e-4, 
        device: str = 'cpu',
        accumulation_steps: int = 4,
        warmup_steps: int = 100
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The DecompositionRanker model to train
            vocab: Vocabulary for encoding suffix chains
            lr: Base learning rate (default: 1e-4)
            device: Device to train on ('cpu' or 'cuda')
            accumulation_steps: Number of steps to accumulate gradients (default: 4)
            warmup_steps: Number of warmup steps for learning rate (default: 100)
        """

        self.model = model.to(device)
        self.base_lr = lr
        self.lr  = TrainingConfig.learning_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # ---- Gradient Accumulation ----
        # Accumulate gradients over multiple examples for stability
        self.accumulation_steps = accumulation_steps
        self.accumulated_loss = 0.0
        self.step_count = 0
        
        # ---- Learning Rate Warmup ----
        # Gradually increase LR from 0 to base_lr over warmup_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # ---- Optimizer Setup ----
        # AdamW includes weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=0.01
        )
        
        # ---- Learning Rate Scheduler ----
        # Reduce LR when loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',      # Minimize loss
            factor=0.5,      # Reduce LR by half
            patience=10      # Wait 10 steps before reducing
        )
        
        # ---- Training History ----
        self.training_history: List[float] = []
    
    def _update_learning_rate(self, loss: float = None) -> None:
        """
        Update learning rate with warmup and scheduling.
        
        During warmup: LR increases linearly from 0 to base_lr
        After warmup: LR adjusts based on loss plateau detection
        
        Args:
            loss: Current loss value for scheduler (optional during warmup)
        """
        self.current_step += 1
        
        # ---- Phase 1: Warmup ----
        if self.current_step <= self.warmup_steps:
            # Linear warmup: LR = base_lr * (step / warmup_steps)
            lr_scale = self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale
        
        # ---- Phase 2: Adaptive Scheduling ----
        elif loss is not None:
            # Use plateau detection to adjust LR
            self.scheduler.step(loss)
    
    def prepare_batch(
        self, 
        suffix_chains: List[List], 
        correct_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Convert suffix chains to padded tensors for batch processing.
        
        This function:
        1. Encodes each suffix chain to numeric IDs
        2. Pads all chains to the same length
        3. Creates attention masks to ignore padding
        
        Args:
            suffix_chains: List of decompositions (each is a list of Suffix objects)
            correct_idx: Index of the correct decomposition
            
        Returns:
            Tuple of:
                - obj_ids_batch: (num_decomps, max_len) - Padded suffix IDs
                - cat_ids_batch: (num_decomps, max_len) - Padded category IDs
                - mask_batch: (num_decomps, max_len) - Padding masks (True = padding)
                - correct_idx: Index of correct decomposition (unchanged)
        """
        # ---- Step 1: Encode all chains ----
        encoded = []
        for chain in suffix_chains:
            object_ids, category_ids = sfx.encode_suffix_chain(chain)
            encoded.append((torch.tensor(object_ids),torch.tensor(category_ids)))
        
        
        # ---- Step 2: Find max length for padding ----
        max_len = max(len(obj_ids) for obj_ids, _ in encoded)
        
        # ---- Step 3: Pad sequences to max_len ----
        padded_obj_ids = []
        padded_cat_ids = []
        masks = []
        
        for obj_ids, cat_ids in encoded:
            pad_len = max_len - len(obj_ids)
            
            # Pad with zeros (padding token ID)
            padded_obj = F.pad(obj_ids, (0, pad_len), value=0)
            padded_cat = F.pad(cat_ids, (0, pad_len), value=0)
            
            # Create attention mask
            # False for real tokens, True for padding (to be ignored)
            mask = torch.cat([
                torch.zeros(len(obj_ids), dtype=torch.bool),  # Real tokens
                torch.ones(pad_len, dtype=torch.bool)          # Padding
            ])
            
            padded_obj_ids.append(padded_obj)
            padded_cat_ids.append(padded_cat)
            masks.append(mask)
        
        # ---- Step 4: Stack into batch tensors ----
        obj_ids_batch = torch.stack(padded_obj_ids).to(self.device)
        cat_ids_batch = torch.stack(padded_cat_ids).to(self.device)
        mask_batch = torch.stack(masks).to(self.device)
        
        return obj_ids_batch, cat_ids_batch, mask_batch, correct_idx
    
    def train_step(
        self, 
        suffix_chains: List[List], 
        correct_idx: int
    ) -> float:
        """
        Single training step with contrastive loss and gradient accumulation.
        
        This method:
        1. Computes scores for all decompositions
        2. Applies contrastive loss (correct vs. all incorrect)
        3. Accumulates gradients over multiple steps
        4. Updates weights periodically
        
        Args:
            suffix_chains: List of all decomposition candidates
            correct_idx: Index of the correct decomposition
            
        Returns:
            Current loss value (accumulated if mid-batch, final if batch complete)
        """
        self.model.train()
        
        # ---- Prepare batch ----
        obj_ids, cat_ids, masks, correct_idx = self.prepare_batch(
            suffix_chains, correct_idx
        )
        
        # ---- Forward pass ----
        scores = self.model(obj_ids, cat_ids, masks)
        
        # ---- Compute contrastive loss ----
        # Correct score
        score_correct = scores[correct_idx:correct_idx+1]
        
        # All incorrect scores
        incorrect_indices = [i for i in range(len(scores)) if i != correct_idx]
        scores_incorrect = scores[incorrect_indices].unsqueeze(0)
        
        loss = contrastive_loss(score_correct, scores_incorrect)
        
        # ---- Normalize by accumulation steps ----
        # This makes the effective learning rate consistent
        loss = loss / self.accumulation_steps
        
        # ---- Backward pass (accumulate gradients) ----
        loss.backward()
        
        # Track accumulated loss
        self.accumulated_loss += loss.item()
        self.step_count += 1
        
        # ---- Update weights every N steps ----
        if self.step_count % self.accumulation_steps == 0:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Apply accumulated gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update learning rate
            self._update_learning_rate(loss=self.accumulated_loss)
            
            # Record final accumulated loss
            self.training_history.append(self.accumulated_loss)
            final_loss = self.accumulated_loss
            self.accumulated_loss = 0.0
            
            return final_loss
        
        # Return current accumulated loss (scaled back for display)
        return self.accumulated_loss * self.accumulation_steps
    
    def train_step_pairwise(
        self, 
        suffix_chains: List[List], 
        better_idx: int, 
        worse_idx: int
    ) -> float:
        """
        Training step for pairwise preference ranking.
        
        This is used when the user indicates one decomposition is better than
        another, but both might be acceptable. Uses margin ranking loss.
        
        Args:
            suffix_chains: List of all decomposition candidates
            better_idx: Index of the preferred decomposition
            worse_idx: Index of the less preferred decomposition
            
        Returns:
            Current loss value (accumulated if mid-batch, final if batch complete)
        """
        self.model.train()
        
        # ---- Prepare batch ----
        obj_ids, cat_ids, masks, _ = self.prepare_batch(suffix_chains, 0)
        
        # ---- Forward pass ----
        scores = self.model(obj_ids, cat_ids, masks)
        
        # ---- Get scores for the pair ----
        score_better = scores[better_idx]
        score_worse = scores[worse_idx]
        
        # ---- Compute pairwise margin ranking loss ----
        loss = preference_loss(
            score_better.unsqueeze(0), 
            score_worse.unsqueeze(0), 
            margin=1.0
        )
        
        # ---- Normalize by accumulation steps ----
        loss = loss / self.accumulation_steps
        
        # ---- Backward pass (accumulate gradients) ----
        loss.backward()
        
        # Track accumulated loss
        self.accumulated_loss += loss.item()
        self.step_count += 1
        
        # ---- Update weights every N steps ----
        if self.step_count % self.accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Apply accumulated gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update learning rate
            self._update_learning_rate(loss=self.accumulated_loss)
            
            # Record final accumulated loss
            self.training_history.append(self.accumulated_loss)
            final_loss = self.accumulated_loss
            self.accumulated_loss = 0.0
            
            return final_loss
        
        # Return current accumulated loss (scaled back for display)
        return self.accumulated_loss * self.accumulation_steps
    
    def predict(
        self, 
        suffix_chains: List[List]
    ) -> Tuple[int, List[float]]:
        """
        Predict the best decomposition among candidates.
        
        Args:
            suffix_chains: List of all decomposition candidates
            
        Returns:
            Tuple of:
                - best_idx: Index of highest-scoring decomposition
                - scores_list: List of all scores (for display/analysis)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare batch (correct_idx doesn't matter for inference)
            obj_ids, cat_ids, masks, _ = self.prepare_batch(suffix_chains, 0)
            
            # Get scores for all candidates
            scores = self.model(obj_ids, cat_ids, masks)
            
            # Convert to list and find best
            scores_list = scores.cpu().tolist()
            best_idx = scores.argmax().item()
        
        return best_idx, scores_list
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save complete training state to file.
        
        Saves:
            - Model weights
            - Optimizer state
            - Learning rate scheduler state
            - Training history
            - Step counters
            - Accumulated gradients
            
        Args:
            path: Filepath to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'current_step': self.current_step,
            'step_count': self.step_count,
            'accumulated_loss': self.accumulated_loss
        }, path)
        print(f"✓ Model checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load complete training state from file.
        
        Args:
            path: Filepath to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler if available
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.training_history = checkpoint.get('training_history', [])
        self.current_step = checkpoint.get('current_step', 0)
        self.step_count = checkpoint.get('step_count', 0)
        self.accumulated_loss = checkpoint.get('accumulated_loss', 0.0)
        
        print(f"✓ Model checkpoint loaded from {path}")
        print(f"  Resumed at step {self.current_step} (warmup: {self.warmup_steps})")
