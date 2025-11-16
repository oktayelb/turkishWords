"""
Improved Machine Learning Ranking Model for Turkish Morphological Decomposition

Key improvements:
    - Removed external dependencies - works with pre-encoded data
    - More concise code without losing features
    - Better type hints and documentation
    - Optimized batch processing
    - Streamlined training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================


#TODO triplets ve lsm her zaman false olacak\ duzelt.
class Ranker(nn.Module):
    """Transformer or LSTM-based ranker for morphological decompositions."""
    
    def __init__(
        self, 
        suffix_vocab_size: int,
        num_categories: int = 2,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        use_lstm: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_lstm = use_lstm
        
        # Embeddings (concatenated: suffix + category + position)
        self.suffix_embed = nn.Embedding(suffix_vocab_size + 1, embed_dim, padding_idx=0)
        self.category_embed = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(50, embed_dim)
        
        # Project concatenated embeddings
        self.input_proj = nn.Linear(embed_dim * 3, embed_dim)
        
        # Encoder
        if use_lstm:
            self.encoder = nn.LSTM(
                embed_dim, embed_dim // 2, num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True, batch_first=True
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, dropout=dropout,
                batch_first=True, activation='gelu'
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) if 'embed' in name else nn.init.kaiming_normal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, suffix_ids: torch.Tensor, category_ids: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            suffix_ids: (batch, seq_len)
            category_ids: (batch, seq_len)
            mask: (batch, seq_len) - True for padding
        Returns:
            scores: (batch,)
        """
        B, L = suffix_ids.shape
        
        # Embeddings
        pos = torch.arange(L, device=suffix_ids.device).unsqueeze(0).expand(B, L)
        x = torch.cat([
            self.suffix_embed(suffix_ids),
            self.category_embed(category_ids),
            self.pos_embed(pos)
        ], dim=-1)
        x = self.input_proj(x)
        
        # Encode
        if self.use_lstm:
            x, _ = self.encoder(x)
        else:
            x = self.encoder(x, src_key_padding_mask=mask)
        
        # Mean pool (mask-aware)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            pooled = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        
        return self.scorer(pooled).squeeze(-1)


# ============================================================================
# DATA UTILITIES
# ============================================================================

class DataAugmenter:
    """Generate negative examples for contrastive learning."""
    
    @staticmethod
    def corrupt_chain(suffix_chain: List[Tuple[int, int]], num_negatives: int = 3) -> List[List[Tuple[int, int]]]:
        """
        Corrupt a valid suffix chain to create negative examples.
        
        Args:
            suffix_chain: List of (suffix_id, category_id) tuples
            num_negatives: Number of corrupted versions to generate
            
        Returns:
            List of corrupted chains
        """
        if not suffix_chain:
            return []
        
        negatives = []
        
        # Strategy 1: Remove random suffix
        if len(suffix_chain) > 1:
            for i in range(min(num_negatives, len(suffix_chain))):
                corrupted = suffix_chain[:i] + suffix_chain[i+1:]
                if corrupted:
                    negatives.append(corrupted)
        
        # Strategy 2: Swap adjacent suffixes
        if len(suffix_chain) > 1:
            for i in range(min(num_negatives - len(negatives), len(suffix_chain) - 1)):
                corrupted = suffix_chain[:i] + [suffix_chain[i+1], suffix_chain[i]] + suffix_chain[i+2:]
                negatives.append(corrupted)
        
        # Strategy 3: Duplicate a suffix
        if len(negatives) < num_negatives and suffix_chain:
            negatives.append(suffix_chain + [suffix_chain[0]])
        
        return negatives[:num_negatives]


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Enhanced trainer with batching, early stopping, and flexible loss."""
    
    def __init__(
        self, 
        model: Ranker,
        model_path: str = "data/turkish_morph_model.pt",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        device: Optional[str] = None,
        patience: int = 10,
        use_triplet_loss: bool = False
    ):
        self.model = model
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.patience = patience
        self.use_triplet_loss = use_triplet_loss
        self.path = model_path
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
        )
        
        # Training state
        self.train_history: List[float] = []
        self.val_history: List[float] = []
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.global_step = 0
        
        # Try loading checkpoint
        try:
            self.load_checkpoint(self.path)
            print(f"✅ Loaded model from {self.path}")
        except:
            print(f"⚠️  Starting fresh (no checkpoint found)")
    
    def _prepare_batch(self, examples: List[Tuple[List[Tuple[int, int]], List[List[Tuple[int, int]]], int]]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Prepare batch from examples.
        
        Args:
            examples: List of (root_chain, candidate_chains, correct_idx)
                     where chains are lists of (suffix_id, category_id) tuples
                     
        Returns:
            (suffix_ids, category_ids, masks, labels, examples_per_word)
        """
        all_suffix_ids, all_category_ids, all_labels = [], [], []
        examples_per_word = []
        
        for root_chain, candidates, correct_idx in examples:
            examples_per_word.append(len(candidates))
            
            for idx, chain in enumerate(candidates):
                # Unzip chain into separate lists
                if chain:
                    suf_ids, cat_ids = zip(*chain)
                else:
                    suf_ids, cat_ids = [], []
                
                all_suffix_ids.append(torch.tensor(suf_ids, dtype=torch.long))
                all_category_ids.append(torch.tensor(cat_ids, dtype=torch.long))
                all_labels.append(1.0 if idx == correct_idx else 0.0)
        
        # Pad to max length
        max_len = max(len(s) for s in all_suffix_ids) if all_suffix_ids else 1
        
        padded_suf, padded_cat, masks = [], [], []
        for suf, cat in zip(all_suffix_ids, all_category_ids):
            pad_len = max_len - len(suf)
            padded_suf.append(F.pad(suf, (0, pad_len), value=0))
            padded_cat.append(F.pad(cat, (0, pad_len), value=0))
            masks.append(torch.cat([torch.zeros(len(suf), dtype=torch.bool), 
                                   torch.ones(pad_len, dtype=torch.bool)]))
        
        return (
            torch.stack(padded_suf).to(self.device),
            torch.stack(padded_cat).to(self.device),
            torch.stack(masks).to(self.device),
            torch.tensor(all_labels, dtype=torch.float32).to(self.device),
            examples_per_word
        )
    
    def _compute_loss(self, scores: torch.Tensor, labels: torch.Tensor, 
                     examples_per_word: List[int]) -> torch.Tensor:
        """Compute contrastive or triplet loss."""
        losses = []
        start = 0
        
        for count in examples_per_word:
            end = start + count
            word_scores = scores[start:end]
            word_labels = labels[start:end]
            
            if self.use_triplet_loss:
                pos_idx = (word_labels == 1.0).nonzero(as_tuple=True)[0]
                neg_idx = (word_labels == 0.0).nonzero(as_tuple=True)[0]
                
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    pos_score = word_scores[pos_idx[0]]
                    neg_score = word_scores[neg_idx].min()
                    losses.append(F.relu(neg_score - pos_score + 1.0))
            else:
                # Softmax cross-entropy
                probs = F.softmax(word_scores / 0.7, dim=0)
                correct_prob = (probs * word_labels).sum()
                losses.append(-torch.log(correct_prob + 1e-8))
            
            start = end
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
    
    def train_epoch(self, data: List) -> float:
        """Train for one epoch."""
        self.model.train()
        losses = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            suf_ids, cat_ids, masks, labels, counts = self._prepare_batch(batch)
            
            scores = self.model(suf_ids, cat_ids, masks)
            loss = self._compute_loss(scores, labels, counts)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            losses.append(loss.item())
            self.global_step += 1
        
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.train_history.append(avg_loss)
        return avg_loss
    
    def validate(self, data: List) -> Tuple[float, float]:
        """Validate and return (loss, accuracy)."""
        self.model.eval()
        losses, correct, total = [], 0, 0
        
        with torch.no_grad():
            for example in data:
                suf_ids, cat_ids, masks, labels, counts = self._prepare_batch([example])
                scores = self.model(suf_ids, cat_ids, masks)
                
                loss = self._compute_loss(scores, labels, counts)
                losses.append(loss.item())
                
                pred_idx = scores.argmax().item()
                if labels[pred_idx] == 1.0:
                    correct += 1
                total += 1
        
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        accuracy = correct / total if total > 0 else 0.0
        self.val_history.append(avg_loss)
        
        return avg_loss, accuracy
    
    def train(self, train_data: List, val_data: Optional[List] = None, 
              num_epochs: int = 100, verbose: bool = True) -> Dict:
        """Full training loop with early stopping."""
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_data)
            
            if val_data:
                val_loss, val_acc = self.validate(val_data)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    self.save_checkpoint(self.path)
                else:
                    self.epochs_no_improve += 1
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | "
                          f"Val: {val_loss:.4f} | Acc: {val_acc:.4f}")
                
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f}")
            
            self.scheduler.step()
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss
        }
    
    def predict(self, candidates: List[List[Tuple[int, int]]]) -> Tuple[int, List[float]]:
        """
        Predict best candidate.
        
        Args:
            candidates: List of suffix chains (each chain is list of (suffix_id, category_id))
            
        Returns:
            (best_idx, all_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Create dummy example
            suf_ids, cat_ids, masks, _, _ = self._prepare_batch([([], candidates, 0)])
            scores = self.model(suf_ids, cat_ids, masks)
            
        return scores.argmax().item(), scores.cpu().tolist()
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }, path)
        print(f"✓ Saved to {path}")
    
    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.train_history = ckpt.get('train_history', [])
        self.val_history = ckpt.get('val_history', [])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
        print(f"✓ Loaded from {path} (step {self.global_step})")

