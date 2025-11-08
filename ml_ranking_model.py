import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import List, Tuple
import suffixes as sfx

# ---------- Suffix to ID Mapping ----------
class SuffixVocabulary:
    def __init__(self):
        # Create mappings for all suffixes
        self.suffix_to_id = {}
        self.id_to_suffix = {}
        self.category_to_id = {
            'Noun': 0,
            'Verb': 1
        }
        
        # Build vocabulary from all registered suffixes
        for idx, suffix in enumerate(sfx.ALL_SUFFIXES):
            self.suffix_to_id[suffix.name] = idx
            self.id_to_suffix[idx] = suffix.name
        
        self.num_suffixes = len(self.suffix_to_id)
        print(f"Loaded {self.num_suffixes} suffixes into vocabulary")
    
    def encode_suffix_chain(self, suffix_objects: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of Suffix objects to tensor representations.
        Returns: (object_ids, category_ids)
        """
        if not suffix_objects:
            # Empty chain - return single padding token
            return torch.tensor([0]), torch.tensor([0])
        
        object_ids = []
        category_ids = []
        
        for suffix_obj in suffix_objects:
            suffix_id = self.suffix_to_id.get(suffix_obj.name, 0)  # 0 for unknown
            category_id = self.category_to_id.get(suffix_obj.makes.name, 0)
            
            object_ids.append(suffix_id)
            category_ids.append(category_id)
        
        return torch.tensor(object_ids), torch.tensor(category_ids)
    
    def save(self, path: str):
        """Save vocabulary to file"""
        data = {
            'suffix_to_id': self.suffix_to_id,
            'id_to_suffix': self.id_to_suffix,
            'category_to_id': self.category_to_id
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.suffix_to_id = data['suffix_to_id']
        self.id_to_suffix = data['id_to_suffix']
        self.category_to_id = data['category_to_id']
        self.num_suffixes = len(self.suffix_to_id)


# ---------- Model ----------
class DecompositionRanker(nn.Module):
    def __init__(self, vocab_size, num_categories=2, embed_dim=128, num_layers=4, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embeddings
        self.suffix_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)  # +1 for padding
        self.category_embed = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(50, embed_dim)  # max 50 suffixes in a chain
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Scoring head with multiple layers
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, suffix_ids, category_ids, mask=None):
        """
        suffix_ids: (batch, seq_len) - IDs of suffixes
        category_ids: (batch, seq_len) - Category IDs (Noun/Verb)
        mask: (batch, seq_len) - attention mask (True = ignore)
        Returns: (batch,) - scores for each decomposition
        """
        B, L = suffix_ids.shape
        
        # Create position indices
        pos = torch.arange(L, device=suffix_ids.device).unsqueeze(0).expand(B, L)
        
        # Combine embeddings
        x = (
            self.suffix_embed(suffix_ids)
            + self.category_embed(category_ids)
            + self.pos_embed(pos)
        )
        
        # Apply transformer encoder
        x = self.encoder(x, src_key_padding_mask=mask)
        
        # Mean pooling (ignoring padding)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(mask_expanded, 0.0)
            seq_len = (~mask).sum(dim=1, keepdim=True).float()
            pooled = x.sum(dim=1) / seq_len.clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        
        # Score
        score = self.scorer(pooled).squeeze(-1)
        return score


# ---------- Training utilities ----------
def preference_loss(score_correct, score_incorrect, margin=1.0):
    """
    Pairwise ranking loss: correct decomposition should score higher
    Uses margin ranking loss for more stable training
    """
    # We want score_correct > score_incorrect
    target = torch.ones_like(score_correct)
    loss = F.margin_ranking_loss(score_correct, score_incorrect, target, margin=margin)
    return loss


def contrastive_loss(score_correct, scores_incorrect, temperature=0.1):
    """
    Contrastive loss: treat correct as positive, all others as negatives
    More suitable when you have multiple incorrect candidates
    """
    # Concatenate correct and incorrect scores
    all_scores = torch.cat([score_correct.unsqueeze(1), scores_incorrect], dim=1)
    
    # Apply softmax with temperature
    probs = F.softmax(all_scores / temperature, dim=1)
    
    # Loss is negative log probability of correct (index 0)
    loss = -torch.log(probs[:, 0] + 1e-8).mean()
    return loss


class DecompositionTrainer:
    def __init__(self, model, vocab, lr=1e-4, device='cpu'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.training_history = []
    
    def prepare_batch(self, suffix_chains: List[List], correct_idx: int):
        """
        Convert suffix chains to model inputs
        suffix_chains: List of decompositions (each is a list of Suffix objects)
        correct_idx: Index of the correct decomposition
        """
        # Encode all chains
        encoded = [self.vocab.encode_suffix_chain(chain) for chain in suffix_chains]
        
        # Find max length for padding
        max_len = max(len(obj_ids) for obj_ids, _ in encoded)
        
        # Pad sequences
        padded_obj_ids = []
        padded_cat_ids = []
        masks = []
        
        for obj_ids, cat_ids in encoded:
            pad_len = max_len - len(obj_ids)
            
            # Pad with zeros
            padded_obj = F.pad(obj_ids, (0, pad_len), value=0)
            padded_cat = F.pad(cat_ids, (0, pad_len), value=0)
            
            # Create mask (True for padding positions)
            mask = torch.cat([
                torch.zeros(len(obj_ids), dtype=torch.bool),
                torch.ones(pad_len, dtype=torch.bool)
            ])
            
            padded_obj_ids.append(padded_obj)
            padded_cat_ids.append(padded_cat)
            masks.append(mask)
        
        # Stack into batches
        obj_ids_batch = torch.stack(padded_obj_ids).to(self.device)
        cat_ids_batch = torch.stack(padded_cat_ids).to(self.device)
        mask_batch = torch.stack(masks).to(self.device)
        
        return obj_ids_batch, cat_ids_batch, mask_batch, correct_idx
    
    def train_step(self, suffix_chains: List[List], correct_idx: int):
        """Single training step with one word's decompositions"""
        self.model.train()
        
        # Prepare batch
        obj_ids, cat_ids, masks, correct_idx = self.prepare_batch(suffix_chains, correct_idx)
        
        # Forward pass
        scores = self.model(obj_ids, cat_ids, masks)
        
        # Compute loss using contrastive approach
        score_correct = scores[correct_idx:correct_idx+1]
        incorrect_indices = [i for i in range(len(scores)) if i != correct_idx]
        scores_incorrect = scores[incorrect_indices].unsqueeze(0)
        
        loss = contrastive_loss(score_correct, scores_incorrect)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Record
        self.training_history.append(loss.item())
        
        return loss.item()
    
    def predict(self, suffix_chains: List[List]) -> Tuple[int, List[float]]:
        """
        Predict the best decomposition
        Returns: (best_index, all_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            obj_ids, cat_ids, masks, _ = self.prepare_batch(suffix_chains, 0)
            scores = self.model(obj_ids, cat_ids, masks)
            scores_list = scores.cpu().tolist()
            best_idx = scores.argmax().item()
        
        return best_idx, scores_list
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"Checkpoint loaded from {path}")

    def train_step_pairwise(self, suffix_chains: List[List], better_idx: int, worse_idx: int):
        """Training step for pairwise preference"""
        self.model.train()
        
        # Prepare batch
        obj_ids, cat_ids, masks, _ = self.prepare_batch(suffix_chains, 0)
        
        # Forward pass
        scores = self.model(obj_ids, cat_ids, masks)
        
        # Get scores for the pair
        score_better = scores[better_idx]
        score_worse = scores[worse_idx]
        
        # Pairwise margin ranking loss
        loss = preference_loss(score_better.unsqueeze(0), score_worse.unsqueeze(0), margin=1.0)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_history.append(loss.item())
    
        return loss.item()


# ---------- Example usage ----------
if __name__ == "__main__":
    # Initialize vocabulary
    vocab = SuffixVocabulary()
    
    # Initialize model
    model = DecompositionRanker(
        vocab_size=vocab.num_suffixes,
        num_categories=2,
        embed_dim=128,
        num_layers=4,
        num_heads=8
    )
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = DecompositionTrainer(model, vocab, lr=1e-4, device=device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {device}")
