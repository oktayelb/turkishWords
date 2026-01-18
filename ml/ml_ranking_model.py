import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .config import config  # Direct import from sibling file

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class Ranker(nn.Module):
    
    def __init__(self, suffix_vocab_size: int):
        """
        Args:
            suffix_vocab_size: The only dynamic parameter (depends on loaded dictionary)
        """
        super().__init__()
        # Load hyperparameters directly from config
        self.embed_dim = config.embed_dim
        
        # Embeddings
        self.suffix_embed = nn.Embedding(suffix_vocab_size + 1, self.embed_dim, padding_idx=0)
        self.category_embed = nn.Embedding(config.category_num + 1, self.embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(50, self.embed_dim)
        
        # Project
        self.input_proj = nn.Linear(self.embed_dim * 3, self.embed_dim)
        
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=config.num_heads,
            dim_feedforward=self.embed_dim * 4, 
            dropout=config.dropout,
            batch_first=True, 
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        
        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), 
            nn.LayerNorm(self.embed_dim), 
            nn.GELU(), 
            nn.Dropout(config.dropout),
            nn.Linear(self.embed_dim, self.embed_dim // 2), 
            nn.GELU(), 
            nn.Dropout(config.dropout),
            nn.Linear(self.embed_dim // 2, 1)
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
        B, L = suffix_ids.shape
        pos = torch.arange(L, device=suffix_ids.device).unsqueeze(0).expand(B, L)
        
        x = torch.cat([
            self.suffix_embed(suffix_ids),
            self.category_embed(category_ids),
            self.pos_embed(pos)
        ], dim=-1)
        
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            pooled = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        
        return self.scorer(pooled).squeeze(-1)

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    
    def __init__(self, model: Ranker):
        """
        Initializes trainer using settings from ml.config
        """
        self.model = model
        
        # Load settings from config
        self.batch_size = config.batch_size
        self.patience = config.patience
        self.path = str(config.model_path) # Convert Path object to string for torch
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay, 
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=config.learning_rate * 0.01
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
        except FileNotFoundError:
            print(f"⚠️  Starting fresh (no checkpoint found at {self.path})")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")

    # ... [Keep the rest of the Trainer methods (_prepare_batch, train_persistent, etc.) exactly as they were] ...
    # ... [Ensure _compute_loss uses config.margin if you want to use it there, or keep it hardcoded] ...
    
    def _prepare_batch(self, examples):
        # (Standard implementation as before)
        all_suffix_ids, all_category_ids, all_labels = [], [], []
        examples_per_word = []
        
        for root_chain, candidates, correct_idx in examples:
            examples_per_word.append(len(candidates))
            for idx, chain in enumerate(candidates):
                if chain:
                    suf_ids, cat_ids = zip(*chain)
                else:
                    suf_ids, cat_ids = [], []
                all_suffix_ids.append(torch.tensor(suf_ids, dtype=torch.long))
                all_category_ids.append(torch.tensor(cat_ids, dtype=torch.long))
                all_labels.append(1.0 if idx == correct_idx else 0.0)
        
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

    def _compute_loss(self, scores, labels, examples_per_word):
        losses = []
        start = 0
        for count in examples_per_word:
            end = start + count
            word_scores = scores[start:end]
            word_labels = labels[start:end]
            
            # Softmax cross-entropy
            probs = F.softmax(word_scores / 0.7, dim=0)
            correct_prob = (probs * word_labels).sum()
            losses.append(-torch.log(correct_prob + 1e-8))
            start = end
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def train_persistent(self, data: List, max_retries: int = 20) -> float:
        # Uses config.margin
        self.model.train()
        final_loss = 0.0
        suf_ids, cat_ids, masks, labels, counts = self._prepare_batch(data)
        
        print(f"   💪 Enforcing correct choice...", end="", flush=True)
        
        for attempt in range(max_retries):
            scores = self.model(suf_ids, cat_ids, masks)
            loss = self._compute_loss(scores, labels, counts)
            final_loss = loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.global_step += 1
            
            with torch.no_grad():
                new_scores = self.model(suf_ids, cat_ids, masks)
                all_learned = True
                start = 0
                for count in counts:
                    end = start + count
                    word_scores = new_scores[start:end]
                    word_labels = labels[start:end]
                    correct_idx = word_labels.argmax()
                    correct_score = word_scores[correct_idx]
                    
                    other_scores = word_scores.clone()
                    other_scores[correct_idx] = -float('inf')
                    max_incorrect = other_scores.max()
                    
                    if correct_score <= (max_incorrect + config.margin):
                        all_learned = False
                        break
                    start = end
                
                if all_learned:
                    print(f" Learned in {attempt + 1} steps.")
                    break
                if attempt % 5 == 0: print(".", end="", flush=True)
        else:
            print(f" Limit reached ({max_retries}).")
            
        self.train_history.append(final_loss)
        return final_loss

    def predict(self, candidates):
        self.model.eval()
        with torch.no_grad():
            suf_ids, cat_ids, masks, _, _ = self._prepare_batch([([], candidates, 0)])
            scores = self.model(suf_ids, cat_ids, masks)
        return scores.argmax().item(), scores.cpu().tolist()

    def batch_predict(self, batch_candidates):
        # Implementation identical to previous version
        self.model.eval()
        dummy_examples = [([], cands, 0) for cands in batch_candidates]
        results = []
        chunk_size = 256
        
        with torch.no_grad():
            for i in range(0, len(dummy_examples), chunk_size):
                chunk = dummy_examples[i:i + chunk_size]
                suf_ids, cat_ids, masks, _, counts = self._prepare_batch(chunk)
                scores = self.model(suf_ids, cat_ids, masks)
                scores_list = scores.cpu().tolist()
                start = 0
                for count in counts:
                    end = start + count
                    word_scores = scores_list[start:end]
                    best_idx = word_scores.index(max(word_scores))
                    results.append((best_idx, word_scores))
                    start = end
        return results

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

    # Add train and validate methods here (omitted for brevity, same as before)