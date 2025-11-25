"""
Trainer for Context-Aware Morphological Decomposition Model
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class Trainer:
    """Trainer with context window support"""
    
    def __init__(
        self, 
        model,  # Ranker
        model_path: str = "data/turkish_morph_contextual_model.pt",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        device: Optional[str] = None,
        patience: int = 10,
        context_window: int = 5,  # words before and after
        max_word_len: int = 20
    ):
        self.model = model
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.patience = patience
        self.path = model_path
        self.context_window = context_window
        self.max_word_len = max_word_len
        
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
        
        try:
            self.load_checkpoint(self.path)
            print(f"✅ Loaded contextual model from {self.path}")
        except:
            print(f"⚠️  Starting fresh (no checkpoint found)")
    
    def _words_to_char_ids(self, words: List[str]) -> torch.Tensor:
        """Convert words to character ID tensors"""
        char_ids = []
        for word in words:
            chars = [min(ord(c), 255) for c in word[:self.max_word_len]]
            chars += [0] * (self.max_word_len - len(chars))
            char_ids.append(chars)
        return torch.tensor(char_ids, dtype=torch.long)
    
    def _prepare_batch_with_context(
        self, 
        examples: List[Tuple[List[str], List[Tuple[int, int]], List[List[Tuple[int, int]]], int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Prepare batch with context information.
        
        Args:
            examples: List of (context_words, root_chain, candidate_chains, correct_idx)
                     context_words: List of surrounding words
                     root_chain: unused (kept for compatibility)
                     candidate_chains: List of suffix chains
                     correct_idx: Index of correct decomposition
                     
        Returns:
            (suffix_ids, category_ids, masks, context_chars, context_masks, labels, examples_per_word)
        """
        all_suffix_ids, all_category_ids, all_labels = [], [], []
        all_context_chars, all_context_masks = [], []
        examples_per_word = []
        
        for context_words, _, candidates, correct_idx in examples:
            examples_per_word.append(len(candidates))
            
            # Convert context words to character IDs
            if context_words:
                ctx_chars = self._words_to_char_ids(context_words)
                ctx_mask = torch.ones(len(context_words), dtype=torch.bool)
            else:
                # No context - create dummy
                ctx_chars = torch.zeros((1, self.max_word_len), dtype=torch.long)
                ctx_mask = torch.zeros(1, dtype=torch.bool)
            
            for idx, chain in enumerate(candidates):
                # Unzip chain into separate lists
                if chain:
                    suf_ids, cat_ids = zip(*chain)
                else:
                    suf_ids, cat_ids = [], []
                
                all_suffix_ids.append(torch.tensor(suf_ids, dtype=torch.long))
                all_category_ids.append(torch.tensor(cat_ids, dtype=torch.long))
                all_labels.append(1.0 if idx == correct_idx else 0.0)
                
                # Same context for all candidates of this word
                all_context_chars.append(ctx_chars)
                all_context_masks.append(ctx_mask)
        
        # Pad decompositions to max length
        max_len = max(len(s) for s in all_suffix_ids) if all_suffix_ids else 1
        
        padded_suf, padded_cat, masks = [], [], []
        for suf, cat in zip(all_suffix_ids, all_category_ids):
            pad_len = max_len - len(suf)
            padded_suf.append(F.pad(suf, (0, pad_len), value=0))
            padded_cat.append(F.pad(cat, (0, pad_len), value=0))
            masks.append(torch.cat([torch.zeros(len(suf), dtype=torch.bool), 
                                   torch.ones(pad_len, dtype=torch.bool)]))
        
        # Pad context to max context length
        max_ctx_len = max(len(ctx) for ctx in all_context_chars)
        padded_ctx_chars, padded_ctx_masks = [], []
        
        for ctx_chars, ctx_mask in zip(all_context_chars, all_context_masks):
            pad_len = max_ctx_len - len(ctx_chars)
            if pad_len > 0:
                padding = torch.zeros((pad_len, self.max_word_len), dtype=torch.long)
                padded_ctx_chars.append(torch.cat([ctx_chars, padding], dim=0))
                padded_ctx_masks.append(torch.cat([ctx_mask, torch.zeros(pad_len, dtype=torch.bool)]))
            else:
                padded_ctx_chars.append(ctx_chars)
                padded_ctx_masks.append(ctx_mask)
        
        return (
            torch.stack(padded_suf).to(self.device),
            torch.stack(padded_cat).to(self.device),
            torch.stack(masks).to(self.device),
            torch.stack(padded_ctx_chars).to(self.device),
            torch.stack(padded_ctx_masks).to(self.device),
            torch.tensor(all_labels, dtype=torch.float32).to(self.device),
            examples_per_word
        )
    
    def _compute_loss(self, scores: torch.Tensor, labels: torch.Tensor, 
                     examples_per_word: List[int]) -> torch.Tensor:
        """Compute softmax cross-entropy loss per word"""
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
    
    def train_epoch(self, data: List) -> float:
        """Train for one epoch with context"""
        self.model.train()
        losses = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            suf_ids, cat_ids, masks, ctx_chars, ctx_masks, labels, counts = \
                self._prepare_batch_with_context(batch)
            
            scores = self.model(suf_ids, cat_ids, masks, ctx_chars, ctx_masks)
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
        """Validate with context"""
        self.model.eval()
        losses, correct, total = [], 0, 0
        
        with torch.no_grad():
            for example in data:
                suf_ids, cat_ids, masks, ctx_chars, ctx_masks, labels, counts = \
                    self._prepare_batch_with_context([example])
                scores = self.model(suf_ids, cat_ids, masks, ctx_chars, ctx_masks)
                
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
        """Full training loop"""
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
    
    def predict(
        self, 
        candidates: List[List[Tuple[int, int]]], 
        context_words: Optional[List[str]] = None
    ) -> Tuple[int, List[float]]:
        """
        Predict best candidate with optional context.
        
        Args:
            candidates: List of suffix chains
            context_words: Optional list of surrounding words
            
        Returns:
            (best_idx, all_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Create example with context
            example = (context_words or [], [], candidates, 0)
            suf_ids, cat_ids, masks, ctx_chars, ctx_masks, _, _ = \
                self._prepare_batch_with_context([example])
            scores = self.model(suf_ids, cat_ids, masks, ctx_chars, ctx_masks)
            
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
    
    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.train_history = ckpt.get('train_history', [])
        self.val_history = ckpt.get('val_history', [])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
