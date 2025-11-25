"""
Context-Aware Machine Learning Ranking Model for Turkish Morphological Decomposition

Key improvements:
    - Bidirectional context encoding (words before and after target)
    - Attention mechanism between context and candidate decompositions
    - Character-level word embeddings for handling OOV words
    - Dual-encoder architecture: context encoder + decomposition encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class CharCNN(nn.Module):
    """Character-level CNN for word embeddings"""
    
    def __init__(self, vocab_size: int = 256, embed_dim: int = 128):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, 32)
        
        # Multiple filter sizes for n-gram features
        self.convs = nn.ModuleList([
            nn.Conv1d(32, embed_dim // 3, kernel_size=k, padding=k//2)
            for k in [3, 4, 5]
        ])
        
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (batch, seq_len, max_word_len)
        Returns:
            word_embeds: (batch, seq_len, embed_dim)
        """
        B, L, W = char_ids.shape
        
        # Flatten for batch processing
        char_ids = char_ids.view(B * L, W)
        x = self.char_embed(char_ids)  # (B*L, W, 32)
        x = x.transpose(1, 2)  # (B*L, 32, W)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (B*L, embed_dim//3, W)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate and project
        combined = torch.cat(conv_outputs, dim=1)  # (B*L, embed_dim)
        combined = self.projection(combined)
        
        return combined.view(B, L, -1)


class ContextEncoder(nn.Module):
    """Encodes surrounding context words"""
    
    def __init__(self, embed_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.char_cnn = CharCNN(embed_dim=embed_dim)
        
        self.encoder = nn.LSTM(
            embed_dim, 
            embed_dim // 2, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, context_chars: torch.Tensor, 
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            context_chars: (batch, context_len, max_char_len)
            context_mask: (batch, context_len) - True for valid words
        Returns:
            context_repr: (batch, embed_dim)
        """
        # Get character-level word embeddings
        word_embeds = self.char_cnn(context_chars)  # (batch, context_len, embed_dim)
        
        # Encode with LSTM
        encoded, _ = self.encoder(word_embeds)  # (batch, context_len, embed_dim)
        encoded = self.layer_norm(encoded)
        
        # Pool over context (mask-aware)
        if context_mask is not None:
            mask_expanded = context_mask.unsqueeze(-1).float()
            encoded = encoded * mask_expanded
            context_repr = encoded.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            context_repr = encoded.mean(dim=1)
        
        return context_repr


class Ranker(nn.Module):
    """Context-aware ranking model for morphological decomposition"""
    
    def __init__(
        self, 
        suffix_vocab_size: int,
        num_categories: int = 2,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_context: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_context = use_context
        
        # Decomposition encoder (same as before)
        self.suffix_embed = nn.Embedding(suffix_vocab_size + 1, embed_dim, padding_idx=0)
        self.category_embed = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(50, embed_dim)
        self.input_proj = nn.Linear(embed_dim * 3, embed_dim)
        
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.decomp_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Context encoder
        if use_context:
            self.context_encoder = ContextEncoder(embed_dim=embed_dim)
            
            # Cross-attention: decomposition attends to context
            self.cross_attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            
            # Gating mechanism to blend context and decomposition
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
        
        # Scoring head
        scorer_input_dim = embed_dim * 2 if use_context else embed_dim
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) if 'embed' in name else nn.init.kaiming_normal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(
        self, 
        suffix_ids: torch.Tensor,
        category_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context_chars: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            suffix_ids: (batch, seq_len) - suffix IDs for decomposition
            category_ids: (batch, seq_len) - category IDs
            mask: (batch, seq_len) - padding mask for decomposition
            context_chars: (batch, context_len, max_char_len) - character IDs of context words
            context_mask: (batch, context_len) - valid context words mask
        Returns:
            scores: (batch,)
        """
        B, L = suffix_ids.shape
        
        # Encode decomposition
        pos = torch.arange(L, device=suffix_ids.device).unsqueeze(0).expand(B, L)
        x = torch.cat([
            self.suffix_embed(suffix_ids),
            self.category_embed(category_ids),
            self.pos_embed(pos)
        ], dim=-1)
        x = self.input_proj(x)
        x = self.decomp_encoder(x, src_key_padding_mask=mask)
        
        # Pool decomposition representation
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            decomp_pooled = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            decomp_pooled = x.mean(dim=1)
        
        # If context is provided, incorporate it
        if self.use_context and context_chars is not None:
            # Encode context
            context_repr = self.context_encoder(context_chars, context_mask)
            
            # Cross-attention: let decomposition attend to context
            context_repr_expanded = context_repr.unsqueeze(1)  # (batch, 1, embed_dim)
            attended, _ = self.cross_attention(
                decomp_pooled.unsqueeze(1),
                context_repr_expanded,
                context_repr_expanded
            )
            attended = attended.squeeze(1)
            
            # Gate mechanism: how much to trust context
            gate_input = torch.cat([decomp_pooled, context_repr], dim=-1)
            gate_value = self.gate(gate_input)
            
            # Combine with gating
            combined = gate_value * attended + (1 - gate_value) * decomp_pooled
            
            # Concatenate for final scoring
            final_repr = torch.cat([combined, context_repr], dim=-1)
        else:
            final_repr = decomp_pooled
        
        return self.scorer(final_repr).squeeze(-1)


# Helper function to convert words to character IDs
def words_to_char_ids(words: List[str], max_word_len: int = 20) -> torch.Tensor:
    """
    Convert words to character ID tensors.
    
    Args:
        words: List of words
        max_word_len: Maximum characters per word
    
    Returns:
        char_ids: (len(words), max_word_len) tensor
    """
    char_ids = []
    for word in words:
        # Convert to lowercase and get char codes (limited to ASCII range)
        chars = [min(ord(c), 255) for c in word[:max_word_len]]
        # Pad to max_word_len
        chars += [0] * (max_word_len - len(chars))
        char_ids.append(chars)
    
    return torch.tensor(char_ids, dtype=torch.long)
