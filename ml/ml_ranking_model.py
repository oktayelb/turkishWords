import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from .config import config  # Direct import from sibling file

# ============================================================================
# SPECIAL TOKENS
# ============================================================================
#
# Token ID 0 is reserved as padding (also used by Embedding padding_idx).
# The vocabulary is laid out as:
#   [0]            → PAD
#   [1]            → WORD_SEP  (boundary between words in a sentence)
#   [2 .. V+1]     → suffix IDs  (suffix_idx + 2, where suffix_idx is 0-based)
#
# Category IDs:
#   0 → Noun, 1 → Verb, 2 → SPECIAL (for PAD and WORD_SEP tokens)

SPECIAL_PAD      = 0
SPECIAL_WORD_SEP = 1
SUFFIX_OFFSET    = 2          # suffix IDs start here
CATEGORY_SPECIAL = 2          # category ID for PAD / WORD_SEP


# ============================================================================
# HELPER: encode / decode sentence-level token sequences
# ============================================================================

def encode_chain(suffix_chain) -> List[Tuple[int, int]]:
    """
    Convert a List[Suffix] → List[(suffix_token_id, category_id)].
    An empty chain (bare root) returns an empty list; the caller must
    still emit a WORD_SEP token.
    """
    from ml.ml_ranking_model import SUFFIX_OFFSET, CATEGORY_SPECIAL  # avoid circular at module level

    suffix_to_id = {
        suffix.name: idx + SUFFIX_OFFSET
        for idx, suffix in enumerate(_get_all_suffixes())
    }
    category_to_id = {'Noun': 0, 'Verb': 1}

    encoded = []
    for s in suffix_chain:
        sid  = suffix_to_id.get(s.name, SUFFIX_OFFSET)  # unknown → first real suffix
        cid  = category_to_id.get(s.makes.name, 0)
        encoded.append((sid, cid))
    return encoded


def _get_all_suffixes():
    """Lazy import to avoid circular deps."""
    import util.decomposer as sfx
    return sfx.ALL_SUFFIXES


def build_sentence_sequence(
    word_chains: List[List[Tuple[int, int]]]
) -> Tuple[List[int], List[int]]:
    """
    Flatten a list of encoded chains (one per word) into a single
    (suffix_ids, category_ids) pair, separated by WORD_SEP tokens.

    Layout for a 2-word sentence  [w1_suf1, w1_suf2 | SEP | w2_suf1 | SEP]:
        suffix_ids   = [w1_suf1_id, w1_suf2_id, WORD_SEP, w2_suf1_id, WORD_SEP]
        category_ids = [w1_cat1,    w1_cat2,    C_SPEC,  w2_cat1,    C_SPEC]

    Each word ends with a WORD_SEP so the model learns to predict the first
    suffix of the next word given all previous context.
    """
    suffix_ids:   List[int] = []
    category_ids: List[int] = []

    for chain in word_chains:
        for (sid, cid) in chain:
            suffix_ids.append(sid)
            category_ids.append(cid)
        # word boundary marker
        suffix_ids.append(SPECIAL_WORD_SEP)
        category_ids.append(CATEGORY_SPECIAL)

    return suffix_ids, category_ids


# ============================================================================
# MODEL
# ============================================================================

class SentenceDisambiguator(nn.Module):
    """
    Causal (decoder-only) Transformer that models the joint distribution
    over suffix token sequences at the sentence level.

    Trained like a language model: given a confirmed sentence decomposition
    as a flat token sequence, minimise cross-entropy of predicting each next
    token from all previous ones.

    At inference time, candidate decompositions for each word are scored by
    summing the log-probabilities assigned to their tokens in context, so
    every word's disambiguation is informed by all surrounding words.
    """

    def __init__(self, suffix_vocab_size: int):
        """
        Args:
            suffix_vocab_size: number of real suffix types (from ALL_SUFFIXES).
                               Total token vocab = suffix_vocab_size + SUFFIX_OFFSET
                               (PAD + WORD_SEP + all suffixes).
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        # Full token vocab: PAD(0), WORD_SEP(1), then real suffixes starting at 2
        self.vocab_size = suffix_vocab_size + SUFFIX_OFFSET

        # Token embeddings
        self.suffix_embed   = nn.Embedding(self.vocab_size,          self.embed_dim, padding_idx=SPECIAL_PAD)
        # Category embedding: 0=Noun, 1=Verb, 2=Special  →  3 categories
        self.category_embed = nn.Embedding(3,                         self.embed_dim)
        # Positional embedding (up to 512 tokens per sentence)
        self.pos_embed      = nn.Embedding(512,                       self.embed_dim)

        # Project concatenated embeddings → model dim
        self.input_proj = nn.Linear(self.embed_dim * 3, self.embed_dim)

        # Causal (decoder) transformer layers
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)

        # Language-model head: hidden → vocab logits
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        # Tie weights (token embedding ↔ LM head), standard LM trick
        self.lm_head.weight = self.suffix_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'embed' not in name:
                nn.init.kaiming_normal_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position i can only attend to 0..i."""
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        suffix_ids:   torch.Tensor,   # (B, L)
        category_ids: torch.Tensor,   # (B, L)
        pad_mask:     Optional[torch.Tensor] = None,  # (B, L) True = padding
    ) -> torch.Tensor:
        """
        Returns logits of shape (B, L, vocab_size).
        Logits[b, i, :] = distribution over the token at position i+1
        given positions 0..i  (standard causal LM).
        """
        B, L = suffix_ids.shape
        pos = torch.arange(L, device=suffix_ids.device).unsqueeze(0).expand(B, L)

        x = torch.cat([
            self.suffix_embed(suffix_ids),
            self.category_embed(category_ids),
            self.pos_embed(pos),
        ], dim=-1)                          # (B, L, embed_dim * 3)

        x = self.input_proj(x)              # (B, L, embed_dim)

        causal = self._causal_mask(L, suffix_ids.device)

        x = self.transformer(x, mask=causal, src_key_padding_mask=pad_mask)

        return self.lm_head(x)              # (B, L, vocab_size)

    def log_probs(
        self,
        suffix_ids:   torch.Tensor,   # (B, L)
        category_ids: torch.Tensor,   # (B, L)
        pad_mask:     Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        """
        Returns token-level log-probs of shape (B, L-1):
            log P(token[i] | token[0..i-1])   for i in 1..L-1.

        Used during scoring to evaluate candidate decompositions.
        """
        logits = self.forward(suffix_ids, category_ids, pad_mask=pad_mask)  # (B, L, V)
        log_p  = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, L-1, V)
        targets = suffix_ids[:, 1:]                          # (B, L-1)
        token_log_probs = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

        if pad_mask is not None:
            # Mask out padding tokens in the target
            target_mask = pad_mask[:, 1:]
            token_log_probs.masked_fill_(target_mask, 0.0)

        return token_log_probs


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """
    Wraps SentenceDisambiguator and handles:
      - Causal LM training on confirmed sentence decompositions
      - Context-aware candidate scoring at inference time
      - Checkpointing
    """

    def __init__(self, model: SentenceDisambiguator):
        self.model = model

        self.checkpoint_frequency = config.checkpoint_frequency
        self.batch_size           = config.batch_size
        self.patience             = config.patience
        self.path                 = str(config.model_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=config.learning_rate * 0.01
        )

        # Training state
        self.train_history: List[float] = []
        self.val_history:   List[float] = []
        self.best_val_loss  = float('inf')
        self.global_step    = 0

        try:
            self.load_checkpoint(self.path)
            print(f"Loaded model from {self.path}")
        except FileNotFoundError:
            print(f"Starting fresh (no checkpoint found at {self.path})")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(
        self, suffix_ids: List[int], category_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert flat id lists to (1, L) tensors on device."""
        s = torch.tensor(suffix_ids,   dtype=torch.long, device=self.device).unsqueeze(0)
        c = torch.tensor(category_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return s, c

    def _get_best_index(self, scores: List[float]) -> int:
        """Helper to pick the best score, explicitly ignoring 0.0 to prevent bare-root dominance."""
        valid_indices = [i for i, s in enumerate(scores) if s != 0.0]
        if valid_indices:
            return int(max(valid_indices, key=lambda i: scores[i]))
        # Fallback if somehow everything is 0.0
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_sentence(
        self,
        word_chains: List[List[Tuple[int, int]]],
        max_retries: int = 20,
    ) -> float:
        """
        Train on a confirmed sentence decomposition.

        Args:
            word_chains: one encoded chain per word in the sentence
                         (from Translation.encode_suffix_chain).
                         Single-word training passes a list with one entry.
        Returns:
            final cross-entropy loss (float).
        """
        # Build flat token sequence for the whole sentence
        suffix_ids, category_ids = build_sentence_sequence(word_chains)

        if len(suffix_ids) < 2:
            # Nothing to train on (bare root with no suffixes → 1 token = just SEP)
            return 0.0

        s_tensor, c_tensor = self._to_tensor(suffix_ids, category_ids)

        # Target for causal LM: predict token[i] from tokens[0..i-1]
        # input = tokens[:-1],  target = tokens[1:]
        input_s = s_tensor[:, :-1]
        input_c = c_tensor[:, :-1]
        target  = s_tensor[:, 1:]    # (1, L-1)

        self.model.train()
        final_loss = 0.0

        print(f"   Learning sentence context...", end="", flush=True)

        for attempt in range(max_retries):
            logits = self.model(input_s, input_c)            # (1, L-1, V)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=SPECIAL_PAD,
            )
            final_loss = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.global_step += 1

            # Early stopping: stop once loss is low enough
            if final_loss < 0.05:
                print(f" Done in {attempt + 1} steps (loss={final_loss:.4f}).")
                break
            if attempt % 5 == 0:
                print(".", end="", flush=True)
        else:
            print(f" Limit reached ({max_retries}), loss={final_loss:.4f}.")

        self.scheduler.step()
        self.train_history.append(final_loss)
        return final_loss

    # Keep the old name as an alias so call-sites in interactive_trainer.py
    # that still use train_persistent() keep working during migration.
    def train_persistent(
        self,
        training_data: List[Tuple],   # old-style: [([], encoded_chains, correct_idx), ...]
        max_retries: int = 20,
    ) -> float:
        """
        Backward-compatible wrapper around train_sentence().

        Converts old-style ([], encoded_chains, correct_idx) tuples into a
        flat sentence sequence using only the confirmed chains, then trains.
        """
        confirmed_chains = []
        for (_, candidates, correct_idx) in training_data:
            if correct_idx < len(candidates):
                confirmed_chains.append(candidates[correct_idx])
            elif candidates:
                confirmed_chains.append(candidates[0])
            else:
                confirmed_chains.append([])

        return self.train_sentence(confirmed_chains, max_retries=max_retries)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        context_chains: List[List[Tuple[int, int]]],   # already-committed words (left context)
        candidates:     List[List[Tuple[int, int]]],   # chains to score for the current word
        right_chains:   Optional[List[List[Tuple[int, int]]]] = None,  # future words (optional)
    ) -> List[float]:
        """
        Score each candidate chain for the *current* word given sentence context.

        For each candidate we build:
            context_tokens  +  candidate_tokens  +  WORD_SEP
        and sum the log-probabilities of the candidate tokens (and their SEP)
        conditioned on the context.  Higher is better.

        Args:
            context_chains: encoded chains for words already chosen (left).
            candidates:     encoded chains to rank for the current word.
            right_chains:   encoded chains for future words (right context).
                            If provided, they are appended after the candidate
                            so the model has bidirectional context during scoring.
                            (This requires the right context to already be committed,
                            e.g. in a re-ranking pass.)
        Returns:
            List of log-prob scores, one per candidate.
        """
        self.model.eval()

        # Build the fixed left-context sequence
        ctx_suffix, ctx_cat = build_sentence_sequence(context_chains) if context_chains else ([], [])

        # Right context (fixed suffix, used to give bidirectional signal)
        if right_chains:
            right_s, right_c = build_sentence_sequence(right_chains)
        else:
            right_s, right_c = [], []

        scores = []
        with torch.no_grad():
            for chain in candidates:
                # Candidate token sequence + SEP
                cand_s, cand_c = build_sentence_sequence([chain])  # ends with WORD_SEP

                # Full sequence: left_ctx | candidate | right_ctx
                full_s = ctx_suffix + cand_s + right_s
                full_c = ctx_cat   + cand_c  + right_c

                if len(full_s) < 2:
                    scores.append(0.0)
                    continue

                s_t, c_t = self._to_tensor(full_s, full_c)

                # Log-probs over positions 1..L-1
                lp = self.model.log_probs(s_t, c_t)   # (1, L-1)

                # We care only about the candidate tokens (index ctx_len .. ctx_len+cand_len-1
                # in the *prediction* dimension, which is shifted by 1)
                ctx_len  = len(ctx_suffix)
                cand_len = len(cand_s)

                # Positions in log_prob tensor: lp[:, i] = log P(token[i+1] | token[0..i])
                # Candidate tokens are at positions ctx_len .. ctx_len+cand_len-1 in full_s
                # → they are predicted by lp at indices ctx_len-1 .. ctx_len+cand_len-2
                start = max(0, ctx_len - 1)
                end   = ctx_len + cand_len - 1  # exclusive

                if end <= start or end > lp.shape[1]:
                    scores.append(lp.sum().item())  # fallback: sum everything
                else:
                    scores.append(lp[0, start:end].sum().item())

        return scores

    def fast_batch_predict(
        self,
        all_candidates: List[List[List[Tuple[int, int]]]],
        batch_size: int = 512
    ) -> List[int]:
        """
        Evaluates a large list of words, each having multiple candidate chains,
        using padded GPU batches to maximize throughput.
        Returns a list of best_indices (one per word).
        """
        self.model.eval()

        # Flatten into one list of jobs: (word_idx, cand_idx, suffix_seq, cat_seq)
        flat_jobs = []
        for w_idx, candidates in enumerate(all_candidates):
            for c_idx, chain in enumerate(candidates):
                cand_s, cand_c = build_sentence_sequence([chain])
                flat_jobs.append((w_idx, c_idx, cand_s, cand_c))

        if not flat_jobs:
            return []

        # w_idx -> list of scores (same order as candidates)
        scores_map = {w_idx: [] for w_idx in range(len(all_candidates))}

        with torch.no_grad():
            for i in range(0, len(flat_jobs), batch_size):
                batch = flat_jobs[i:i + batch_size]
                max_len = max(len(job[2]) for job in batch)

                if max_len < 2:
                    for job in batch:
                        scores_map[job[0]].append(0.0)
                    continue

                bsz = len(batch)
                
                # Allocate tensors
                s_t = torch.full((bsz, max_len), SPECIAL_PAD, dtype=torch.long, device=self.device)
                c_t = torch.full((bsz, max_len), CATEGORY_SPECIAL, dtype=torch.long, device=self.device)
                p_mask = torch.ones((bsz, max_len), dtype=torch.bool, device=self.device) # True = pad

                for b_idx, (_, _, seq_s, seq_c) in enumerate(batch):
                    seq_len = len(seq_s)
                    s_t[b_idx, :seq_len] = torch.tensor(seq_s, dtype=torch.long, device=self.device)
                    c_t[b_idx, :seq_len] = torch.tensor(seq_c, dtype=torch.long, device=self.device)
                    p_mask[b_idx, :seq_len] = False

                lp = self.model.log_probs(s_t, c_t, pad_mask=p_mask) # (bsz, max_len - 1)

                # Sum the log probabilities for each sequence
                sums = lp.sum(dim=1).tolist()

                for b_idx, job in enumerate(batch):
                    scores_map[job[0]].append(sums[b_idx])

        # Pick best index per word
        best_indices = []
        for w_idx in range(len(all_candidates)):
            c_scores = scores_map[w_idx]
            if not c_scores:
                best_indices.append(0)
            else:
                best_indices.append(self._get_best_index(c_scores))

        return best_indices

    def predict(
        self,
        candidates: List[List[Tuple[int, int]]],
        context_chains: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Tuple[int, List[float]]:
        """
        Pick the best candidate for a single word (with optional left context).

        Returns: (best_index, all_scores)
        """
        ctx = context_chains or []
        scores = self.score_candidates(ctx, candidates)
        best = self._get_best_index(scores)
        return best, scores

    def batch_predict(
        self,
        batch_candidates: List[List[List[Tuple[int, int]]]],
    ) -> List[Tuple[int, List[float]]]:
        """
        Score candidates for multiple words independently (no cross-word context).
        Used for initial ranking before the user has made any choices.

        For context-aware sentence-level ranking, use sentence_predict() instead.
        """
        results = []
        for candidates in batch_candidates:
            best_idx, scores = self.predict(candidates)
            results.append((best_idx, scores))
        return results

    def sentence_predict(
        self,
        all_candidates: List[List[List[Tuple[int, int]]]],
    ) -> List[Tuple[int, List[float]]]:
        """
        Greedy left-to-right sentence-level disambiguation.

        For each word in order, score its candidates given all previously
        committed choices as left context, then commit the winner.

        Returns: list of (best_idx, scores) per word.
        """
        committed: List[List[Tuple[int, int]]] = []
        results: List[Tuple[int, List[float]]] = []

        for candidates in all_candidates:
            scores  = self.score_candidates(committed, candidates)
            best    = self._get_best_index(scores)
            results.append((best, scores))
            committed.append(candidates[best])

        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self):
        torch.save({
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'train_history':   self.train_history,
            'val_history':     self.val_history,
            'best_val_loss':   self.best_val_loss,
            'global_step':     self.global_step,
        }, self.path)
        print(f"Saved to {self.path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.train_history = ckpt.get('train_history', [])
        self.val_history   = ckpt.get('val_history',   [])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.global_step   = ckpt.get('global_step',   0)
        print(f"Loaded from {path} (step {self.global_step})")