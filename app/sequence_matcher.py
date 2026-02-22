import torch
from typing import List, Dict, Tuple
from ml.ml_ranking_model import build_sentence_sequence

def find_matching_combinations(word_data: List[Dict], target_str: str, trainer) -> Tuple[List[Dict], str, int]:
    """
    Uses a Depth-First Search with STRICT token-based shape matching.
    Returns the scored matches, the furthest matched text prefix, and the index of the word that failed.
    """
    matches = []
    furthest_match_text = ""
    furthest_word_idx = 0
    
    clean_target = " ".join(target_str.replace("(ø)", "").split())
    target_tokens = clean_target.split()
    
    def is_valid_prefix(current_tokens: List[str], t_tokens: List[str]) -> bool:
        if not t_tokens or not current_tokens:
            return True
            
        min_len = min(len(current_tokens), len(t_tokens))
        for i in range(min_len - 1):
            if current_tokens[i] != t_tokens[i]:
                return False
                
        idx = min_len - 1
        if len(t_tokens) > len(current_tokens):
            if current_tokens[idx] != t_tokens[idx]:
                return False
        else:
            if not current_tokens[idx].startswith(t_tokens[idx]):
                return False
        return True

    def dfs(word_idx, current_indices, current_text_parts):
        nonlocal furthest_match_text, furthest_word_idx
        
        full_text = " ".join(current_text_parts).strip()
        clean_full = " ".join(full_text.replace("(ø)", "").split())
        current_tokens = clean_full.split()
        
        if word_idx > furthest_word_idx:
            furthest_word_idx = word_idx
            furthest_match_text = full_text
        elif word_idx == furthest_word_idx and len(current_tokens) > len(" ".join(furthest_match_text.replace("(ø)", "").split()).split()):
            furthest_match_text = full_text

        if word_idx == len(word_data):
            if len(target_tokens) > len(current_tokens):
                return
            if is_valid_prefix(current_tokens, target_tokens):
                matches.append((current_indices, full_text, current_text_parts))
            return
            
        for d_idx, t_str in enumerate(word_data[word_idx]['typing_strings']):
            next_parts = current_text_parts + [t_str]
            next_text = " ".join(next_parts).strip()
            clean_next = " ".join(next_text.replace("(ø)", "").split())
            next_tokens = clean_next.split()
            
            if is_valid_prefix(next_tokens, target_tokens):
                dfs(word_idx + 1, current_indices + [d_idx], next_parts)

    dfs(0, [], [])
    
    scored_matches = []
    trainer.model.eval()
    with torch.no_grad():
        for indices, full_text, parts in matches:
            sentence_chains = [word_data[w_idx]['encoded_chains'][d_idx] for w_idx, d_idx in enumerate(indices)]
            full_s, full_c = build_sentence_sequence(sentence_chains)
            
            if len(full_s) < 2:
                total_score = 0.0
            else:
                s_t, c_t = trainer._to_tensor(full_s, full_c)
                lp = trainer.model.log_probs(s_t, c_t)
                total_score = lp.sum().item()
                
            scored_matches.append({
                'score': total_score,
                'combo_indices': indices,
                'text': full_text,
                'parts': parts
            })
            
    scored_matches.sort(key=lambda x: x['score'], reverse=True)
    return scored_matches, furthest_match_text, furthest_word_idx