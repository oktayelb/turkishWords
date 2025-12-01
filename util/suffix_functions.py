from typing import List, Tuple

from util.suffixes.v2v_suffixes import VERB2VERB
from util.suffixes.n2v_suffixes import NOUN2VERB
from util.suffixes.n2n_suffixes import NOUN2NOUN
from util.suffixes.v2n_suffixes import VERB2NOUN
import util.word_methods as wrd
from util.rules.suffix_rules import validate_suffix_addition as validate
from util.suffix import Type 

ALL_SUFFIXES = VERB2NOUN + VERB2VERB + NOUN2NOUN + NOUN2VERB 

# Updated logic to handle Type.BOTH
SUFFIX_TRANSITIONS = {
    'noun': {
        # Comes to Noun (or Both) AND Makes Noun (or Both)
        'noun': [s for s in ALL_SUFFIXES 
                 if s.comes_to in [Type.NOUN, Type.BOTH] 
                 and s.makes in [Type.NOUN, Type.BOTH]],
        
        # Comes to Noun (or Both) AND Makes Verb (or Both)
        'verb': [s for s in ALL_SUFFIXES 
                 if s.comes_to in [Type.NOUN, Type.BOTH] 
                 and s.makes in [Type.VERB, Type.BOTH]]
    },
    'verb': {
        # Comes to Verb (or Both) AND Makes Noun (or Both)
        'noun': [s for s in ALL_SUFFIXES 
                 if s.comes_to in [Type.VERB, Type.BOTH] 
                 and s.makes in [Type.NOUN, Type.BOTH]],
                 
        # Comes to Verb (or Both) AND Makes Verb (or Both)
        'verb': [s for s in ALL_SUFFIXES 
                 if s.comes_to in [Type.VERB, Type.BOTH] 
                 and s.makes in [Type.VERB, Type.BOTH]]
    }
}

suffix_to_id = {suffix.name: idx for idx, suffix in enumerate(ALL_SUFFIXES)}
id_to_suffix = {idx: suffix.name for idx, suffix in enumerate(ALL_SUFFIXES)}
category_to_id = {'Noun': 0, 'Verb': 1}


def encode_suffix_chain(suffix_objects: List) -> Tuple[List, List]:
    if not suffix_objects:
        return [0], [0]
    
    object_ids = [suffix_to_id.get(s.name, 0) for s in suffix_objects]
    category_ids = [category_to_id.get(s.makes.name, 0) for s in suffix_objects]
    
    return object_ids, category_ids


def find_suffix_chain(word, start_pos, root, current_chain=None, visited=None): 
    """
    Recursive function to find valid suffix chains.
    
    Args:
        word: The complete target word.
        start_pos: The part of speech of the current root ('noun' or 'verb').
        root: The portion of the word constructed so far.
        current_chain: List of suffix objects added so far.
        visited: Set for memoization/cycle prevention.
    """
    if current_chain is None:
        current_chain = []
    if visited is None:
        visited = set()
    
    # We must include the chain history in the visited key because 
    # rule validity depends on the path taken, not just the current position.
    # Using tuple of names for hashability.
    chain_signature = tuple(s.name for s in current_chain)
    state_key = (len(root), start_pos, chain_signature)
    
    if state_key in visited:
        return []
    
    visited.add(state_key)
    
    rest = word[len(root):]
    
    # Base Case: Word is fully consumed
    if not rest:
        return [([], start_pos)]
    
    if start_pos not in SUFFIX_TRANSITIONS:
        return []
    
    results = []
    
    for target_pos, suffix_list in SUFFIX_TRANSITIONS[start_pos].items():
        for suffix_obj in suffix_list:
            
            # --- FIX: Pre-Validation ---
            # Check if adding this suffix to the EXISTING chain is valid.
            # This handles Incompatibility, Sequence rules, OnlyAfter, etc.
            if not validate(current_chain, suffix_obj):
                continue

            # Check if any form of the suffix matches the start of the remaining text
            for suffix_form in suffix_obj.form(root):
                if rest.startswith(suffix_form):
                    next_root = root + suffix_form
                    remaining = rest[len(suffix_form):]
                    
                    # --- RECURSION ---
                    # Pass the UPDATED chain (current_chain + [suffix_obj])
                    if remaining:
                        subchains = find_suffix_chain(
                            word, 
                            target_pos, 
                            next_root, 
                            current_chain + [suffix_obj], 
                            visited
                        )
                    else:
                        subchains = [([], target_pos)]
                    
                    for chain, final_pos in subchains:
                        # Append current suffix to the result chain coming back from recursion
                        results.append(([suffix_obj] + chain, final_pos))
                    
                    # Removed 'break' here to allow handling ambiguity 
                    # (e.g., if two different suffixes produce the same form)

    return results


def decompose(word: str) -> List[Tuple]:
    """Find all possible legal decompositions of a word."""
    if not word:
        return []
    
    analyses = []
    for i in range(1, len(word) + 1):
        root = word[:i]
        exists_status = wrd.exists(root)
        
        if exists_status == 0:
            continue
        
        pos = "verb" if wrd.can_be_verb(root) else "noun"

        # Note: We don't need to pass current_chain here, it defaults to []
        chains = (find_suffix_chain(word, "verb", root) +
                  find_suffix_chain(word, "noun", root)) if pos == "verb" \
                  else find_suffix_chain(word, "noun", root)

        for chain, final_pos in chains:
            analyses.append((root, pos, chain, final_pos))
    
    return analyses