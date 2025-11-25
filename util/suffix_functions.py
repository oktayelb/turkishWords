from typing import List, Tuple


from util.suffixes.v2v_suffixes import VERB2VERB
from util.suffixes.n2v_suffixes import NOUN2VERB
from util.suffixes.n2n_suffixes import NOUN2NOUN
from util.suffixes.v2n_suffixes import VERB2NOUN
import util.word_methods as wrd
from util.rules.suffix_rules import validate_suffix_addition as validate
from util.suffix import Type 
ALL_SUFFIXES = VERB2NOUN + VERB2VERB + NOUN2NOUN + NOUN2VERB 

SUFFIX_TRANSITIONS = {
    'noun': {
        'noun': [s for s in ALL_SUFFIXES if s.comes_to == Type.NOUN and s.makes == Type.NOUN],
        'verb': [s for s in ALL_SUFFIXES if s.comes_to == Type.NOUN and s.makes == Type.VERB]
    },
    'verb': {
        'noun': [s for s in ALL_SUFFIXES if s.comes_to == Type.VERB and s.makes == Type.NOUN],
        'verb': [s for s in ALL_SUFFIXES if s.comes_to == Type.VERB and s.makes == Type.VERB]
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


def find_suffix_chain(word, start_pos, root, visited=None): 
    if visited is None:
        visited = set()
    
    state_key = (len(root), start_pos)
    if state_key in visited:
        return []
    
    visited = visited | {state_key}
    rest = word[len(root):]
    
    if not rest:
        return [([], start_pos)]
    
    if start_pos not in SUFFIX_TRANSITIONS:
        return []
    
    results = []
    for target_pos, suffix_list in SUFFIX_TRANSITIONS[start_pos].items():
        for suffix_obj in suffix_list:
            for suffix_form in suffix_obj.form(root):
                if rest.startswith(suffix_form):
                    next_root = root + suffix_form
                    remaining = rest[len(suffix_form):]
                    subchains = find_suffix_chain(word, target_pos, next_root, visited) if remaining else [([], target_pos)]
                    
                    for chain, final_pos in subchains:
                        if not validate([], suffix_obj):
                            continue
                        
                        if chain and not validate([suffix_obj], chain[0]):
                            continue
                        
                        results.append(([suffix_obj] + chain, final_pos))
                    break
    
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

        chains = (find_suffix_chain(word, "verb", root) +
                find_suffix_chain(word, "noun", root)) if pos == "verb" \
                else find_suffix_chain(word, "noun", root)

        for chain, final_pos in chains:
            analyses.append((root, pos, chain, final_pos))
    
    return analyses