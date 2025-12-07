from typing import List, Tuple

# Mevcut importlarınız...
from util.suffixes.v2v_suffixes import VERB2VERB
from util.suffixes.n2v_suffixes import NOUN2VERB
from util.suffixes.n2n_suffixes import NOUN2NOUN
from util.suffixes.v2n_suffixes import VERB2NOUN
import util.word_methods as wrd
from util.suffix import Type, Suffix, SuffixGroup

ALL_SUFFIXES = VERB2NOUN + VERB2VERB + NOUN2NOUN + NOUN2VERB 

# Suffix transition kurallarınız
SUFFIX_TRANSITIONS = {
    'noun': {
        'noun': [s for s in ALL_SUFFIXES 
                 if s.comes_to in [Type.NOUN, Type.BOTH] 
                 and s.makes in [Type.NOUN, Type.BOTH]],
        'verb': [s for s in ALL_SUFFIXES 
                 if s.comes_to in [Type.NOUN, Type.BOTH] 
                 and s.makes in [Type.VERB, Type.BOTH]]
    },
    'verb': {
        'noun': [s for s in ALL_SUFFIXES 
                 if s.comes_to in [Type.VERB, Type.BOTH] 
                 and s.makes in [Type.NOUN, Type.BOTH]],
        'verb': [s for s in ALL_SUFFIXES 
                 if s.comes_to in [Type.VERB, Type.BOTH] 
                 and s.makes in [Type.VERB, Type.BOTH]]
    }
}

# --- 1. HIERARCHY STATE MACHINE ---

def is_valid_transition(last_suffix: Suffix, next_suffix: Suffix) -> bool:
    """
    Implements the State Machine logic for Turkish Agglutination.
    
    1. Waterfall Principle: Generally next_group >= last_group.
    2. Derivational Locking: Group 15 can ONLY jump to Group >= 50.
    3. Post-Case Loop: Group 45 (-ki) is allowed to loop back to Group 10.
    """
    last_g = last_suffix.group
    next_g = next_suffix.group

    # --- RULE 1: The Post-Case Loop (-ki Exception) ---
    # Exception: Post-Case (45) can loop back to Derivational (10)
    # Example: Ev-de-ki-ler (40 -> 45 -> 10)
    if last_g == SuffixGroup.POST_CASE and next_g == SuffixGroup.DERIVATIONAL:
        return True

    # --- RULE 2: Derivational Locking (The Valve) ---
    # Exception: If we are in Locking (15), we MUST skip Possessive(30) and Case(40).
    # We can only go to Predicative(50) or Terminal(60).
    if last_g == SuffixGroup.DERIVATIONAL_LOCKING:
        if next_g < SuffixGroup.PREDICATIVE:
            return False

    # --- RULE 3: The Waterfall (Gravity) ---
    # You cannot go upstream (e.g., Case 40 cannot go to Possessive 30)
    if next_g < last_g:
        return False

    # --- RULE 4: Self-Looping Constraints ---
    if next_g == last_g:
        # Allowed to chain: Derivational (10) and Predicative (50)
        # Example: Göz-lük-çü (10->10), Gel-di-yse (50->50)
        if last_g in [SuffixGroup.DERIVATIONAL, SuffixGroup.PREDICATIVE, SuffixGroup.TERMINAL]:
            return True
        
        # Not allowed to chain: Possessive (30), Case (40), Locking (15), PostCase (45)
        # Example: Ev-im-im (Invalid), Ev-de-den (Invalid)
        return False

    return True


# --- 2. DEFINITIONS ---

def pekistirme_dummy_form(word, suffix_obj):
    return ["dummy_form"]
pekistirme_suffix = Suffix("pekistirme", "pekistirme", Type.BOTH, Type.BOTH, form_function=pekistirme_dummy_form, is_unique=True)

def find_suffix_chain(word, start_pos, root, current_chain=None, visited=None): 
    """
    Recursive suffix chain finder utilizing the Hierarchy State Machine.
    """
    if current_chain is None:
        current_chain = []
    if visited is None:
        visited = set()
    
    chain_signature = tuple(s.name for s in current_chain)
    state_key = (len(root), start_pos, chain_signature)
    
    if state_key in visited:
        return []
    visited.add(state_key)
    
    rest = word[len(root):]
    
    # Base Case
    if not rest:
        return [([], start_pos)]
    
    if start_pos not in SUFFIX_TRANSITIONS:
        return []
    
    results = []
    iyor_variations = ['iyor', 'ıyor', 'uyor', 'üyor']

    for target_pos, suffix_list in SUFFIX_TRANSITIONS[start_pos].items():
        for suffix_obj in suffix_list:
            
            # --- HIERARCHY VALIDATION ---
            if current_chain:
                last_suffix = current_chain[-1]
                if not is_valid_transition(last_suffix, suffix_obj):
                    continue
            
            # --- UNIQUENESS CHECK ---
            if suffix_obj.is_unique:
                if any(s.name == suffix_obj.name for s in current_chain):
                    continue
            
            # --- FORM GENERATION ---
            suffix_forms = suffix_obj.form(root)
            
            for suffix_form in suffix_forms:
                
                # --- MATCH TYPE 1: Standard ---
                if rest.startswith(suffix_form):
                    next_root = root + suffix_form
                    subchains = find_suffix_chain(
                        word, 
                        target_pos, 
                        next_root, 
                        current_chain + [suffix_obj], 
                        visited
                    )
                    for chain, final_pos in subchains:
                        results.append(([suffix_obj] + chain, final_pos))

                # --- MATCH TYPE 2: Vowel Narrowing (Daralma) ---
                elif len(suffix_form) > 0 and suffix_form[-1] in ['a', 'e']:
                    narrowed_form = suffix_form[:-1] 
                    if rest.startswith(narrowed_form):
                        rest_after_narrowing = rest[len(narrowed_form):]
                        if any(rest_after_narrowing.startswith(v) for v in iyor_variations):
                            next_root = root + suffix_form 
                            repaired_word = root + suffix_form + rest_after_narrowing
                            
                            subchains = find_suffix_chain(
                                repaired_word, 
                                target_pos, 
                                next_root, 
                                current_chain + [suffix_obj], 
                                visited
                            )
                            for chain, final_pos in subchains:
                                results.append(([suffix_obj] + chain, final_pos))
                    
    return results


def get_root_candidates(surface_root: str) -> List[Tuple[str, str]]:
    """
    Analyzes the text segment and returns (Surface Form, Dictionary Lemma).
    """
    candidates = [] 
    
    def is_new_candidate(lemma):
        return not any(cand[1] == lemma for cand in candidates)

    if wrd.exists(surface_root):
        candidates.append((surface_root, surface_root))
        
    if not surface_root:
        return candidates

    last_char = surface_root[-1]
    
    def get_unsoftened_char(char, text_ending):
        if char == 'b': return 'p'
        if char == 'c': return 'ç'
        if char == 'd': return 't'
        if char == 'ğ': return 'k'
        if char == 'g' and text_ending.endswith("ng"): return 'nk'
        return None

    # Yumuşama
    target_char = get_unsoftened_char(last_char, surface_root)
    if target_char:
        if target_char == 'nk': 
            candidate_lemma = surface_root[:-2] + 'nk'
        else:
            candidate_lemma = surface_root[:-1] + target_char
            
        if wrd.exists(candidate_lemma) and is_new_candidate(candidate_lemma):
            candidates.append((surface_root, candidate_lemma))

    # Ünlü Düşmesi
    if len(surface_root) >= 2 and wrd.ends_with_consonant(surface_root):
        prefix = surface_root[:-1]
        suffix_char = surface_root[-1]
        
        narrow_vowels = ['ı', 'i', 'u', 'ü']
        for vowel in narrow_vowels:
            restored_lemma = prefix + vowel + suffix_char 
            
            if wrd.exists(restored_lemma) and is_new_candidate(restored_lemma):
                candidates.append((surface_root, restored_lemma))
            
            last_char_of_restored = restored_lemma[-1]
            unsoftened_char = get_unsoftened_char(last_char_of_restored, restored_lemma)
            
            if unsoftened_char:
                if unsoftened_char == 'nk':
                    restored_unsoftened = restored_lemma[:-2] + 'nk'
                else:
                    restored_unsoftened = restored_lemma[:-1] + unsoftened_char
                
                if wrd.exists(restored_unsoftened) and is_new_candidate(restored_unsoftened):
                    candidates.append((surface_root, restored_unsoftened))

    # Vowel Narrowing at Root
    if not wrd.exists(surface_root) and len(surface_root) > 1:
        for terminal_vowel in ['a', 'e']:
            restored_lemma = surface_root + terminal_vowel
            if wrd.exists(restored_lemma) and is_new_candidate(restored_lemma):
                 candidates.append((surface_root, restored_lemma))

    return candidates

# --- 3. PEKISTIRME ---
def check_pekistirme(word):
    if len(word) < 4:
        return []
    
    pekistirme_letters = "psrm"
    first_vowel_index = -1
    
    for i in range(len(word)):
        if word[i] in wrd.VOWELS:
            first_vowel_index = i
            break
            
    if first_vowel_index == -1: 
        return []

    if first_vowel_index + 1 >= len(word):
        return []

    if word[first_vowel_index+1] not in pekistirme_letters:  
        return []
    
    # Case 1: Standard (mas-mavi)
    split_index_1 = first_vowel_index + 2
    if split_index_1 < len(word):
        potential_rest = word[split_index_1:]
        prefix = word[:split_index_1]
        
        if potential_rest.startswith(word[:first_vowel_index+1]):
             for k in range(len(potential_rest), 1, -1):
                 candidate_root = potential_rest[:k]
                 if wrd.exists(candidate_root):
                     return [prefix, candidate_root]

    # Case 2: Extended (gü-p-e-gündüz)
    split_index_2 = first_vowel_index + 3
    if split_index_2 < len(word):
        if word[first_vowel_index+2] in ['a', 'e']:
            potential_rest = word[split_index_2:]
            prefix = word[:split_index_2]
            
            if potential_rest.startswith(word[:first_vowel_index+1]):
                for k in range(len(potential_rest), 1, -1):
                    candidate_root = potential_rest[:k]
                    if wrd.exists(candidate_root):
                        return [prefix, candidate_root]
    
    return []

def decompose(word: str) -> List[Tuple]:
    """Find all possible legal decompositions of a word."""
    if not word:
        return []
    
    analyses = []
    
    # --- PEKISTIRME ---
    pekistirme_data = check_pekistirme(word) 
    
    if pekistirme_data:
        prefix_str = pekistirme_data[0]
        root_str = pekistirme_data[1]
        
        if len(prefix_str + root_str) == len(word):
            analyses.append((root_str, "noun", [pekistirme_suffix], "noun"))
        else:
            virtual_word_part = word[len(prefix_str):] 
            chains = find_suffix_chain(virtual_word_part, "noun", root_str)
            for chain, final_pos in chains:
                full_chain = [pekistirme_suffix] + chain
                analyses.append((root_str, "noun", full_chain, final_pos))

    # --- STANDARD ---
    for i in range(1, len(word) + 1):
        surface_part = word[:i] 
        root_pairs = get_root_candidates(surface_part)
        
        if not root_pairs:
            continue
            
        for surface_root, lemma_root in root_pairs:
            virtual_word = lemma_root + word[len(surface_root):]
            pos = "noun"
            noun_chains = []
            verb_chains = []
            if(wrd.exists(lemma_root)):
                noun_chains = find_suffix_chain(virtual_word, "noun", lemma_root)
            if(wrd.can_be_verb(lemma_root)):
                verb_chains = find_suffix_chain(virtual_word, "verb", lemma_root)

            chains = noun_chains + verb_chains
            for chain, final_pos in chains:
                analyses.append((lemma_root, pos, chain, final_pos))

    return analyses