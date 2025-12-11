from functools import lru_cache
from typing import List, Tuple, Set 

# ============================================================================
# IMPORTS
# ============================================================================
from util.suffixes.v2v_suffixes import VERB2VERB
from util.suffixes.n2v_suffixes import NOUN2VERB
from util.suffixes.n2n_suffixes import NOUN2NOUN
from util.suffixes.v2n_suffixes import VERB2NOUN
import util.word_methods as wrd
from util.suffix import Type, Suffix, SuffixGroup

ALL_SUFFIXES = VERB2NOUN + VERB2VERB + NOUN2NOUN + NOUN2VERB 

# ============================================================================
# SUFFIX TYPES
# ============================================================================
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

# ============================================================================
# SUFFIX HIERARCHY RULES
# ============================================================================
def is_valid_transition(last_suffix: Suffix, next_suffix: Suffix) -> bool:
    """
    Implements the State Machine logic for Turkish Agglutination.
    """
    last_g = last_suffix.group
    next_g = next_suffix.group

    # --- RULE 1: The Post-Case Loop (-ki Exception) ---
    if last_g == SuffixGroup.POST_CASE and next_g <= SuffixGroup.POST_CASE:
        return True

    # isim tamlamasından sonra ek gelmez
    if last_g == SuffixGroup.COMPOUND:
        return False
        
    # isim tamlaması yalnızca yalın isim, pol ve kişi ekli ve ki ekli isimlere gelebilir
    if next_g == SuffixGroup.COMPOUND and not (last_g == SuffixGroup.DERIVATIONAL or 
                                             last_g == SuffixGroup.POST_CASE or 
                                             last_g == SuffixGroup.POSSESSIVE):
        return False
        
    # --- RULE 2: Derivational Locking (The Valve) ---
    if last_g == SuffixGroup.DERIVATIONAL_LOCKING:
        if next_g < SuffixGroup.PREDICATIVE:
            return False

    # --- RULE 3: The Waterfall (Gravity) ---
    if next_g < last_g:
        return False
    
    # copula gibi eklerden sonra kişi çekim gelmeli
    if last_g == SuffixGroup.PREDICATIVE and next_g != SuffixGroup.CONJUGATION:
        return False

    # --- RULE 4: Self-Looping Constraints ---
    if next_g == last_g:
        if last_g in [SuffixGroup.DERIVATIONAL, SuffixGroup.PREDICATIVE]:
            return True
        return False

    return True


# ============================================================================
# 🚀 OPTIMIZATION: FAST LOOKUP INDEX
# ============================================================================

# Maps: start_pos -> target_pos -> first_char -> [list of suffixes]
FAST_SUFFIX_INDEX = {}
def _build_fast_index():
    """
    Categorizes suffixes by their starting characters to avoid checking 
    irrelevant suffixes during recursion. Runs ONCE on import.
    """
    global FAST_SUFFIX_INDEX
    if FAST_SUFFIX_INDEX:
        return

    # Probe roots to detect possible starting chars for each suffix
    probe_roots = ["a", "e", "ol", "ak", "ev", "süt", "buz", "top"]
    
    for start_pos, targets in SUFFIX_TRANSITIONS.items():
        FAST_SUFFIX_INDEX[start_pos] = {}
        
        for target_pos, suffix_list in targets.items():
            FAST_SUFFIX_INDEX[start_pos][target_pos] = {
                'vowel': [],      # Suffixes starting with vowels
                'consonant_map': {} # Map 'k' -> [suffixes starting with k]
            }
            
            for suffix in suffix_list:
                possible_starts = set()
                try:
                    # Generate forms for all probe roots
                    forms = set()
                    for r in probe_roots:
                        forms.update(suffix.form(r))
                    
                    for f in forms:
                        if not f: continue
                        possible_starts.add(f[0])
                        
                except Exception:
                    pass
                
                # If no forms generated or complex, add to vowel list (safest fallback)
                if not possible_starts:
                    FAST_SUFFIX_INDEX[start_pos][target_pos]['vowel'].append(suffix)
                    continue

                is_vowel_start = any(c in wrd.VOWELS for c in possible_starts)
                
                if is_vowel_start:
                    FAST_SUFFIX_INDEX[start_pos][target_pos]['vowel'].append(suffix)
                else:
                    c_map = FAST_SUFFIX_INDEX[start_pos][target_pos]['consonant_map']
                    for char in possible_starts:
                        if char not in c_map: c_map[char] = []
                        if suffix not in c_map[char]:
                            c_map[char].append(suffix)
_build_fast_index()


# ============================================================================
# CORE RECURSIVE LOGIC
# ============================================================================
def find_suffix_chain(word: str, start_pos: str, root: str, 
                     current_chain: List = None, visited: Set = None) -> List: 
    """
    Optimized suffix chain finder using Lookahead Indexing.
    """
    
    if current_chain is None: current_chain = []
    if visited is None: visited = set()
    
    # Memoization key
    chain_signature = tuple(s.name for s in current_chain)
    state_key = (len(root), start_pos, chain_signature)
    
    if state_key in visited: return []
    visited.add(state_key)
    
    root_len = len(root)
    rest = word[root_len:]
    
    # Base Case
    if not rest:
        return [([], start_pos)]
    
    if start_pos not in FAST_SUFFIX_INDEX:
        return []
    
    results = []
    next_char = rest[0]
    iyor_variations = ('iyor', 'ıyor', 'uyor', 'üyor')

    for target_pos, lookup_data in FAST_SUFFIX_INDEX[start_pos].items():
        
        # 🚀 FILTER CANDIDATES
        candidates = []
        candidates.extend(lookup_data['vowel']) # Always check vowel-starters (they might drop chars)
        candidates.extend(lookup_data['consonant_map'].get(next_char, []))
        
        # Special check for narrowing scenarios
        if len(rest) > 1 and any(rest[1:].startswith(v) for v in iyor_variations):
             candidates.extend(lookup_data['vowel'])

        for suffix_obj in candidates:
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
                sf_len = len(suffix_form)
                
                # Fast Fail: Suffix longer than remaining text
                if sf_len > len(rest):
                    continue

                # --- MATCH TYPE 1: Standard ---
                if rest.startswith(suffix_form):
                    subchains = find_suffix_chain(
                        word, target_pos, 
                        word[:root_len + sf_len], 
                        current_chain + [suffix_obj], 
                        visited
                    )
                    for chain, final_pos in subchains:
                        results.append(([suffix_obj] + chain, final_pos))

                # --- MATCH TYPE 2: Vowel Narrowing ---
                elif sf_len > 0 and suffix_form[-1] in ('a', 'e'):
                    narrowed_form = suffix_form[:-1]
                    if rest.startswith(narrowed_form):
                        rest_after = rest[len(narrowed_form):]
                        if any(rest_after.startswith(v) for v in iyor_variations):
                            subchains = find_suffix_chain(
                                root + suffix_form + rest_after, 
                                target_pos, 
                                root + suffix_form, 
                                current_chain + [suffix_obj], 
                                visited
                            )
                            for chain, final_pos in subchains:
                                results.append(([suffix_obj] + chain, final_pos))
                    
    return results
    
@lru_cache(maxsize=100000)
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
    if len(surface_root) > 2 and len(surface_root) < 6 and  wrd.ends_with_consonant(surface_root):
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

@lru_cache(maxsize=50000)
def decompose(word: str) -> List[Tuple]:
    """
    Find all possible legal decompositions of a word.
    Uses caching to speed up repeated calls.
    """
    if not word:
        return []
    
    analyses = []
    
    
    def pekistirme_dummy_form(word, suffix_obj):
        return ["dummy_form"]
    pekistirme_suffix = Suffix("pekistirme", "pekistirme", Type.BOTH, Type.BOTH, form_function=pekistirme_dummy_form, is_unique=True)

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

            is_valid_noun = wrd.exists(lemma_root)
            is_valid_verb = wrd.can_be_verb(lemma_root)
            
            if not (is_valid_noun or is_valid_verb):
                continue

            virtual_word = lemma_root + word[len(surface_root):]
            
            if is_valid_noun:
                noun_chains = find_suffix_chain(virtual_word, "noun", lemma_root)
                for chain, final_pos in noun_chains:
                    analyses.append((lemma_root, "noun", chain, final_pos))
            
            if is_valid_verb:
                verb_chains = find_suffix_chain(virtual_word, "verb", lemma_root)
                for chain, final_pos in verb_chains:
                    analyses.append((lemma_root, "verb", chain, final_pos))

    return analyses