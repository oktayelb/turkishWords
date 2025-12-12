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
ALL_SUFFIXES = NOUN2NOUN + NOUN2VERB + VERB2NOUN + VERB2VERB
# ============================================================================
# SUFFIX TYPES
# ============================================================================
SUFFIX_TRANSITIONS = {
    'noun': {
        # Standard N->N + (V->N suffixes that also accept Nouns)
        'noun': NOUN2NOUN + [s for s in VERB2NOUN if s.comes_to == Type.BOTH],
        
        # Standard N->V + (V->V suffixes that also accept Nouns)
        'verb': NOUN2VERB + [s for s in VERB2VERB if s.comes_to == Type.BOTH]
    },
    'verb': {
        # Standard V->N + (N->N suffixes that also accept Verbs)
        'noun': VERB2NOUN + [s for s in NOUN2NOUN if s.comes_to == Type.BOTH],
        
        # Standard V->V + (N->V suffixes that also accept Verbs)
        'verb': VERB2VERB + [s for s in NOUN2VERB if s.comes_to == Type.BOTH]
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
    """Analyzes the text segment and returns (Surface Form, Dictionary Lemma)."""
    candidates = [] 

    def check_and_add_softened(form_to_check):
        if not form_to_check: return

        last_char = form_to_check[-1]
        candidate = form_to_check

        if last_char   == 'b':  candidate = form_to_check[:-1] + 'p'
        elif last_char == 'c':  candidate = form_to_check[:-1] + 'ç'
        elif last_char == 'd':  candidate = form_to_check[:-1] + 't'
        elif last_char == 'ğ':  candidate = form_to_check[:-1] + 'k'
        elif last_char == 'g':  candidate = form_to_check[:-1] + 'k'

        # 2. Check & Append
        if wrd.exists(candidate):
            candidates.append(candidate)



    check_and_add_softened(surface_root)


    if 2 < len(surface_root) < 10 and wrd.ends_with_consonant(surface_root):
        prefix = surface_root[:-1]
        suffix_char = surface_root[-1]
        
        for vowel in ['ı', 'i', 'u', 'ü']:
            restored = prefix + vowel + suffix_char 
            
            # Case A: Restored form is the root (e.g. burn -> burun)
            if wrd.exists(restored):
                candidates.append( restored)
            
            check_and_add_softened(restored)

    # 4. Vowel Narrowing at Root (e.g. diye -> de, yiye -> ye)
    if not wrd.exists(surface_root) and len(surface_root) > 1:
        for terminal_vowel in ['a', 'e']:
            restored = surface_root + terminal_vowel
            if wrd.exists(restored):
                 candidates.append( restored)

    return candidates


pekistirme_suffix = Suffix("pekistirme", "pekistirme", Type.NOUN, Type.NOUN, is_unique=True)
def get_pekistirme_analyses(word: str) -> List[Tuple]:
    """
    Encapsulates all logic for identifying and analyzing intensified adjectives (Pekiştirme).
    Example: 'masmavi' -> (mavi, noun, [pekistirme], noun)
    """

    analyses = []
    
    # Fast exit
    if len(word) < 4: 
        return analyses

    # 1. Detect Vowels and Special Letters (m, p, r, s)
    first_vowel_index = -1
    for i in range(len(word)):
        if word[i] in wrd.VOWELS:
            first_vowel_index = i
            break
            
    if first_vowel_index == -1 or (first_vowel_index + 1) >= len(word):
        return analyses

    if word[first_vowel_index + 1] not in "psrm":  
        return analyses

    # 2. Strategy: Try to find a valid (Prefix, Root) pair
    detected_root = None
    detected_prefix = None
    
    # Case A: Standard (mas-mavi) -> Split at index + 2
    idx_std = first_vowel_index + 2
    # Case B: Extended (gü-p-e-gündüz) -> Split at index + 3 (if connecting vowel exists)
    idx_ext = first_vowel_index + 3

    # Helper to scan for root
    def find_root_in_rest(prefix_len):
        potential_rest = word[prefix_len:]
        # The rest must start with the duplicated part (e.g. 'mas-mavi' -> 'mavi' starts with 'ma')
        if potential_rest.startswith(word[:first_vowel_index + 1]):
            # Greedy search: match longest possible valid dictionary word
            for k in range(len(potential_rest), 1, -1):
                candidate = potential_rest[:k]
                if wrd.exists(candidate):
                    return word[:prefix_len], candidate
        return None, None

    # Try Case A
    detected_prefix, detected_root = find_root_in_rest(idx_std)

    # Try Case B if A failed, checking for connecting vowel (a/e)
    if not detected_root and idx_ext < len(word):
        if word[first_vowel_index + 2] in ['a', 'e']:
             detected_prefix, detected_root = find_root_in_rest(idx_ext)

    if not detected_root:
        return analyses

    # 3. Build Analysis Chain
    # Scenario 1: Pure Pekiştirme (e.g., "masmavi")
    if len(detected_prefix + detected_root) == len(word):
        analyses.append((detected_root, "noun", [pekistirme_suffix], "noun"))
    
    # Scenario 2: Pekiştirme with suffixes (e.g., "masmaviyim")
    else:
        # Extract the logical part for suffix analysis: "mas" + "maviyim"
        virtual_word_part = word[len(detected_prefix):] # e.g., "maviyim"
        
        # Recursive call: Find suffixes attached to the root "mavi" within "maviyim"
        chains = find_suffix_chain(virtual_word_part, "noun", detected_root)
        
        for chain, final_pos in chains:
            full_chain = [pekistirme_suffix] + chain
            analyses.append((detected_root, "noun", full_chain, final_pos))

    return analyses

@lru_cache(maxsize=50000)
def decompose(word: str) -> List[Tuple]:
    """
    Find all possible legal decompositions of a word.
    """
    if not word:
        return []
    
    # 1. Get Pekiştirme Analyses (Cleaner!)
    analyses = get_pekistirme_analyses(word)

    # 2. Standard Analyses
    ## GET ROOT candidates yalnızca exists 0 döndürürse çalışmalı (mı??)
    for i in range(1, len(word) + 1):
        root = word[:i]
        if wrd.exists(root):
            noun_chains = find_suffix_chain(word, "noun", root)
            for chain, final_pos in noun_chains:
                analyses.append((root, "noun", chain, final_pos))
            
            
            if wrd.can_be_verb(root):
                verb_chains = find_suffix_chain(word, "verb", root)
                for chain, final_pos in verb_chains:
                    analyses.append((root, "verb", chain, final_pos))

        
        else :
            root_pairs = get_root_candidates(word[:i])
            for  lemma_root in root_pairs:
                
                virtual_word = lemma_root + word[i:]
                
                if wrd.exists(lemma_root):
                    noun_chains = find_suffix_chain(virtual_word, "noun", lemma_root)
                    for chain, final_pos in noun_chains:
                        analyses.append((lemma_root, "noun", chain, final_pos))
                
                if wrd.can_be_verb(lemma_root):
                    verb_chains = find_suffix_chain(virtual_word, "verb", lemma_root)
                    for chain, final_pos in verb_chains:
                        analyses.append((lemma_root, "verb", chain, final_pos))

    return analyses