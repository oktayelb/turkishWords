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
    ## şelale akışına istisna olarak  yapım eklerinden sonra fiil ekleri gelebilir 
    if last_g == SuffixGroup.DERIVATIONAL and next_g <= SuffixGroup.DERIVATIONAL:
        return True   
    ## ebilmekten gibi eklerden sonra  fiil ekleri gelebilir. 
    if last_g == SuffixGroup.VERB_COMPOUND and next_g <= SuffixGroup.VERB_COMPOUND:
        return True
    # isim tamlamasından sonra yalnızca ki gelebilir
    if last_g == SuffixGroup.COMPOUND and next_g != SuffixGroup.POST_CASE:
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
        if last_g in [SuffixGroup.DERIVATIONAL, SuffixGroup.VERB_DERIVATIONAL, SuffixGroup.PREDICATIVE]:
            return True
        return False

    return True


# ============================================================================
# CORE RECURSIVE LOGIC
# ============================================================================
def find_suffix_chain(word: str, start_pos: str, root: str, 
                     current_chain: List = None, visited: Set = None) -> List: 
    """
    Recursive suffix chain finder.
    Iterates through SUFFIX_TRANSITIONS directly without pre-indexing.
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
    
    # Base Case: No more characters to consume -> Valid chain found
    if not rest:
        return [([], start_pos)]
    
    # Safety check: Ensure the current POS has defined transitions
    if start_pos not in SUFFIX_TRANSITIONS:
        return []
    
    results = []
    iyor_variations = ('iyor', 'ıyor', 'uyor', 'üyor')

    # Iterate over all possible next Parts of Speech (target_pos)
    # and all suffixes available for that transition.
    for target_pos, candidate_suffixes in SUFFIX_TRANSITIONS[start_pos].items():
        
        for suffix_obj in candidate_suffixes:
            
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
            # Generate possible forms of this suffix when attached to 'root'
            suffix_forms = suffix_obj.form(root)
            
            for suffix_form in suffix_forms:
                sf_len = len(suffix_form)
                
                # Fast Fail: Suffix generated is longer than remaining text
                if sf_len > len(rest):
                    continue

                # --- MATCH TYPE 1: Standard Match ---
                if rest.startswith(suffix_form):
                    subchains = find_suffix_chain(
                        word, target_pos, 
                        word[:root_len + sf_len], 
                        current_chain + [suffix_obj], 
                        visited
                    )
                    for chain, final_pos in subchains:
                        results.append(([suffix_obj] + chain, final_pos))

                # --- MATCH TYPE 2: Vowel Narrowing (e.g., bekle -> bekl-iyor) ---
                elif sf_len > 0 and suffix_form[-1] in ('a', 'e'):
                    narrowed_form = suffix_form[:-1]
                    # Check if the shortened form matches (e.g. 'bekl')
                    if rest.startswith(narrowed_form):
                        rest_after = rest[len(narrowed_form):]
                        # This narrowing only happens if the NEXT part is -iyor/uyor
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


# ============================================================================
# PEKİŞTİRME LOGIC
# ============================================================================
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

def append_analysis(word, pos, root, analyses_list):
    possible_chains = find_suffix_chain(word, pos, root)
    for chain, final_pos in possible_chains:
        analyses_list.append((root, pos, chain, final_pos))

def decompose(word: str) -> List[Tuple]:
    """
    Bİr sözcük için olası tüm kök-ek ayrışımlarını bulur.

    """

    
    # Pekiştirme var mı diye yoklama
    analyses = get_pekistirme_analyses(word)

    # sözbaşından sonuna dek tüm ayrımları dene
    for i in range(1, len(word) + 1):
        root = word[:i]
        # sözlükte var mı
        if wrd.can_be_noun(root):
            append_analysis(word, "noun", root, analyses)
        # sözlükte mastarlı hali var mı? (aç, açmak)   
        if wrd.can_be_verb(root):
            append_analysis(word, "verb", root, analyses)
        
        # eğer iki türlü de sözlükte yoksa kök değişime uğramış olabilir
        # 1- kökte ünsüz yumuşaması (git-> gidecek)
        # 2- kökte ünlü düşmesi     (beniz -> benzemek)
        elif not wrd.can_be_noun(root) and not wrd.can_be_verb(root):
            root_pairs = wrd.get_root_candidates(word[:i])
            for  lemma_root in root_pairs:
                
                virtual_word = lemma_root + word[i:]
                
                if wrd.can_be_noun(lemma_root):
                    append_analysis(virtual_word, "noun", lemma_root, analyses)
                
                if wrd.can_be_verb(lemma_root):
                    append_analysis(virtual_word, "verb", lemma_root, analyses)


    return analyses