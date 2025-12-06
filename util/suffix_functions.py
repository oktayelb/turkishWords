from typing import List, Tuple

# Mevcut importlarınız...
from util.suffixes.v2v_suffixes import VERB2VERB
from util.suffixes.n2v_suffixes import NOUN2VERB
from util.suffixes.n2n_suffixes import NOUN2NOUN
from util.suffixes.v2n_suffixes import VERB2NOUN
import util.word_methods as wrd
from util.rules.suffix_rules import validate_suffix_addition as validate
from util.suffix import Type , Suffix

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

suffix_to_id = {suffix.name: idx for idx, suffix in enumerate(ALL_SUFFIXES)}
category_to_id = {'Noun': 0, 'Verb': 1}

# --- 1. DEFINITIONS MOVED UP ---

# Create the suffix object here so 'decompose' can see it
def pekistirme_dummy_form(word, suffix_obj):
    """
    Dummy form function. Since we handle pekistirme manually in decompose,
    this is just to satisfy the Suffix class requirement.
    """
    return ["dummy_form"]
pekistirme_suffix = Suffix("pekistirme", "pekistirme", Type.BOTH, Type.BOTH, form_function=pekistirme_dummy_form, is_unique=True)

def find_suffix_chain(word, start_pos, root, current_chain=None, visited=None): 
    """
    Recursive suffix chain finder.
    Bu fonksiyon artık 'Sanal Olarak Düzeltilmiş' (Virtual) kelime üzerinde çalışır.
    """
    if current_chain is None:
        current_chain = []
    if visited is None:
        visited = set()
    
    # Kural geçerliliği için imza
    chain_signature = tuple(s.name for s in current_chain)
    
    # State key: (Kök uzunluğu, mevcut pos, zincir)
    state_key = (len(root), start_pos, chain_signature)
    
    if state_key in visited:
        return []
    visited.add(state_key)
    
    # Kelimenin geri kalanı
    rest = word[len(root):]
    
    # Base Case: Kelime bitti
    if not rest:
        return [([], start_pos)]
    
    if start_pos not in SUFFIX_TRANSITIONS:
        return []
    
    results = []
    
    # -iyor narrow variations check list
    iyor_variations = ['iyor', 'ıyor', 'uyor', 'üyor']

    for target_pos, suffix_list in SUFFIX_TRANSITIONS[start_pos].items():
        for suffix_obj in suffix_list:
            
            # --- Validasyonlar ---
            if current_chain:
                last_suffix = current_chain[-1]
                if suffix_obj.group < last_suffix.group:
                    continue
                if suffix_obj.group == last_suffix.group and suffix_obj.group > 10:
                    continue
            
            if suffix_obj.is_unique:
                if any(s.name == suffix_obj.name for s in current_chain):
                    continue
            
            # --- Form Kontrolü ---
            suffix_forms = suffix_obj.form(root)
            
            for suffix_form in suffix_forms:
                
                # --- CASE 1: Standard Match ---
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

                # --- CASE 2: Vowel Narrowing (Daralma) Match ---
                # Check if suffix ends in 'a' or 'e' (e.g. -ma, -me, -la) AND is followed by -iyor
                elif len(suffix_form) > 0 and suffix_form[-1] in ['a', 'e']:
                    narrowed_form = suffix_form[:-1] # Drop the vowel (me -> m)
                    
                    # Ensure the 'rest' matches the narrowed form (starts with 'm')
                    if rest.startswith(narrowed_form):
                        # Look ahead: Does the part AFTER the narrowed form start with iyor/uyor?
                        rest_after_narrowing = rest[len(narrowed_form):]
                        
                        if any(rest_after_narrowing.startswith(v) for v in iyor_variations):
                            # We found a narrowing case! (e.g. git-m-iyor)
                            # To fix the chain, we must 'repair' the word for the next recursive step.
                            # We construct a virtual word that includes the dropped vowel: "gitmeiyor"
                            
                            next_root = root + suffix_form # Root grows by FULL suffix (gitme)
                            
                            # Construct repaired word for recursion
                            # Current 'word' is "gitmiyor". We insert the dropped vowel.
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
    Metinde geçen parçayı analiz eder ve (Yüzey Hali, Sözlük Hali) döndürür.
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

    # --- NEW: Ünlü Daralması (Vowel Narrowing at Root) ---
    # Example: 'bekl' (from bekliyor) -> 'bekle'
    if not wrd.exists(surface_root) and len(surface_root) > 1:
        # If the root doesn't exist, try adding 'a' or 'e' to see if it's a verb stem
        for terminal_vowel in ['a', 'e']:
            restored_lemma = surface_root + terminal_vowel
            # Check if exists and is NOT a new candidate (to avoid duplicates)
            if wrd.exists(restored_lemma) and is_new_candidate(restored_lemma):
                 candidates.append((surface_root, restored_lemma))

    return candidates

# --- 2. UPDATED CHECK_PEKISTIRME ---
def check_pekistirme(word):
    """
    Returns [prefix, valid_root] if found, otherwise [].
    Checks roots dynamically to allow for suffixes (e.g., masmaviler).
    """
    if len(word) < 4:
        return []
    
    pekistirme_letters = "psrm"
    first_vowel_index = -1
    
    # Safe vowel finder
    for i in range(len(word)):
        if word[i] in wrd.VOWELS:
            first_vowel_index = i
            break
            
    if first_vowel_index == -1: 
        return []

    # Safe index checking
    if first_vowel_index + 1 >= len(word):
        return []

    # Check the consonant (m, p, r, s)
    if word[first_vowel_index+1] not in pekistirme_letters:  
        return []
    
    # --- Logic to allow suffixes (e.g. masmaviler) ---
    # We check if the REST of the word starts with a valid root
    
    # Case 1: Standard (mas-mavi)
    split_index_1 = first_vowel_index + 2
    if split_index_1 < len(word):
        potential_rest = word[split_index_1:]
        prefix = word[:split_index_1]
        
        # Optimization: Rest must start with same syllable (ma-s-ma...)
        if potential_rest.startswith(word[:first_vowel_index+1]):
             # Try to match the start of 'potential_rest' to a dictionary root
             for k in range(len(potential_rest), 1, -1):
                 candidate_root = potential_rest[:k]
                 if wrd.exists(candidate_root):
                     return [prefix, candidate_root]

    # Case 2: Extended (gü-p-e-gündüz)
    split_index_2 = first_vowel_index + 3
    if split_index_2 < len(word):
        # Check linking vowel
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
    
    # --- PEKISTIRME CHECK ---
    pekistirme_data = check_pekistirme(word) # Returns [prefix, root]
    
    if pekistirme_data:
        prefix_str = pekistirme_data[0]
        root_str = pekistirme_data[1]
        
        # 1. Add the plain form (masmavi) if no suffixes remain
        if len(prefix_str + root_str) == len(word):
             # !!! FIXED TYPE MISMATCH: Wrapped pekistirme_suffix in [] !!!
            analyses.append((root_str, "noun", [pekistirme_suffix], "noun"))
            
        # 2. Add suffixed forms (masmaviler)
        else:
            # Construct a 'virtual word' that excludes the prefix 
            # so find_suffix_chain can work normally on the rest
            # e.g. "masmaviler" -> virtual="maviler", root="mavi"
            virtual_word_part = word[len(prefix_str):] 
            
            chains = find_suffix_chain(virtual_word_part, "noun", root_str)
            
            for chain, final_pos in chains:
                # Prepend the pekistirme suffix to the found chain
                full_chain = [pekistirme_suffix] + chain
                analyses.append((root_str, "noun", full_chain, final_pos))

    # --- STANDARD DECOMPOSITION ---
    for i in range(1, len(word) + 1):
        surface_part = word[:i] 
        root_pairs = get_root_candidates(surface_part)
        
        if not root_pairs:
            continue
            
        for surface_root, lemma_root in root_pairs:
            # Reconstruct virtual word for analysis
            # If lemma is 'bekle' and surface is 'bekl', virtual word becomes 'bekle' + 'iyor'
            virtual_word = lemma_root + word[len(surface_root):]
            
            pos = "verb" if wrd.can_be_verb(lemma_root) else "noun"

            chains = (find_suffix_chain(virtual_word, "verb", lemma_root) +
                      find_suffix_chain(virtual_word, "noun", lemma_root)) if pos == "verb" \
                      else find_suffix_chain(virtual_word, "noun", lemma_root)

            for chain, final_pos in chains:
                analyses.append((lemma_root, pos, chain, final_pos))

    return analyses