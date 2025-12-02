from typing import List, Tuple

# Mevcut importlarınız...
from util.suffixes.v2v_suffixes import VERB2VERB
from util.suffixes.n2v_suffixes import NOUN2VERB
from util.suffixes.n2n_suffixes import NOUN2NOUN
from util.suffixes.v2n_suffixes import VERB2NOUN
import util.word_methods as wrd
from util.rules.suffix_rules import validate_suffix_addition as validate
from util.suffix import Type 

ALL_SUFFIXES = VERB2NOUN + VERB2VERB + NOUN2NOUN + NOUN2VERB 

# Suffix transition kurallarınız (Aynen kalıyor)
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

def find_suffix_chain(word, start_pos, root, current_chain=None, visited=None): 
    """
    Recursive suffix chain finder.
    'root' parametresi sözlükteki orijinal kökü (örn: 'git') temsil eder.
    'word' parametresi analiz edilen metni (örn: 'gidiyorum') temsil eder.
    """
    if current_chain is None:
        current_chain = []
    if visited is None:
        visited = set()
    
    # Kural geçerliliği için imza
    chain_signature = tuple(s.name for s in current_chain)
    
    # State key: (Kök uzunluğu, mevcut pos, zincir)
    # Not: root 'git' olsa bile uzunluğu 3, 'gid' olsa bile 3'tür.
    # Dilimleme (slicing) doğru çalışır.
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
            # Suffix formunu Orijinal Kök (root='git') üzerinde deneriz.
            # form('git') -> 'iyor' döner.
            # 'gidiyorum' kelimesinin kalanı 'iyorum' ile başlar mı? Evet.
            for suffix_form in suffix_obj.form(root):
                if rest.startswith(suffix_form):
                    
                    # Zinciri devam ettirirken 'sanal' kökü büyütebiliriz 
                    # ama recursive yapıda önemli olan kelimenin ne kadarının tüketildiğidir.
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
                    
    return results


def get_root_candidates(surface_root: str) -> List[str]:
    """
    Metinde geçen kök parçası (surface_root) için sözlükteki olası orijinalleri döndürür.
    Örn: 'gid' -> ['git']
         'ağac' -> ['ağaç']
         'reng' -> ['renk']
    """
    candidates = []
    
    # 1. Kelime olduğu gibi sözlükte var mı? (gel, git, ev)
    if wrd.exists(surface_root):
        candidates.append(surface_root)
        
    # 2. Yumuşama Kontrolü (Softening Check)
    if not surface_root:
        return candidates

    last_char = surface_root[-1]
    candidate = None
    
    # b -> p (kitab -> kitap)
    if last_char == 'b':
        candidate = surface_root[:-1] + 'p'
    # c -> ç (ağac -> ağaç)
    elif last_char == 'c':
        candidate = surface_root[:-1] + 'ç'
    # d -> t (gid -> git)
    elif last_char == 'd':
        candidate = surface_root[:-1] + 't'
    # ğ -> k (köpeğ -> köpek)
    elif last_char == 'ğ':
        candidate = surface_root[:-1] + 'k'
    # g -> k (reng -> renk - istisna nk/ng değişimi)
    elif last_char == 'g': 
        if surface_root.endswith("ng"):
            candidate = surface_root[:-2] + 'nk'

    # Eğer bir aday oluşturulduysa ve sözlükte varsa ekle
    if candidate and wrd.exists(candidate):
        candidates.append(candidate)
            
    return candidates


def decompose(word: str) -> List[Tuple]:
    """Find all possible legal decompositions of a word."""
    if not word:
        return []
    
    analyses = []
    for i in range(1, len(word) + 1):
        surface_root = word[:i] # Örn: 'gid'
        
        # 'gid' sözlükte yoksa bile 'git'i bulup getirecek.
        root_candidates = get_root_candidates(surface_root)
        
        # Eğer ne olduğu gibi ne de yumuşamış hali sözlükte yoksa atla
        if not root_candidates:
            continue
            
        for root in root_candidates:
            # root: 'git' (Sözlükteki hali)
            # surface_root: 'gid' (Metindeki hali)
            
            pos = "verb" if wrd.can_be_verb(root) else "noun"

            # find_suffix_chain'e 'git' (root) verilir ki ses uyumları buna göre hesaplansın.
            chains = (find_suffix_chain(word, "verb", root) +
                      find_suffix_chain(word, "noun", root)) if pos == "verb" \
                      else find_suffix_chain(word, "noun", root)

            for chain, final_pos in chains:
                if final_pos == "verb":
                    continue

                analyses.append((root, pos, chain, final_pos))
    
    return analyses 