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

def find_suffix_chain(word, start_pos, root, current_chain=None, visited=None): 
    """
    Recursive suffix chain finder.
    Bu fonksiyon artık 'Sanal Olarak Düzeltilmiş' (Virtual) kelime üzerinde çalışır.
    Örn: word='beniziyor', root='beniz' -> rest='iyor'.
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
    # Not: Sanal kelime gönderildiği için len(root) artık doğru kesim yapar.
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
            # Suffix formunu Orijinal Kök (root='beniz') üzerinde deneriz.
            suffix_forms = suffix_obj.form(root)
            
            for suffix_form in suffix_forms:
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
                    
    return results


def get_root_candidates(surface_root: str) -> List[Tuple[str, str]]:
    """
    Metinde geçen parçayı (surface_root) analiz eder ve
    (Yüzey Hali, Sözlük Hali) çiftlerini döndürür.
    
    Güncelleme: Ünlü düşmesi kontrolü sırasında oluşan kelime için
    ayrıca ünsüz sertleşmesi (orijinal haline dönüş) kontrolü eklenmiştir.
    Örn: 'kayb' -> 'kayıb' (yok) -> 'kayıp' (var)
    """
    candidates = [] # List of tuples: (surface, lemma)
    
    # Yardımcı fonksiyon: Aday listesinde zaten var mı?
    def is_new_candidate(lemma):
        return not any(cand[1] == lemma for cand in candidates)

    # 1. Olduğu gibi var mı?
    if wrd.exists(surface_root):
        candidates.append((surface_root, surface_root))
        
    if not surface_root:
        return candidates

    last_char = surface_root[-1]
    
    # --- YARDIMCI MANTIK: Yumuşamayı Geri Alma (Unsoftening Logic) ---
    def get_unsoftened_char(char, text_ending):
        if char == 'b': return 'p'
        if char == 'c': return 'ç'
        if char == 'd': return 't'
        if char == 'ğ': return 'k'
        if char == 'g' and text_ending.endswith("ng"): return 'nk'
        return None

    # --- 2. Sadece Yumuşama Kontrolü (Softening) ---
    # Örn: 'gid' -> 'git'
    target_char = get_unsoftened_char(last_char, surface_root)
    if target_char:
        if target_char == 'nk': # 'ng' -> 'nk' özel durumu
            candidate_lemma = surface_root[:-2] + 'nk'
        else:
            candidate_lemma = surface_root[:-1] + target_char
            
        if wrd.exists(candidate_lemma) and is_new_candidate(candidate_lemma):
            candidates.append((surface_root, candidate_lemma))

    # --- 3. Ünlü Düşmesi Kontrolü (Syncope) ---
    # Metinde 'ayr' var -> 'ayır' mı? 
    # Metinde 'kayb' var -> 'kayıp' mı? (Hem düşme hem yumuşama)
    if len(surface_root) >= 2 and wrd.ends_with_consonant(surface_root):
        prefix = surface_root[:-1]
        suffix_char = surface_root[-1] # Bu karakter yumuşamış olabilir (örn: b)
        
        narrow_vowels = ['ı', 'i', 'u', 'ü']
        for vowel in narrow_vowels:
            # A) Sadece Ünlü Düşmesi: 'benz' -> 'beniz'
            restored_lemma = prefix + vowel + suffix_char 
            
            if wrd.exists(restored_lemma) and is_new_candidate(restored_lemma):
                candidates.append((surface_root, restored_lemma))
            
            # B) Ünlü Düşmesi + Ünsüz Değişimi: 'kayb' -> 'kayıb' -> 'kayıp'
            # restored_lemma şu an 'kayıb'. Sonu 'b' ile bitiyor, bunu 'p' yapmalıyız.
            last_char_of_restored = restored_lemma[-1]
            unsoftened_char = get_unsoftened_char(last_char_of_restored, restored_lemma)
            
            if unsoftened_char:
                if unsoftened_char == 'nk':
                    restored_unsoftened = restored_lemma[:-2] + 'nk'
                else:
                    restored_unsoftened = restored_lemma[:-1] + unsoftened_char
                
                if wrd.exists(restored_unsoftened) and is_new_candidate(restored_unsoftened):
                    candidates.append((surface_root, restored_unsoftened))

    return candidates

def decompose(word: str) -> List[Tuple]:
    """Find all possible legal decompositions of a word."""
    if not word:
        return []
    
    analyses = []
    
    for i in range(1, len(word) + 1):
        surface_part = word[:i] # Örn: 'benz'
        
        # 'benz' için [('benz', 'beniz')] döner
        root_pairs = get_root_candidates(surface_part)
        
        if not root_pairs:
            continue
            
        for surface_root, lemma_root in root_pairs:
            # surface_root: 'benz' (Metindeki hali)
            # lemma_root:   'beniz' (Sözlük/Gramatik hali)
            
            # --- SANAL KELİME RESTORASYONU (VIRTUAL WORD) ---
            # Senin önerin: "Sözcüğü kendimiz benizemek olarak varsayalım"
            # Fiziksel kelimenin geri kalanı: word[len(surface_root):] -> 'emek' veya 'iyor'
            # Sanal kelime = Sözlük Kökü + Kalan Parça
            # Örn: 'beniz' + 'emek' = 'benizemek'
            # Örn: 'beniz' + 'iyor' = 'beniziyor' (Aslında benziyor ama analiz için beniziyor varsayıyoruz)
            
            virtual_word = lemma_root + word[len(surface_root):]
            
            pos = "verb" if wrd.can_be_verb(lemma_root) else "noun"

            # find_suffix_chain'e artık SANAL kelimeyi ve DÜZGÜN kökü gönderiyoruz.
            # Böylece len(root) ile kesme işlemi (slicing) tam oturuyor.
            chains = (find_suffix_chain(virtual_word, "verb", lemma_root) +
                      find_suffix_chain(virtual_word, "noun", lemma_root)) if pos == "verb" \
                      else find_suffix_chain(virtual_word, "noun", lemma_root)

            for chain, final_pos in chains:
                # Analiz sonucunda kök olarak 'beniz' (lemma) döner.
                analyses.append((lemma_root, pos, chain, final_pos))
    
    return analyses