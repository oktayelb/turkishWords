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
    Updated with Hierarchy, Group Repetition, and Global Uniqueness checks.
    """
    if current_chain is None:
        current_chain = []
    if visited is None:
        visited = set()
    
    # Kural geçerliliği yola bağlı olduğu için zincir geçmişini visited key'e ekliyoruz.
    chain_signature = tuple(s.name for s in current_chain)
    state_key = (len(root), start_pos, chain_signature)
    
    if state_key in visited:
        return []
    
    visited.add(state_key)
    
    rest = word[len(root):]
    
    # Base Case: Kelime tamamen tüketildi
    if not rest:
        return [([], start_pos)]
    
    if start_pos not in SUFFIX_TRANSITIONS:
        return []
    
    results = []
    
    for target_pos, suffix_list in SUFFIX_TRANSITIONS[start_pos].items():
        for suffix_obj in suffix_list:
            
            # --- OPTİMİZASYON VE MANTIKSAL KONTROLLER ---
            
            if current_chain:
                last_suffix = current_chain[-1]
                
                # KURAL 1: HİYERARŞİ KONTROLÜ (Hierarchy Check)
                # Yeni ek, kendinden önceki ekin grubundan daha düşük öncelikli olamaz.
                # Örn: Hal Eki (40) -> Çoğul Eki (20) OLMAZ.
                # İstisna: Eğer ileride hiyerarşi dışı özel bir grup tanımlanırsa buraya 'or' ile eklenebilir.
                if suffix_obj.group < last_suffix.group:
                    continue

                # KURAL 2: GRUP TEKRAR KONTROLÜ (Group Repetition Check)
                # Aynı gruptaki ekler peş peşe gelemez.
                # İSTİSNA: Yapım ekleri (Group 10 - DERIVATIONAL) peş peşe gelebilir (örn: yap-tır-t).
                # Çekim ekleri (Grup 20, 30, 40 vb.) kendi kendilerini tekrar edemez.
                if suffix_obj.group == last_suffix.group:
                    if suffix_obj.group > 10:  # 10 = SuffixGroup.DERIVATIONAL
                        continue
            
            # KURAL 3: KÜRESEL BENZERSİZLİK (Global Uniqueness Check)
            # Eğer ek 'is_unique=True' ise (örn: olumsuzluk eki -me),
            # zincirin GEÇMİŞİNDE herhangi bir yerde kullanılmış mı diye bakarız.
            if suffix_obj.is_unique:
                # current_chain içindeki objelerin isimlerini kontrol et
                if any(s.name == suffix_obj.name for s in current_chain):
                    continue

            # --- ESKİ VALIDASYON (Opsiyonel) ---
            # sequence_rules.py içindeki özel 'yasaklı ikililer' varsa kontrol et.
            if not validate(current_chain, suffix_obj):
                continue

            # --- FORM OLUŞTURMA VE RECURSION ---
            # Ekin olası formlarını (ses uyumlarına göre) oluştur ve kelimenin kalanıyla eşleşiyor mu bak.
            for suffix_form in suffix_obj.form(root):
                if rest.startswith(suffix_form):
                    next_root = root + suffix_form
                    remaining = rest[len(suffix_form):]
                    
                    if remaining:
                        # Kelime bitmedi, aramaya devam et
                        subchains = find_suffix_chain(
                            word, 
                            target_pos, 
                            next_root, 
                            current_chain + [suffix_obj], 
                            visited
                        )
                    else:
                        # Kelime bitti, geçerli bir bitiş noktası
                        subchains = [([], target_pos)]
                    
                    for chain, final_pos in subchains:
                        results.append(([suffix_obj] + chain, final_pos))
                    
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
            # --- FİLTRELEME ---
            # Eğer zincir sonucunda kelime hala "verb" tipindeyse,
            # bu geçerli bir tam kelime değildir (fiil kökü veya gövdesidir).
            # Çekimlenmiş fiiller (gel-di), tanımlarında Type.NOUN ürettikleri için
            # (n2n_suffixes.py içindeki tanımlarına göre) bu filtreden geçerler.
            # eğer cümlede gel kal gibi  yapıda şeyler varsa bunların sonuna
            # aslında 2. kişi emir kipi 0 eki geldiği var sayılacak.
            if final_pos == "verb":
                continue

            analyses.append((root, pos, chain, final_pos))
    
    return analyses