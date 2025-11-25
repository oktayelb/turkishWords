from enum import Enum
from typing import List, Tuple

import util.word_methods as wrd
from util.rules.suffix_rules import validate_suffix_addition as validate
# Import custom form functions
from util.rules import suffix_forms as forms


class Type(Enum):
    NOUN = 'noun'
    VERB = 'verb'


class HasMajorHarmony(Enum):
    Yes = 0
    No = 1


class HasMinorHarmony(Enum):
    Yes = 0
    No = 1


_SUFFIX_REGISTRY = []


class Suffix:
    def __init__(self, name, suffix, comes_to, makes, form_function=None, major_harmony=None, minor_harmony=None, needs_y_buffer=False):
        self.name = name
        self.suffix = str(suffix)
        self.comes_to = comes_to
        self.makes = makes
        self.major_harmony = major_harmony
        self.minor_harmony = minor_harmony
        self.needs_y_buffer = needs_y_buffer
        self.form_function = form_function if form_function else self._default_form
        _SUFFIX_REGISTRY.append(self)
    
    def form(self, word):
        return self.form_function(word, self)
    
    @staticmethod
    def _default_form(word, suffix_obj):
        result = suffix_obj.suffix
        
        result = Suffix._apply_major_harmony(word, result, suffix_obj.major_harmony)
        result = Suffix._apply_minor_harmony(word, result, suffix_obj.minor_harmony)
        result = Suffix._apply_consonant_hardening(word, result)
        
        result_list = [result]
        
        if Suffix._should_add_buffer_variants(word, result):
            if suffix_obj.needs_y_buffer:
                result_list.append('y' + result)
                result_list.append('ğ' + result)
            if len(result) > 1:
                result_list.append(result[1:])
        
        return result_list
    
    @staticmethod
    def _apply_major_harmony(word, result, major_harmony):
        if major_harmony != HasMajorHarmony.Yes:
            return result
        
        if wrd.major_harmony(word) == wrd.MajorHarmony.BACK:
            result = result.replace("e", "a")
            result = result.replace("i", "ı")
            result = result.replace("ü", "u")
            result = result.replace("ö", "o")
        
        return result
    
    @staticmethod
    def _apply_minor_harmony(word, result, minor_harmony):
        if minor_harmony != HasMinorHarmony.Yes:
            return result
        
        word_harmony = wrd.minor_harmony(word)
        
        if word_harmony == wrd.MinorHarmony.BACK_ROUND:
            result = result.replace("ı", "u")
        elif word_harmony == wrd.MinorHarmony.FRONT_ROUND:
            result = result.replace("i", "ü")
        
        return result
    
    @staticmethod
    def _apply_consonant_hardening(word, result):
        if not word or not result:
            return result
        
        if word[-1] not in wrd.HARD_CONSONANTS:
            return result
        
        first_char = result[0]
        if first_char not in ['g', 'c', 'd', 'ğ']:
            return result
        
        hardening_map = {'g': 'k', 'd': 't', 'c': 'ç', 'ğ': 'k'}
        return hardening_map.get(first_char, first_char) + result[1:]
    
    @staticmethod
    def _should_add_buffer_variants(word, result):
        return (word and result and 
                word[-1] in wrd.VOWELS and 
                result[0] in wrd.VOWELS)


# ============================================================================
# VERB TO VERB SUFFIXES (v2v)
# ============================================================================

reflexive_is = Suffix("reflexive_is", "iş", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
reflexive_ik = Suffix("reflexive_ik", "ik", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
active_it = Suffix("active_it", "it", Type.VERB, Type.VERB, form_function=forms.form_for_active_it, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
active_dir = Suffix("active_dir", "dir", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
active_ir = Suffix("active_ir", "ir", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
passive_il = Suffix("passive_il", "il", Type.VERB, Type.VERB, form_function=forms.form_for_passive_il, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
reflexive_in = Suffix("reflexive_in", "in", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
negative_me = Suffix("negative_me", "me", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)

# ============================================================================
# NOUN TO NOUN SUFFIXES (n2n)
# ============================================================================

plural_ler = Suffix("plural_ler", "ler", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
counting_er = Suffix("counting_er", "er", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
cooperative_daş = Suffix("cooperative_daş", "daş", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
relative_ce = Suffix("relative_ce", "ca", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
relative_sel = Suffix("relative_sel", "sel", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
approximative_imtrak = Suffix("approximative_imtrak", "trak", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)
pluralizer_archaic_iz = Suffix("pluralizer_archaic_iz", "iz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
posessive_im = Suffix("posessive_im", "im", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
posessive_in = Suffix("posessive_in", "in", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
dimunitive_cik = Suffix("dimunitive_cik", "cik", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
actor_ci = Suffix("actor_ci", "ci", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
ordinal_inci = Suffix("ordinal_inci", "inci", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
approximative_si = Suffix("approximative_si", "si", Type.NOUN, Type.NOUN, form_function= forms.form_for_approximative_si, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
privative_siz = Suffix("privative_siz", "siz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
philicative_cil = Suffix("philicative_cil", "cil", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
composessive_li = Suffix("composessive_li", "li", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
suitative_lik = Suffix("suitative_lik", "lik", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
marking_ki = Suffix("marking_ki", "ki", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)
temporative_leyin = Suffix("temporative_leyin", "leyin", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
ideologicative_izm = Suffix("ideologicative_izm", "izm", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)
scientist_olog = Suffix("scientist_olog", "olog", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)
confactuous_le = Suffix("confactuous_le", "le", Type.NOUN, Type.NOUN, form_function= forms.form_for_confactuous_le, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)


accusative = Suffix("accusative", "i", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
locative_den = Suffix("locative_den", "den", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
dative_e = Suffix("dative_e", "e", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
ablative_de = Suffix("ablative_de", "de", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
# ============================================================================
# NOUN TO VERB SUFFIXES (n2v)
# ============================================================================

absentative_se = Suffix("absentative_se", "se", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
onomatopea_de  = Suffix("onomatopea_de",  "de", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
verbifier_e    = Suffix("verbifier_e",     "e", Type.NOUN, Type.VERB, form_function=forms.form_for_verbifier_e, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
aplicative_le  = Suffix("aplicative_le",  "le", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)

# ============================================================================
# VERB TO NOUN SUFFIXES (v2n)
# ============================================================================

nounifier_ecek = Suffix("nounifier_ecek", "ecek", Type.VERB, Type.NOUN, form_function=forms.form_for_nounifier_ecek, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
infinitive_me = Suffix("infinitive_me", "me", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
infinitive_mek = Suffix("infinitive_mek", "mek", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
factative_en = Suffix("factative_en", "en", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
factative_ir = Suffix("factative_ir", "ir", Type.VERB, Type.NOUN, form_function=forms.form_for_factative_ir, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
toolative_ek = Suffix("toolative_ek", "ek", Type.VERB, Type.NOUN, form_function=forms.form_for_toolative_ek, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
adverbial_e = Suffix("adverbial_e", "e", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
constofactative_gen = Suffix("constofactative_gen", "gen", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
constofactative_gin = Suffix("constofactative_gin", "gin", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_iş = Suffix("nounifier_iş", "iş", Type.VERB, Type.NOUN, form_function=forms.form_for_nounifier_iş, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
perfectative_ik = Suffix("perfectative_ik", "ik", Type.VERB, Type.NOUN, form_function=forms.form_for_perfectative_ik, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
nounifier_i = Suffix("nounifier_i", "i", Type.VERB, Type.NOUN, form_function=forms.form_for_nounifier_i, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
nounifier_gi = Suffix("nounifier_gi", "gi", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_ge = Suffix("nounifier_ge", "ge", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
nounifier_im = Suffix("nounifier_im", "im", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_in = Suffix("nounifier_in", "in", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_it = Suffix("nounifier_it", "it", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_inç = Suffix("nounifier_inç", "inç", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_inti = Suffix("nounifier_inti", "inti", Type.VERB, Type.NOUN, form_function=forms.form_for_nounifier_inti, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
toolifier_geç = Suffix("toolifier_geç", "geç", Type.VERB, Type.NOUN, form_function=forms.form_for_toolifier_geç, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
neverfactative_mez = Suffix("neverfactative_mez", "mez", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
pastfactative_miş = Suffix("pastfactative_miş", "miş", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
adjectiative_ay_v2n = Suffix("adjectiative_ay", "ay", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
nounifier_anak = Suffix("nounifier_anak", "anak", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
nounifier_amak = Suffix("nounifier_amak", "amak", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
abstractifier_iyat =Suffix("abstractifier_iyat", "iyat", Type.VERB, Type.NOUN,form_function=forms.form_for_abstractifier_iyat, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)



ALL_SUFFIXES = _SUFFIX_REGISTRY

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