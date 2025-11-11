# suffixes.py

from enum import Enum
from typing import List, Tuple, Dict

import util.word_methods as wrd

class HasMajorHarmony(Enum):      
    Yes = 0                      
    No = 1                       

class HasMinorHarmony(Enum):
    Yes = 0
    No = 1

# Registry to auto-collect all suffixes
_SUFFIX_REGISTRY = []

class Suffix():
    def __init__(self, name, suffix, comes_to, makes, major_harmony=None, minor_harmony=None, needs_y_buffer=False):
        self.suffix = str(suffix)
        self.comes_to = comes_to
        self.makes = makes
        self.name = name
        self.needs_y_buffer = needs_y_buffer  # New attribute for y-buffer requirement
        
        # Auto-determine major harmony if not specified
        if major_harmony is None:
            self.major_harmony = HasMajorHarmony.Yes if suffix not in ['trak','ki','yor','gil','leyin','man'] else HasMajorHarmony.No
        else:
            self.major_harmony = major_harmony
        
        # Auto-determine minor harmony if not specified
        if minor_harmony is None:
            self.minor_harmony = HasMinorHarmony.Yes if (self.major_harmony == HasMajorHarmony.Yes and 'i' in self.suffix) else HasMinorHarmony.No
        else:
            self.minor_harmony = minor_harmony
        
        # Auto-register this suffix
        _SUFFIX_REGISTRY.append(self)
    
    def form(self, word):
        """Generate harmonized suffix form for given word"""
        if not word:
            return self.suffix
            
        result = self.suffix
        
        # Y-buffer: Add 'y' before vowel-initial suffixes when word ends in vowel
        # This applies to specific suffixes that need y-buffer (marked with #)
        if self.needs_y_buffer and word and word[-1] in wrd.VOWELS and result and result[0] in wrd.VOWELS:
            result = 'y' + result
            # Skip the normal vowel drop logic for y-buffer suffixes
            # Apply harmonies and return
            if self.major_harmony == HasMajorHarmony.Yes:
                word_harmony = wrd.major_harmony(word)
                if word_harmony == wrd.MajorHarmony.BACK:
                    result = result.replace("e", "a")
                    result = result.replace("i", "ı")
                    result = result.replace("ü", "u")
                    result = result.replace("ö", "o")
            
            if self.minor_harmony == HasMinorHarmony.Yes:
                word_harmony = wrd.minor_harmony(word)
                if word_harmony == wrd.MinorHarmony.BACK_ROUND:
                    result = result.replace("ı", "u")
                elif word_harmony == wrd.MinorHarmony.FRONT_ROUND:
                    result = result.replace("i", "ü")
            
            return result
        
        # Consonant hardening: g->k, d->t, c->ç when word ends in hard consonant
        if word[-1] in wrd.HARD_CONSONANTS and result and result[0] in ['g','c','d']:
            if result[0] == 'g':
                result = 'k' + result[1:]
            elif result[0] == 'd':
                result = 't' + result[1:]
            elif result[0] == 'c':
                result = 'ç' + result[1:]
        
        # Vowel drop: if word ends in vowel and suffix starts with vowel
        if result and word[-1] in wrd.VOWELS and result[0] in wrd.VOWELS and len(result) > 1:
            result = result[1:]
            # If suffix becomes consonant-only after dropping first vowel, return as is
            if not result or wrd.has_no_vowels(result) or len(result) == 1:
                return result
        
        # Major vowel harmony (back/front)
        if self.major_harmony == HasMajorHarmony.Yes:
            word_harmony = wrd.major_harmony(word)
            if word_harmony == wrd.MajorHarmony.BACK:
                # Replace front vowels with back vowels
                result = result.replace("e", "a")
                result = result.replace("i", "ı")
                result = result.replace("ü", "u")
                result = result.replace("ö", "o")
        
        # Minor vowel harmony (round/unround)
        if self.minor_harmony == HasMinorHarmony.Yes:
            word_harmony = wrd.minor_harmony(word)
            
            if word_harmony == wrd.MinorHarmony.BACK_ROUND:
                # o, u -> change ı to u
                result = result.replace("ı", "u")
            elif word_harmony == wrd.MinorHarmony.FRONT_ROUND:
                # ö, ü -> change i to ü
                result = result.replace("i", "ü")
        
        return result


# ============================================================================
# SUFFIX DEFINITIONS
# ============================================================================

# V2V suffixes (Verb to Verb)
reflexive_is            = Suffix("reflexive_is", "iş", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
reflexive_ik            = Suffix("reflexive_ik", "ik", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_t                = Suffix("active_t", "it", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_tir              = Suffix("active_tir", "dir", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_ir               = Suffix("active_ir", "ir", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_il              = Suffix("passive_il", "il", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_in              = Suffix("passive_in", "in", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
negative_me             = Suffix("negative_me", "me", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.No)

# N2N suffixes (Noun to Noun)
dative_archaic_ke       = Suffix("dative_archaic_ke", "ke", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
locative_den            = Suffix("locative_den", "den", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
dative_e                = Suffix("dative_e", "e", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No, needs_y_buffer=True)  # Added y-buffer
ablative_de             = Suffix("ablative_de", "de", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
plural_ler              = Suffix("plural_ler", "ler", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
counting_er             = Suffix("counting_er", "er", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
cooperative_daş         = Suffix("cooperative_daş", "daş", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_ce             = Suffix("relative_ce", "ca", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_sel            = Suffix("relative_sel", "sel", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
dimunitive_ek_archaic   = Suffix("dimunitive_ek_archaic", "ek", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
approximative_imtrak    = Suffix("approximative_imtrak", "trak", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.No, HasMinorHarmony.No)
accusative              = Suffix("accusative", "i", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes, needs_y_buffer=True)  # Added y-buffer
pluralizer_archaic_iz   = Suffix("pluralizer_archaic_iz", "iz", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
genitive_im             = Suffix("genitive_im", "im", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
genitive_in             = Suffix("genitive_in", "in", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
dimunitive_cik          = Suffix("dimunitive_cik", "cik", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
actor_ci                = Suffix("actor_ci", "ci", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
ordinal_inci            = Suffix("ordinal_inci", "inci", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
approximative_si        = Suffix("approximative_si", "si", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
privative_siz           = Suffix("privative_siz", "siz", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
philicative_cil         = Suffix("philicative_cil", "cil", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
composessive_li         = Suffix("composessive_li", "li", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
suitative_lik           = Suffix("suitative_lik", "lik", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
adjectiative_ay         = Suffix("adjectiative_ay", "ay", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
marking_ki              = Suffix("marking_ki", "ki", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.No, HasMinorHarmony.No)
temporative_leyin       = Suffix("temporative_leyin", "leyin", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes) ## buraya bak
ideologicative_izm      = Suffix("ideologicative_izm", "izm", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.No, HasMinorHarmony.No) ## buraya bak
locative_le             = Suffix("locative_le", "le", wrd.Type.NOUN, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No) ## buraya bak

# N2V suffixes (Noun to Verb)
absentative_se          = Suffix("absentative_se", "se", wrd.Type.NOUN, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.No)
verbifier_e             = Suffix("verbifier_e", "e", wrd.Type.NOUN, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.No)
aplicative_le           = Suffix("aplicative_le", "le", wrd.Type.NOUN, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.No)
grantative_let          = Suffix("grantative_let", "let", wrd.Type.NOUN, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.No)
transformative_les      = Suffix("transformative_les", "leş", wrd.Type.NOUN, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.No)
acquirative_len         = Suffix("acquirative_len", "len", wrd.Type.NOUN, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.No)
verbifier_ik            = Suffix("verbifier_ik", "ik", wrd.Type.VERB, wrd.Type.VERB, HasMajorHarmony.Yes, HasMinorHarmony.Yes)

# V2N suffixes (Verb to Noun)
nounifier_ecek          = Suffix("nounifier_ecek" , "ecek", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No, needs_y_buffer=True)  # Added y-buffer
infinitive_ma           = Suffix("infinitive_ma"  , "me", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
infinitive_mak          = Suffix("infinitive_mak" , "mek", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
factative_en            = Suffix("factative_en"   , "en", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No, needs_y_buffer=True)  # Added y-buffer
factative_er            = Suffix("factative_er", "er", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
toolative_ek            = Suffix("toolative_ek", "ek", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
adverbial_e             = Suffix("adverbial_e", "e", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No, needs_y_buffer=True)  # Added y-buffer
constofactative_gen     = Suffix("constofactative_gen", "gen", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
constofactative_gin     = Suffix("constofactative_gin", "gin", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_iş            = Suffix("nounifier_iş", "iş", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes, needs_y_buffer=True)  # Added y-buffer
perfectative_ik         = Suffix("perfectative_ik", "ik", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes, needs_y_buffer=True)  # Added y-buffer
nounifier_i             = Suffix("nounifier_i", "i", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_gi            = Suffix("nounifier_gi", "gi", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_im            = Suffix("nounifier_im", "im", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_in            = Suffix("nounifier_in", "in", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_inç           = Suffix("nounifier_inç", "inç", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_inti          = Suffix("nounifier_inti", "inti", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
toolifier_geç           = Suffix("toolifier_geç", "geç", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
toolifier_eç            = Suffix("toolifier_eç", "eç", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
neverfactative_mez      = Suffix("neverfactative_mez", "mez", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)
pastfactative_miş       = Suffix("pastfactative_miş", "miş", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
adjectiative_ay         = Suffix("adjectiative_ay", "ay", wrd.Type.VERB, wrd.Type.NOUN, HasMajorHarmony.Yes, HasMinorHarmony.No)

# ============================================================================
# AUTO-COLLECTED SUFFIX LISTS
# ============================================================================

# All suffixes in order of definition
ALL_SUFFIXES = _SUFFIX_REGISTRY

# Organized by transition type for the mapping
SUFFIX_TRANSITIONS = {
    'noun': {
        'noun': [s for s in ALL_SUFFIXES if s.comes_to == wrd.Type.NOUN and s.makes == wrd.Type.NOUN],
        'verb': [s for s in ALL_SUFFIXES if s.comes_to == wrd.Type.NOUN and s.makes == wrd.Type.VERB]
    },
    'verb': {
        'noun': [s for s in ALL_SUFFIXES if s.comes_to == wrd.Type.VERB and s.makes == wrd.Type.NOUN],
        'verb': [s for s in ALL_SUFFIXES if s.comes_to == wrd.Type.VERB and s.makes == wrd.Type.VERB]
    }
}

suffix_to_id = {}
id_to_suffix = {}
category_to_id: Dict[str, int] = {
            'Noun': 0,
            'Verb': 1
        }
for idx, suffix in enumerate(ALL_SUFFIXES):
    suffix_to_id[suffix.name] = idx
    id_to_suffix[idx] = suffix.name

    
def encode_suffix_chain( suffix_objects: List) -> Tuple[List, List]:
    """
    Convert a list of Suffix objects to numeric tensor representations.
    
    Args:
        suffix_objects: List of Suffix objects from a decomposition
        
    Returns:
        Tuple of:
            - object_ids: Tensor of suffix IDs
            - category_ids: Tensor of POS category IDs (Noun/Verb)
            
    Note:
        Empty chains return single padding tokens (ID 0)
    """
    # Handle empty suffix chains
    if not suffix_objects:
        return [0], [0]
    
    object_ids = []
    category_ids = []
    
    # Convert each suffix to its numeric representation
    for suffix_obj in suffix_objects:
        # Get suffix ID (default to 0 for unknown suffixes)
        suffix_id = suffix_to_id.get(suffix_obj.name, 0)
        
        # Get category ID (Noun=0, Verb=1)
        category_id = category_to_id.get(suffix_obj.makes.name, 0)
        
        object_ids.append(suffix_id)
        category_ids.append(category_id)
    
    return object_ids, category_ids
    

    
