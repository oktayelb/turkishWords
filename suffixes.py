# suffixes.py

from enum import Enum
import word_methods as wrd


class WType(Enum):
    Noun = 0
    Verb = 1

class HasMajorHarmony(Enum):      
    Yes = 0                      
    No = 1                       

class HasMinorHarmony(Enum):
    Yes = 0
    No = 1

# Registry to auto-collect all suffixes
_SUFFIX_REGISTRY = []

class Suffix():
    def __init__(self,name, suffix, comes_to, makes, major_harmony=None, minor_harmony=None):
        self.suffix = str(suffix)
        self.comes_to = comes_to
        self.makes = makes
        self.name = name
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
reflexive_is            = Suffix("reflexive_is", "iş", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
reflexive_ik            = Suffix("reflexive_ik", "ik", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_t                = Suffix("active_t", "it", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_tir              = Suffix("active_tir", "dir", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes) 
active_ir               = Suffix("active_ir", "ir", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_il              = Suffix("passive_il", "il", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_in              = Suffix("passive_in", "in", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
negative_me             = Suffix("negative_me", "me", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)

# N2N suffixes (Noun to Noun)
dative_archaic_ke       = Suffix("dative_archaic_ke", "ke", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
locative_den            = Suffix("locative_den", "den", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
dative_e                = Suffix("dative_e", "e", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
ablative_de             = Suffix("ablative_de", "de", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
plural_ler              = Suffix("plural_ler", "ler", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
counting_er             = Suffix("counting_er", "er", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
cooperative_daş         = Suffix("cooperative_daş", "daş", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_ce             = Suffix("relative_ce", "ca", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_sel            = Suffix("relative_sel", "sel", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
dimunitive_ek_archaic   = Suffix("dimunitive_ek_archaic", "ek", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
approximative_imtrak    = Suffix("approximative_imtrak", "trak", WType.Noun, WType.Noun, HasMajorHarmony.No, HasMinorHarmony.No)
accusative              = Suffix("accusative", "i", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
pluralizer_archaic_iz   = Suffix("pluralizer_archaic_iz", "iz", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
genitive_im             = Suffix("genitive_im", "im", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
genitive_in             = Suffix("genitive_in", "in", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
dimunitive_cik          = Suffix("dimunitive_cik", "cik", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
actor_ci                = Suffix("actor_ci", "ci", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
ordinal_inci            = Suffix("ordinal_inci", "inci", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
approximative_si        = Suffix("approximative_si", "si", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
privative_siz           = Suffix("privative_siz", "siz", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
philicative_cil         = Suffix("philicative_cil", "cil", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
composessive_li         = Suffix("composessive_li", "li", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
suitative_lik           = Suffix("suitative_lik", "lik", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)

# N2V suffixes (Noun to Verb)
absentative_se          = Suffix("absentative_se", "se", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
verbifier_e             = Suffix("verbifier_e", "e", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
aplicative_le           = Suffix("aplicative_le", "le", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
grantative_let          = Suffix("grantative_let", "let", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
transformative_les      = Suffix("transformative_les", "leş", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
acquirative_len         = Suffix("acquirative_len", "len", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
verbifier_ik            = Suffix("verbifier_ik", "ik", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)

# V2N suffixes (Verb to Noun)
nounifier_ecek          = Suffix("nounifier_ecek", "ecek", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
infinitive_ma           = Suffix("infinitive_ma", "me", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
infinitive_mak          = Suffix("infinitive_mak", "mek", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
factative_en            = Suffix("factative_en", "en", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
factative_er            = Suffix("factative_er", "er", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
factative_en_y          = Suffix("factative_en_y", "yen", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
toolative_en            = Suffix("toolative_en", "ek", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
adverbial_en            = Suffix("adverbial_en", "e", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
constofactative_gen      = Suffix("constofactative_gen", "gen", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
constofactative_gin      = Suffix("constofactative_gin", "gin", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_iş             = Suffix("nounifier_iş", "iş", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_iş_y           = Suffix("nounifier_iş_y", "yiş", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
perfectative_ik          = Suffix("perfectative_ik", "ik", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
perfectative_ik_y        = Suffix("perfectative_ik_y", "yik", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_i              = Suffix("nounifier_i", "i", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_gi             = Suffix("nounifier_gi", "gi", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_im             = Suffix("nounifier_im", "im", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_in             = Suffix("nounifier_in", "in", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_inç            = Suffix("nounifier_inç", "inç", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
 

# ============================================================================
# AUTO-COLLECTED SUFFIX LISTS
# ============================================================================

# All suffixes in order of definition
ALL_SUFFIXES = _SUFFIX_REGISTRY

# Organized by transition type for the mapping
SUFFIX_TRANSITIONS = {
    'noun': {
        'noun': [s for s in ALL_SUFFIXES if s.comes_to == WType.Noun and s.makes == WType.Noun],
        'verb': [s for s in ALL_SUFFIXES if s.comes_to == WType.Noun and s.makes == WType.Verb]
    },
    'verb': {
        'noun': [s for s in ALL_SUFFIXES if s.comes_to == WType.Verb and s.makes == WType.Noun],
        'verb': [s for s in ALL_SUFFIXES if s.comes_to == WType.Verb and s.makes == WType.Verb]
    }
}
