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
    def __init__(self, suffix, comes_to, makes, major_harmony=None, minor_harmony=None):
        self.suffix = str(suffix)
        self.comes_to = comes_to
        self.makes = makes
        
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
reflexive_is            = Suffix('iş', WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
reflexive_ik            = Suffix("ik", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_t                = Suffix("it", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_tir              = Suffix("dir", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes) 
active_ir               = Suffix("ir", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_il              = Suffix("il", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_in              = Suffix("in", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
negative_me             = Suffix("me", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)

# N2N suffixes (Noun to Noun)
dative_archaic_ke       = Suffix("ke", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
locative_den            = Suffix("den", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
dative_e                = Suffix("e", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
ablative_de             = Suffix("de", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
plural_ler              = Suffix("ler", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
counting_er             = Suffix("er", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
cooperative_daş         = Suffix("daş", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_ce             = Suffix("ca", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_sel            = Suffix("sel", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
dimunitive_ek_archaic   = Suffix("ek", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
approximative_imtrak    = Suffix("trak", WType.Noun, WType.Noun, HasMajorHarmony.No, HasMinorHarmony.No)
accusative              = Suffix("i", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
pluralizer_archaic_iz   = Suffix("iz", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes) 
genitive_im             = Suffix("im", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
genitive_in             = Suffix("in", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
dimunitive_cik          = Suffix("cik", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
actor_ci                = Suffix("ci", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
ordinal_inci            = Suffix("inci", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
approximative_si        = Suffix("si", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
privative_siz           = Suffix("siz", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
philicative_cil         = Suffix("cil", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
composessive_li         = Suffix("li", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
suitative_lik           = Suffix("lik", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)

# N2V suffixes (Noun to Verb)
absentative_se          = Suffix("se", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
verbifier_e             = Suffix("e", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
aplicative_le           = Suffix("le", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
grantative_let          = Suffix("let", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
transformative_les      = Suffix("leş", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
acquirative_len         = Suffix("len", WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)

# V2N suffixes (Verb to Noun)
nounifier_ecek = Suffix("ecek", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
infinitive_ma = Suffix("me", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
infinitive_mak = Suffix("mek", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
factative_en = Suffix("en", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
toolative_en = Suffix("ek", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
adverbial_en = Suffix("e", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
constofactative_gen = Suffix("gen", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
constofactative_gin = Suffix("gin", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_iş = Suffix("iş", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
perfectative_ik = Suffix("ik", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_i = Suffix("i", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_gi = Suffix("gi", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_im = Suffix("im", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_in = Suffix("in", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_inç = Suffix("inç", WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)

## Çıkarım 1 küçük ünlü uyumu yalnızca eki i ı u ü olarak değiştirir
## Çıkarım 2 küçük ünlü uyumu e/a lı eklerde görülmez
## Çıkarım 3 eylemden eylem yapan eklerin hepsi küçük ünlü uyumuna girer
## Çıkarım 4 addan eylem yapan eklerin hiçi küçük ünlü uyumuna girmez , ik ekini saymadık 


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


## if vowels in suffix are already in wrd.front_vowels do nothing
## since the standart I have put dictates that the default form
## of the suffix will be the front vowel form.


## Kucuk unlu uyumuna ugrayan eklerde e/a bulunmaz.
## Standardimca KUU ya giren tum ekleri `i` ile yazdim
## KUU ya giren her ek BUU ya uyar
## O zaman ek buraya geldiginde ya ı ya da i var
## O zaman aslinda Back.Wide ve Front.Wide a gerek yok.
## Ancak yine de tutacagim
## sozcuk Round ise i/ı yi u/ü ile degisirecegim 