from util.suffix import Suffix, Type, SuffixGroup


from util.suffixes.n2n.case_suffixes            import CASESUFFIX
from util.suffixes.n2n.posessive_suffix         import POSESSIVE_SUFFIX
from util.suffixes.n2n.plural_suffix            import PLURALS
from util.suffixes.n2n.derivationals            import DERIVATIONALS
from util.suffixes.n2n.conjugation_suffixes     import CONJUGATIONS 
from util.suffixes.n2n.copula                   import COPULA
from util.suffixes.n2n.marking_suffix           import MARKINGS   
VOWELS = ["a","e","ı","i","o","ö","u","ü"]



def form_for_when_ken(word, suffix_obj):
    base = "ken"
    if word and word[-1] in VOWELS:
        base = "y" + base
    return [base]




def form_for_confactuous_le(word, suffix_obj):
    base = "le"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in VOWELS:
        base = "y" + base

    return [base]

# ============================================================================
# SUFFIX DEFINITIONS
# ============================================================================

## LERI GECE-LERI EKLENMEKLİ Mİ? 
#-- new group, can only get predicative
temporative_leyin = Suffix("temporative_leyin", "leyin", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.DERIVATIONAL_LOCKING)
#tempolocative_leri = Suffix("tempolocative_leri", "leri", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL_LOCKING)    
adverbial_in = Suffix("adverbial_in", "in", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.DERIVATIONAL_LOCKING)
adverbial_cesine = Suffix("adverbial_cesine", "cesine", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=False, group=SuffixGroup.DERIVATIONAL_LOCKING)

## le den sonra ken gelebiliyor
when_ken = Suffix("when_ken", "ken", Type.NOUN, Type.NOUN, form_function= form_for_when_ken , has_major_harmony=False, has_minor_harmony=False, group=SuffixGroup.DERIVATIONAL_LOCKING) #zarf sanrırım

# --- Group 45: POST_CASE (İstisna) ---                                                                 
##le yi halletmeli
confactous_le = Suffix("confactuous_le", "le", Type.NOUN, Type.NOUN, form_function= form_for_confactuous_le, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.WITH_LE, is_unique=True) 




NOUN2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
NOUN2NOUN = NOUN2NOUN + CASESUFFIX + POSESSIVE_SUFFIX + PLURALS + DERIVATIONALS +CONJUGATIONS + COPULA +MARKINGS