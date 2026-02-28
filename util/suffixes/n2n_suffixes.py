from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

from util.suffixes.n2n.case_suffixes import CASESUFFIX
from util.suffixes.n2n.posessive_suffix import POSESSIVE_SUFFIX
from util.suffixes.n2n.plural_suffix import PLURALS
from util.suffixes.n2n.derivationals import DERIVATIONALS
from util.suffixes.n2n.conjugation_suffixes import CONJUGATIONS 
VOWELS = ["a","e","ı","i","o","ö","u","ü"]


####
###   Form for rest
####

def form_for_if_suffix  (word, suffix_obj):

    base= "se"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)

    if(word and word[-1] in VOWELS):
        base = "y" + base
    
    return [base]






def form_for_pasttense_noundi(word,suffix_obj):
    # Hem "doktor-du" (Ek fiil) hem "gel-ecek-ti" (Hikaye birleşik zaman)
    di_base = "di"
    di_base = Suffix._apply_major_harmony(word, di_base, suffix_obj.major_harmony)
    di_base = Suffix._apply_minor_harmony(word, di_base, suffix_obj.minor_harmony)
    di_base = Suffix._apply_consonant_hardening(word, di_base)
    
    if word and word[-1] in VOWELS:
        ydi_base = "y" + di_base
        return [ydi_base, di_base]

    return [di_base]

def form_for_when_ken(word, suffix_obj):
    base = "ken"
    if word and word[-1] in VOWELS:
        base = "y" + base
    return [base]






def form_for_copula_mis (word, suffix_obj):
    base = "miş"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    
    if word and word[-1] in VOWELS:
        base = "y" + base

    return [base]


def form_for_confactuous_le(word, suffix_obj):
    base = "le"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in VOWELS:
        base = "y" + base

    return [base]

# ============================================================================
# SUFFIX DEFINITIONS
# ============================================================================

## LERI GECE-LERI EKLENMEKLİ Mİ? 
#-- new group, can only get predicative
approximative_imtrak = Suffix("approximative_imtrak", "imtrak", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL_LOCKING)
temporative_leyin = Suffix("temporative_leyin", "leyin", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL_LOCKING)
#yazın kışın
adverbial_ince = Suffix("adverbial_in", "in", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL_LOCKING)
adverbial_cesine = Suffix("adverbial_cesine", "cesine", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL_LOCKING)
when_ken = Suffix("when_ken", "ken", Type.NOUN, Type.NOUN, form_function= form_for_when_ken , major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL_LOCKING) #zarf sanrırım

# --- Group 45: POST_CASE (İstisna) ---                                                                 
marking_ki = Suffix("marking_ki", "ki", Type.NOUN, Type.NOUN,major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.POST_CASE) # is_unique çünkü sadece bir kere gelebilir.
##le yi halletmeli
confactous_le = Suffix("confactuous_le", "le", Type.NOUN, Type.NOUN, form_function= form_for_confactuous_le, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.POST_CASE, is_unique=True) 
# --- Group 50: PREDICATIVE (Bildirme/Şahıs Ekleri) ---
nounaorist_dir =  Suffix("nounaorist_dir", "dir", Type.NOUN, Type.NOUN , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.PREDICATIVE)
pasttense_noundi = Suffix("pasttense_noundi", "di", Type.BOTH, Type.NOUN,form_function= form_for_pasttense_noundi, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.PREDICATIVE)
if_suffix = Suffix("if_suffix", "se", Type.NOUN, Type.NOUN, form_function= form_for_if_suffix ,major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.PREDICATIVE)
copula_mis = Suffix("copula_mis", "miş", Type.NOUN, Type.NOUN,form_function= form_for_copula_mis , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.PREDICATIVE)




NOUN2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
NOUN2NOUN = NOUN2NOUN + CASESUFFIX + POSESSIVE_SUFFIX + PLURALS + DERIVATIONALS +CONJUGATIONS