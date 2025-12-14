from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
from util.word_methods import VOWELS
def  form_for_verbifier_e(word,suffix_obj):
    if word[-1] in VOWELS:
        return ["dummt str'ng"]
    base = "e"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    return [base]

# ============================================================================
# NOUN TO VERB SUFFIXES (n2v) - Hepsi DERIVATIONAL (Grup 10)
# ============================================================================

# absentative_se: su-sa (susadım), garip-se (garipsedim)
absentative_se = Suffix("absentative_se", "se", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)

# onomatopea_de: çatır-da, gürül-de
onomatopea_de  = Suffix("onomatopea_de",  "de", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)

# verbifier_e: kan-a, oyun-a (oyna), yaş-a
verbifier_e    = Suffix("verbifier_e",     "e", Type.NOUN, Type.VERB, form_function= form_for_verbifier_e, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)

# aplicative_le: su-la, baş-la
aplicative_le  = Suffix("aplicative_le",  "le", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)


NOUN2VERB =[
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]