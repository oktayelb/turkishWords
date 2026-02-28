from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

VOWELS = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']


def form_for_approximative_si(word, suffix_obj):
    base1 = "imsi"

    base1 = Suffix._apply_major_harmony(word, base1, suffix_obj.major_harmony)
    base1 = Suffix._apply_minor_harmony(word, base1, suffix_obj.minor_harmony)

    base2 = "si"
    base2 = Suffix._apply_major_harmony(word, base2, suffix_obj.major_harmony)
    base2 = Suffix._apply_minor_harmony(word, base2, suffix_obj.minor_harmony)

    if word[-1] in VOWELS:
        return [base1[1:]]

    return [base1, base2]

def form_for_abstractifier_iyat(word, suffix_obj):
    result_list = ["iye","iyet","iyat","at","et"]
    return result_list

def form_for_counting_er(word, suffix_obj):

    base = "er"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)


    if (word and word[-1] in VOWELS):
        base = "ş" + base
    
    return [base]

def form_for_dimunitive_cik(word, suffix_obj):  
    base = "cik"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base) 

    base2 = "cek"
    base2 = Suffix._apply_major_harmony(word, base2, suffix_obj.major_harmony)
    base2 = Suffix._apply_minor_harmony(word, base2, suffix_obj.minor_harmony)
    base2 = Suffix._apply_consonant_hardening(word, base2)
    
## cağız ayırabilirim
    base3 = "cağız"
    base3 = Suffix._apply_major_harmony(word, base3, suffix_obj.major_harmony)
    base3 = Suffix._apply_consonant_hardening(word, base3)


    return [base,base2,base3]


actor_ci        = Suffix("actor_ci", "ci", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
privative_siz   = Suffix("privative_siz", "siz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
composessive_li = Suffix("composessive_li", "li", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
suitative_lik   = Suffix("suitative_lik", "lik", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)


counting_er         = Suffix("counting_er", "er", Type.NOUN, Type.NOUN, form_function= form_for_counting_er,  major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
cooperative_daş     = Suffix("cooperative_daş", "daş", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
relative_ce         = Suffix("relative_ce", "ce", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
relative_sel        = Suffix("relative_sel", "sel", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
dimunitive_cik      = Suffix("dimunitive_cik", "cik", Type.NOUN, Type.NOUN,  form_function= form_for_dimunitive_cik,major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
ordinal_inci        = Suffix("ordinal_inci", "inci", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
philicative_cil     = Suffix("philicative_cil", "cil", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
abstractifier_iyat  = Suffix("abstractifier_iyat", "iyat", Type.NOUN, Type.NOUN,form_function= form_for_abstractifier_iyat, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
ideologicative_izm  = Suffix("ideologicative_izm", "izm", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
scientist_olog      = Suffix("scientist_olog", "olog", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
familative_gil      = Suffix("familative_gil", "gil", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
approximative_si    = Suffix("approximative_si", "si", Type.NOUN, Type.NOUN, form_function= form_for_approximative_si, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL_LOCKING)




DERIVATIONALS = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]