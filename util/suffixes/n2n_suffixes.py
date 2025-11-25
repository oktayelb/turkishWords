

from util.suffix import Suffix, Type, HasMajorHarmony,HasMinorHarmony

VOWELS = ["a","e","ı","i","o","ö","u","ü"]


def form_for_confactuous_le(word, suffix_obj):
    
    from util.suffix import Suffix

    base = "le"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in VOWELS:
        base = "y" + base

    return [base]


def form_for_approximative_si(word, suffix_obj):
    
    from util.suffix import Suffix
    base1 = "imsi"
    base2 = "si"

    

    base1 = Suffix._apply_major_harmony(word, base1, suffix_obj.major_harmony)
    base1 = Suffix._apply_minor_harmony(word, base1, suffix_obj.minor_harmony)

    base2 = Suffix._apply_major_harmony(word, base2, suffix_obj.major_harmony)
    base2 = Suffix._apply_minor_harmony(word, base2, suffix_obj.minor_harmony)

    if word[-1] in VOWELS:
        base1 =  base1[1:]

    return [base1, base2]


def form_for_ablative_de(word, suffix_obj):
    """Form function for ablative_de suffix"""
    from util.suffix import Suffix

    base = "de"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in ["ı","i","u","ü"]:   # akuzatif
        base = "n" + base

    return [base]



def form_for_pasttense_noundi(word,suffix_obj):

    from util.suffix import Suffix

    di_base = "di"
    di_base = Suffix._apply_major_harmony(word, di_base, suffix_obj.major_harmony)
    di_base = Suffix._apply_minor_harmony(word, di_base, suffix_obj.minor_harmony)
    di_base = Suffix._apply_consonant_hardening(word, di_base)
    
    dik_base = "dik"
    dik_base = Suffix._apply_major_harmony(word, dik_base, suffix_obj.major_harmony)
    dik_base = Suffix._apply_minor_harmony(word, dik_base, suffix_obj.minor_harmony)
    dik_base = Suffix._apply_consonant_hardening(word, dik_base)

    if word[-1] in VOWELS:
        di_base = "y" + di_base
        dik_base = "y" + dik_base

    return [dik_base,di_base]

def form_for_abstractifier_iyat(word, suffix_obj):
    result_list = ["iye","iyet","iyat","at","et"]
    
    return result_list


def form_for_adverbial_erek (word, suffix_obj):
    from util.suffix import Suffix

    base = "erek"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in VOWELS:
        base = "y" + base

    return [base]


abstractifier_iyat =Suffix("abstractifier_iyat", "iyat", Type.NOUN, Type.NOUN,form_function= form_for_abstractifier_iyat, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
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
approximative_si = Suffix("approximative_si", "si", Type.NOUN, Type.NOUN, form_function= form_for_approximative_si, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
privative_siz = Suffix("privative_siz", "siz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
philicative_cil = Suffix("philicative_cil", "cil", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
composessive_li = Suffix("composessive_li", "li", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
suitative_lik = Suffix("suitative_lik", "lik", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
marking_ki = Suffix("marking_ki", "ki", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)
temporative_leyin = Suffix("temporative_leyin", "leyin", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
ideologicative_izm = Suffix("ideologicative_izm", "izm", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)
scientist_olog = Suffix("scientist_olog", "olog", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)
confactuous_le = Suffix("confactuous_le", "le", Type.NOUN, Type.NOUN, form_function= form_for_confactuous_le, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)


accusative = Suffix("accusative", "i", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
locative_den = Suffix("locative_den", "den", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
dative_e = Suffix("dative_e", "e", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
ablative_de = Suffix("ablative_de", "de", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)

NOUN2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]