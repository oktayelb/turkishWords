

from util.suffix import  Suffix, Type, HasMajorHarmony,HasMinorHarmony

VOWELS = ["a","e","ı","i","o","ö","u","ü"]

##bunları dolduracan 
def form_for_conjugation_2sg(word, suffix_obj):

    return_list = []
    if  len(word) > 3 and word[-3:] in ["miş","müş","mış","muş", "ir","ür","ur","ır","ecek","acak","cek","cak","mez","maz","yor"]:
        sin_base = "sin"
        sin_base = Suffix._apply_major_harmony(word, sin_base, suffix_obj.major_harmony)
        sin_base = Suffix._apply_minor_harmony(word, sin_base, suffix_obj.minor_harmony)
        return_list.append(sin_base)
    
    if len(word)>2 and word[-2:] in ["di","dı","du","dü"]:
        n_base = "n"
        n_base = Suffix._apply_major_harmony(word, n_base, suffix_obj.major_harmony)
        n_base = Suffix._apply_minor_harmony(word, n_base, suffix_obj.minor_harmony)
        return_list.append(n_base)

    return return_list 

def form_for_conjugation_3sg(word, suffix_obj):
    pass

def form_for_conjugation_1pl(word, suffix_obj):
    pass

def form_for_conjugation_2pl(word, suffix_obj):
    pass

def form_for_conjugation_3pl(word, suffix_obj):
    pass


def form_for_posessive_3sg(word, suffix_obj):
    base = "i"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in VOWELS:
        base = "s" + base

    return [base]
  


def form_for_confactuous_le(word, suffix_obj):
    
    

    base = "le"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in VOWELS:
        base = "y" + base

    return [base]


def form_for_approximative_si(word, suffix_obj):
    
    
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
    

    base = "de"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in ["ı","i","u","ü"]:   # akuzatif
        base = "n" + base

    return [base]



def form_for_pasttense_noundi(word,suffix_obj):

    

    di_base = "di"
    di_base = Suffix._apply_major_harmony(word, di_base, suffix_obj.major_harmony)
    di_base = Suffix._apply_minor_harmony(word, di_base, suffix_obj.minor_harmony)
    di_base = Suffix._apply_consonant_hardening(word, di_base)
    
    ## bu aslında sadece isim kökenliler için ama fiile gelirken olmuyor, ve wordun fiil oldugunu anlama yöntemim yok su anda
    ydi_base = "y" + di_base

    return [di_base,ydi_base]

def form_for_abstractifier_iyat(word, suffix_obj):
    result_list = ["iye","iyet","iyat","at","et"]
    
    return result_list


def form_for_adverbial_erek (word, suffix_obj):
    

    base = "erek"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in VOWELS:
        base = "y" + base

    return [base]


plural_ler = Suffix("plural_ler", "ler", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
counting_er = Suffix("counting_er", "er", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
cooperative_daş = Suffix("cooperative_daş", "daş", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
relative_ce = Suffix("relative_ce", "ca", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
relative_sel = Suffix("relative_sel", "sel", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
approximative_imtrak = Suffix("approximative_imtrak", "trak", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)
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
abstractifier_iyat =Suffix("abstractifier_iyat", "iyat", Type.NOUN, Type.NOUN,form_function= form_for_abstractifier_iyat, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No)

### 
pasttense_noundi = Suffix("pasttense_noundi", "di", Type.BOTH, Type.NOUN,form_function= form_for_pasttense_noundi, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)

accusative = Suffix("accusative", "i", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
locative_den = Suffix("locative_den", "den", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
dative_e = Suffix("dative_e", "e", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
ablative_de = Suffix("ablative_de", "de", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)


posessive_1sg = Suffix("posessive_1sg", "im", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
posessive_2sg = Suffix("posessive_2sg", "in", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
posessive_3sg = Suffix("posessive_3sg", "i", Type.NOUN, Type.NOUN, form_function=form_for_posessive_3sg, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)

posessive_1pl = Suffix("posessive_1pl", "imiz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes) ## bunlar yerine de iz eki eklenebilir
posessive_2pl = Suffix("posessive_2pl", "iniz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
posessive_3pl = Suffix("posessive_3pl", "leri", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)  ##?? olmalı mı 

##buranın gerekliliği tartışılmalı
conjugation_1sg = Suffix("conjugation_1sg", "im", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
conjugation_2sg = Suffix("conjugation_2sg","sin", Type.NOUN, Type.NOUN, form_function= form_for_conjugation_2sg, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
#conjugation_3sg = Suffix("conjugation_3sg", "", Type.NOUN, Type.NOUN, form_function=form_for_conjugation_3sg , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)

#conjugation_1sg = Suffix("conjugation_1sg", "im", Type.NOUN, Type.NOUN, form_function= form_for_conjugation_1pl, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
#conjugation_2sg = Suffix("conjugation_2sg", "im", Type.NOUN, Type.NOUN, form_function=form_for_conjugation_2pl , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
#conjugation_3sg = Suffix("conjugation_3sg", "im", Type.NOUN, Type.NOUN, form_function=form_for_conjugation_3pl , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)

NOUN2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]