from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

VOWELS = ["a","e","ı","i","o","ö","u","ü"]

# ============================================================================
# FORM FUNCTIONS
# ============================================================================

def form_for_conjugation_1sg(word, suffix_obj):
    """
    1. Tekil Şahıs (-m, -im)
    - Gel-di-m (Sadece m)
    - Doktor-um (im)
    - Baba-y-ım (y + im)
    - sev-me-m (me + m)
    """
    return_list = []
    
    # 1. Durum: Geçmiş zaman (-di) ve Şart (-se) sonrası sadece 'm'
    if len(word) > 2 and word[-2:] in ["di","dı","du","dü", "ti","tı","tu","tü", "se","sa","me","ma"]:
        return_list.append("m")
        return return_list 
    #TODO eyim geleyim gideyim ekle
    # 2. Durum: Standart -im hali (Geleceğ-im, Doktor-um, Arkadaşlar-ım)
    base = "im"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    
    # Kaynaştırma harfi (Hasta-y-ım)
    if word and word[-1] in VOWELS:
        return_list.append('y' + base)
    else:
        return_list.append(base)


    return return_list


def form_for_conjugation_2sg(word, suffix_obj):
    """
    2. Tekil Şahıs (-n, -sin)
    - Gel-di-n (Sadece n)
    - Doktor-sun (sin)
    - Arkadaşlarımızdan-sın (sin)
    """
    return_list = []

    # 1. Durum: Geçmiş zaman (-di) veya Şart (-se) sonrası sadece 'n'
    if len(word) > 2 and word[-2:] in ["di","dı","du","dü", "ti","tı","tu","tü", "se","sa"]:
        return_list.append("n")
        
    ## TODO if is verb  add ""
    
    # 2. Durum: Standart -sin hali (Predicative / Geniş Zaman / Şimdiki Zaman)
    sin_base = "sin"
    sin_base = Suffix._apply_major_harmony(word, sin_base, suffix_obj.major_harmony)
    sin_base = Suffix._apply_minor_harmony(word, sin_base, suffix_obj.minor_harmony)
    return_list.append(sin_base)

    in_base = "in"
    in_base = Suffix._apply_major_harmony(word, in_base, suffix_obj.major_harmony)
    in_base = Suffix._apply_minor_harmony(word, in_base, suffix_obj.minor_harmony)
    if word and word[-1] in VOWELS:
        return_list.append('y' + in_base)
    else:
        return_list.append(in_base)
    return return_list 


def form_for_conjugation_3sg(word, suffix_obj):
    """
    3. Tekil Şahıs (Genelde boştur, emir kipinde -sin)
    """
    # 1. Durum: Emir kipi (Gel-sin, Yap-sın, Gel-me-sin)
    # Sadece fiil kökenli veya olumsuzluk ekinden sonra mantıklıdır.
    # "O doktor-sun" denmez, "O doktor" denir. O yüzden her yere -sin eklemiyoruz.
    if len(word) > 2 and word[-2:] in ["me","ma"]:
         sin_base = "sin"
         sin_base = Suffix._apply_major_harmony(word, sin_base, suffix_obj.major_harmony)
         sin_base = Suffix._apply_minor_harmony(word, sin_base, suffix_obj.minor_harmony)
         return [sin_base]

    # Eğer kelime saf fiil kökü ise (Gel-sin)
    if wrd.can_be_verb(word):
         sin_base = "sin"
         sin_base = Suffix._apply_major_harmony(word, sin_base, suffix_obj.major_harmony)
         sin_base = Suffix._apply_minor_harmony(word, sin_base, suffix_obj.minor_harmony)
         # Bunu listeye ekleyelim ama return etmeyelim, belki aşağıdan boş string de döner.
         # Ancak 3. tekil şahıs genelde null suffix'tir.
         return [sin_base, ""]

    # Standart durum: 3. Tekil Şahıs eki YOKTUR. (O doktor, O geldi)
    return [""]


def form_for_conjugation_1pl(word, suffix_obj):
    """
    1. Çoğul Şahıs (-k, -iz)
    - Gel-di-k (k)
    - Doktor-uz (iz)
    - Hasta-y-ız (y + iz)
    """
    return_list = []
    
    # 1. Durum: Geçmiş zaman (-di) ve Şart (-se) sonrası 'k'
    if len(word) > 2 and word[-2:] in ["di","dı","du","dü", "ti","tı","tu","tü", "se","sa"]:
        k_base = "k" 
        return_list.append(k_base)
        
    # 2. Durum: Standart -iz hali
    iz_base = "iz"
    iz_base = Suffix._apply_major_harmony(word, iz_base, suffix_obj.major_harmony)
    iz_base = Suffix._apply_minor_harmony(word, iz_base, suffix_obj.minor_harmony)

    if word and word[-1] in VOWELS:
        return_list.append('y' + iz_base)
    else:
        return_list.append(iz_base)

    return return_list 
 

def form_for_conjugation_2pl(word, suffix_obj):
    """
    2. Çoğul Şahıs (-niz, -siniz)
    - Gel-di-niz (niz)
    - Doktor-sunuz (siniz)
    - Arkadaşlarımızdan-sınız (siniz)
    """
    return_list = []
    
    # 1. Durum: Geçmiş zaman (-di) ve Şart (-se) sonrası '-niz'
    if len(word) > 2 and word[-2:] in ["di","dı","du","dü", "ti","tı","tu","tü", "se","sa"]:
        niz_base = "niz"
        niz_base = Suffix._apply_major_harmony(word, niz_base, suffix_obj.major_harmony)
        niz_base = Suffix._apply_minor_harmony(word, niz_base, suffix_obj.minor_harmony)
        return_list.append(niz_base)
        
    # 2. Durum: Standart -siniz (Predicative)
    siniz_base = "siniz"
    siniz_base = Suffix._apply_major_harmony(word, siniz_base, suffix_obj.major_harmony)
    siniz_base = Suffix._apply_minor_harmony(word, siniz_base, suffix_obj.minor_harmony)
    return_list.append(siniz_base)
    
    # 3. Durum: Emir kipi (Gel-in, Gel-iniz) - Sadece fiilse
    if wrd.can_be_verb(word) or (len(word)>2 and word[-2:] in ["me","ma"]):
         in_base = "in"
         in_base = Suffix._apply_major_harmony(word, in_base, suffix_obj.major_harmony)
         in_base = Suffix._apply_minor_harmony(word, in_base, suffix_obj.minor_harmony)
         
         if word[-1] in VOWELS:
             return_list.append('y' + in_base)
         else:
             return_list.append(in_base)

    return return_list 


def form_for_conjugation_3pl(word, suffix_obj):
    result_list  = []
    # Standart: -ler (Gel-ir-ler, Ev-ler)
    base = "ler"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    result_list.append(base)
    
    # Emir kipi 3. çoğul: Gel-sin-ler
    if(wrd.can_be_verb(word)) or word[-2:] in ['me','ma']:
        base_2 = "sinler"
        base_2 = Suffix._apply_major_harmony(word, base_2, suffix_obj.major_harmony)
        base_2 = Suffix._apply_minor_harmony(word, base_2, suffix_obj.minor_harmony)
        result_list.append(base_2)

    return result_list

####
###   Form for rest
####

def form_for_if_suffix  (word, suffix_obj):

    base= "se"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)

    if(word and word[-1] in VOWELS):
        base = "y" + base
    
    return [base]

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
##TODO umsu ekini bi hallet, belki suffix hiyerarşisini yeniden düzenlemek gerekebilir.
def form_for_approximative_si(word, suffix_obj):
    base1 = "imsi"

    base1 = Suffix._apply_major_harmony(word, base1, suffix_obj.major_harmony)
    base1 = Suffix._apply_minor_harmony(word, base1, suffix_obj.minor_harmony)

    if word[-1] in VOWELS:
        base1 =  base1[1:]

    return [base1]

def form_for_ablative_de(word, suffix_obj):
    base = "de"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in ["ı","i","u","ü"]: 
        nbase = "n" + base
        return [nbase,base]
    return [base]

def form_for_locative_den(word, suffix_obj):
    base = "den"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in ["ı","i","u","ü"]: 
        nbase = "n" + base
        return [nbase,base]
    return [base]

def form_for_pasttense_noundi(word,suffix_obj):
    # Hem "doktor-du" (Ek fiil) hem "gel-ecek-ti" (Hikaye birleşik zaman)
    di_base = "di"
    di_base = Suffix._apply_major_harmony(word, di_base, suffix_obj.major_harmony)
    di_base = Suffix._apply_minor_harmony(word, di_base, suffix_obj.minor_harmony)
    di_base = Suffix._apply_consonant_hardening(word, di_base)
    
    if word and word[-1] in VOWELS:
        di_base = "y" + di_base

    return [di_base]

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
    

    base3 = "cağız"
    base3 = Suffix._apply_major_harmony(word, base3, suffix_obj.major_harmony)
    base3 = Suffix._apply_consonant_hardening(word, base3)


    return [base,base2,base3]
def form_for_when_ken(word, suffix_obj):
    base = "ken"
    if word and word[-1] in VOWELS:
        base = "y" + base
    return [base]


def form_for_marking_ki(word, suffix_obj):
    base = "ki"
    base2 = "kü"
    base_prefix = "ın"
    
    base_prefix = Suffix._apply_major_harmony(word, base_prefix, suffix_obj.major_harmony)
    base_prefix = Suffix._apply_minor_harmony(word, base_prefix, suffix_obj.minor_harmony)

    return [base, base_prefix + base, base2]



def form_for_copula_mis (word, suffix_obj):
    base = "miş"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    
    if word and word[-1] in VOWELS:
        base = "y" + base

    return [base]

def form_for_noun_compound_suffix(word, suffix_obj):
    base = "ın"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    if word and word[-1] in VOWELS:
        base = "n" + base

    return [base]

def form_for_accusative(word, suffix_obj):
    base = "i"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    
    if word and word[-1] in VOWELS:
        nbase = "n" + base
        base = "y" + base
        return [nbase, base]
    

    return [base]
# ============================================================================
# SUFFIX DEFINITIONS
# ============================================================================


#-- new group, can only get predicative
approximative_imtrak = Suffix("approximative_imtrak", "imtrak", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL_LOCKING)
temporative_leyin = Suffix("temporative_leyin", "leyin", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL_LOCKING)
adverbial_ince = Suffix("adverbial_in", "in", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL_LOCKING)
adverbial_cesine = Suffix("adverbial_cesine", "cesine", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL_LOCKING)
approximative_si = Suffix("approximative_si", "si", Type.NOUN, Type.NOUN, form_function= form_for_approximative_si, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL_LOCKING)

# --- Group 10: DERIVATIONAL (Yapım Ekleri) ---
counting_er = Suffix("counting_er", "er", Type.NOUN, Type.NOUN, form_function= form_for_counting_er,  major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
cooperative_daş = Suffix("cooperative_daş", "daş", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
relative_ce = Suffix("relative_ce", "ce", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
relative_sel = Suffix("relative_sel", "sel", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
dimunitive_cik = Suffix("dimunitive_cik", "cik", Type.NOUN, Type.NOUN,  form_function= form_for_dimunitive_cik,major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
actor_ci = Suffix("actor_ci", "ci", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
ordinal_inci = Suffix("ordinal_inci", "inci", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
privative_siz = Suffix("privative_siz", "siz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
philicative_cil = Suffix("philicative_cil", "cil", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
composessive_li = Suffix("composessive_li", "li", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
suitative_lik = Suffix("suitative_lik", "lik", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
abstractifier_iyat =Suffix("abstractifier_iyat", "iyat", Type.NOUN, Type.NOUN,form_function= form_for_abstractifier_iyat, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
ideologicative_izm = Suffix("ideologicative_izm", "izm", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
scientist_olog = Suffix("scientist_olog", "olog", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
familative_gil = Suffix("familative_gil", "gil", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
plural_ler  = Suffix("plural_ler", "ler", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)

# --- Group 30: POSSESSIVE (İyelik) ---
posessive_1sg = Suffix("posessive_1sg", "im", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.POSSESSIVE)
posessive_2sg = Suffix("posessive_2sg", "in", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.POSSESSIVE)
posessive_3sg = Suffix("posessive_3sg", "i", Type.NOUN, Type.NOUN, form_function=form_for_posessive_3sg, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.POSSESSIVE)
posessive_1pl = Suffix("posessive_1pl", "imiz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.POSSESSIVE)
posessive_2pl = Suffix("posessive_2pl", "iniz", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.POSSESSIVE)
posessive_3pl = Suffix("posessive_3pl", "leri", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.POSSESSIVE)

noun_compound = Suffix("noun_compound", "in", Type.NOUN, Type.NOUN, form_function=form_for_noun_compound_suffix, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.COMPOUND)

# --- Group 40: CASE (Hal Ekleri) ---
accusative = Suffix("accusative", "i", Type.NOUN, Type.NOUN,form_function= form_for_accusative, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True, group=SuffixGroup.CASE)
locative_den = Suffix("locative_den", "den", Type.NOUN, Type.NOUN, form_function=form_for_locative_den, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.CASE)
dative_e = Suffix("dative_e", "e", Type.NOUN, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True, group=SuffixGroup.CASE)
ablative_de = Suffix("ablative_de", "de", Type.NOUN, Type.NOUN, form_function=form_for_ablative_de ,major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.CASE)
confactuous_le = Suffix("confactuous_le", "le", Type.NOUN, Type.NOUN, form_function= form_for_confactuous_le, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.CASE)


# --- Group 45: POST_CASE (İstisna) ---
marking_ki = Suffix("marking_ki", "ki", Type.NOUN, Type.NOUN, form_function= form_for_marking_ki, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.POST_CASE) # is_unique çünkü sadece bir kere gelebilir.
when_ken = Suffix("when_ken", "ken", Type.NOUN, Type.NOUN, form_function= form_for_when_ken , major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.POST_CASE) #zarf sanrırım


# --- Group 50: PREDICATIVE (Bildirme/Şahıs Ekleri) ---
nounaorist_dir =  Suffix("nounaorist_dir", "dir", Type.NOUN, Type.NOUN , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.PREDICATIVE)
pasttense_noundi = Suffix("pasttense_noundi", "di", Type.BOTH, Type.NOUN,form_function= form_for_pasttense_noundi, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.PREDICATIVE)
if_suffix = Suffix("if_suffix", "se", Type.NOUN, Type.NOUN, form_function= form_for_if_suffix ,major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.PREDICATIVE)
copula_mis = Suffix("copula_mis", "miş", Type.NOUN, Type.NOUN,form_function= form_for_copula_mis , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.PREDICATIVE)

# --- Group 60: CONJUGATION (Ek-Fiil / Bitiş Ekleri) ---
conjugation_1sg = Suffix("conjugation_1sg", "im", Type.BOTH, Type.NOUN, form_function= form_for_conjugation_1sg, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, is_unique=True, group=SuffixGroup.CONJUGATION)
conjugation_2sg = Suffix("conjugation_2sg","sin", Type.BOTH, Type.NOUN, form_function= form_for_conjugation_2sg, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, is_unique=True, group=SuffixGroup.CONJUGATION)
conjugation_3sg = Suffix("conjugation_3sg", "", Type.BOTH, Type.NOUN, form_function=form_for_conjugation_3sg , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, is_unique=True, group=SuffixGroup.CONJUGATION)

conjugation_1pl = Suffix("conjugation_1pl", "iz", Type.BOTH, Type.NOUN, form_function= form_for_conjugation_1pl, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, is_unique=True, group=SuffixGroup.CONJUGATION)
conjugation_2pl = Suffix("conjugation_2pl", "siniz", Type.BOTH, Type.NOUN, form_function=form_for_conjugation_2pl , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, is_unique=True, group=SuffixGroup.CONJUGATION)
conjugation_3pl = Suffix("conjugation_3pl", "ler", Type.BOTH, Type.NOUN, form_function=form_for_conjugation_3pl , major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, is_unique=True, group=SuffixGroup.CONJUGATION)




NOUN2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]