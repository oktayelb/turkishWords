from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

# ============================================================================
# FORM FUNCTIONS
# ============================================================================

def form_for_active_it(word, suffix_obj):
    """
    Form function for active_it (Ettirgen -t) suffix
    - Örnek: üşü-t, kork-ut, ak-ıt
    """
    result_list = []
    
    # 1. Durum: Ünlü ile bitiyorsa veya r ile bitiyorsa 't' gelir (basit kural)
    # Ancak suffix objesi 'it' olarak tanımlı.
    
    # Base form (it/ıt/ut/üt) -> Kork-ut
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    # Hardening genellikle burada gerekmez ama kural gereği kalsın
    base = Suffix._apply_consonant_hardening(word, base)
    
    # Eğer kelime ünsüzle bitiyorsa 'it/ut' formunu ekle
    if word and word[-1] not in wrd.VOWELS:
        result_list.append(base)
    
    # 2. Durum: Ünlü ile bitiyorsa sadece 't' gelir. (Ara-t, Oku-t)
    # Veya bazı istisnalarda (ak-ıt) araya ünlü girer.
    
    t_form = 't'
    t_form = Suffix._apply_consonant_hardening(word, t_form) # d -> t (nadiren)
    
    # Kelime ünlüyle bitiyorsa 't' ekle
    if  word[-1] in wrd.VOWELS or  word[-1] in ['r', 'l']:
        result_list.append(t_form)
    
    # Özel durum: Çok heceli ve 'r' veya 'l' ile biten bazı kelimelerde de 't' gelebilir
    # (Bu kısım kural tabanlı sistemde karmaşıktır, basit bırakıyoruz)
    
    return result_list


def form_for_passive_il(word, suffix_obj):
    """
    Form function for passive_il (Edilgen) suffix
    - Normali: -il, -ıl, -ul, -ül (Yap-ıl)
    - Sonu 'l' ile bitiyorsa: -in, -ın, -un, -ün (Bul-un)
    - Sonu ünlü ile bitiyorsa: -n (Ara-n)
    """
    result_list = []
    


    # 1. Durum: Kelime ünlü ile bitiyorsa -> sadece 'n' (Ara-n-mak)
    if word[-1] in wrd.VOWELS:
        n_suffix = "n"
        l_suffix = "l"
        result_list.append(n_suffix)
        result_list.append(l_suffix)
        return result_list

    # 2. Durum: Kelime 'l' ünsüzü ile bitiyorsa -> -in (Bul-un-mak)
    elif word[-1] == 'l':
        in_base = "in"
        in_base = Suffix._apply_major_harmony(word, in_base, suffix_obj.major_harmony)
        in_base = Suffix._apply_minor_harmony(word, in_base, suffix_obj.minor_harmony)
        result_list.append(in_base)
        
    # 3. Durum: Diğer ünsüzler -> -il (Yap-ıl-mak)
    else:
        il_base = suffix_obj.suffix # il
        il_base = Suffix._apply_major_harmony(word, il_base, suffix_obj.major_harmony)
        il_base = Suffix._apply_minor_harmony(word, il_base, suffix_obj.minor_harmony)
        il_base = Suffix._apply_consonant_hardening(word, il_base)
        result_list.append(il_base)

    return result_list

def form_for_active_ir(word, suffix_obj):

    base = "ir"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)

    base2 = "er"
    base2 = Suffix._apply_major_harmony(word, base2, suffix_obj.major_harmony)
    base2 = Suffix._apply_minor_harmony(word, base2, suffix_obj.minor_harmony)
    return [base, base2]

# ============================================================================
# VERB TO VERB SUFFIXES (v2v) - Hepsi VERB_DERIVATIONAL (Grup 10)
# ============================================================================

# Not: is_unique=False yapıyoruz çünkü çatı ekleri üst üste gelebilir (Yap-tır-t-tır).
# Sadece Olumsuzluk eki (-me) unique olmalıdır.

# İşteş (Reciprocal/Reflexive): Gül-üş, Gör-üş
reflexive_is = Suffix("reflexive_is", "iş", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.VERB_DERIVATIONAL, is_unique=False)

# Dönüşlü? (Reflexive) / Geçişsizleştiren: Gec-ik, Bir-ik
reflexive_ik = Suffix("reflexive_ik", "ik", Type.BOTH, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.VERB_DERIVATIONAL, is_unique=False)

# Ettirgen (Causative) -it: Kork-ut, Ak-ıt
active_it = Suffix("active_it", "it", Type.VERB, Type.VERB, form_function=form_for_active_it, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.VERB_DERIVATIONAL, is_unique=False)

# Ettirgen (Causative) -dir: Yap-tır, Koş-tur
active_dir = Suffix("active_dir", "dir", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.VERB_DERIVATIONAL, is_unique=False)

# Ettirgen (Causative) -ir: Piş-ir, Düş-ür
active_ir = Suffix("active_ir", "ir", Type.VERB, Type.VERB, form_function=form_for_active_ir, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.VERB_DERIVATIONAL, is_unique=False)

# Edilgen (Passive): Yap-ıl, Bul-un
passive_il = Suffix("passive_il", "il", Type.VERB, Type.VERB, form_function=form_for_passive_il, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.VERB_DERIVATIONAL, is_unique=False)

# Dönüşlü (Reflexive) -in: Giy-in, Sev-in
reflexive_in = Suffix("reflexive_in", "in", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.VERB_DERIVATIONAL, is_unique=False)

randomative_ele = Suffix("randomative_ele", "ele", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.VERB_DERIVATIONAL, is_unique=False)

possibiliative_ebil = Suffix("possibilitative_ebil", "ebil", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.No, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.VERB_COMPOUND, is_unique=True)

# Olumsuzluk (Negative): Gel-me.
# Bu ek Yapım eklerinden sonra gelir, ama Çekim eklerinden önce gelir.
# Hiyerarşide VERB_DERIVATIONAL grubunda kalabilir ama is_unique=True olmalı.
negative_me = Suffix("negative_me", "me", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.VERB_NEGATING, is_unique=True)

negative_consto = Suffix("negative_consto", "eme", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.VERB_NEGATING, is_unique=True)

VERB2VERB =   [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]