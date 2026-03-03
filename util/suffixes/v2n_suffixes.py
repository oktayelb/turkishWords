from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

from util.suffixes.v2n.gerunds import GERUNDS
from util.suffixes.v2n.infinitives import INFINITIVES
from util.suffixes.v2n.participles import PARICIPLES
from util.suffixes.v2n.nounifiers import NOUNIFIERS
VOWELS = ["a","e","ı","i","o","ö","u","ü"]

# ============================================================================
# FORM FUNCTIONS
# ============================================================================

def form_for_continuous_iyor(word, suffix_obj):
    base = "i"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)

    base = base+"yor"
    return [base]

""""
def form_for_nounifier_ecek(word, suffix_obj):

    result_list = []
    
    # Base form: ecek
    base = "ecek"   
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    # Softened base: eceğ / acağ
    soft_base = Suffix._apply_softening(base)
    if soft_base != base:
        result_list.append(soft_base)
    
    # Add buffer variants for vowel-ending words
    if word and word[-1] in VOWELS:
        # yecek variant (yıkayacak)
        y_form = 'y' + base
        result_list.append(y_form)
        
        # Softened y_form: yeceğ
        soft_y = Suffix._apply_softening(y_form)
        if soft_y != y_form:
            result_list.append(soft_y)
        
        # cek variant (vowel dropped - uncommon in standard V2N but listed)
        short_form = base[1:] 
        result_list.append(short_form)
        
        # Softened short_form: ceğ
        soft_short = Suffix._apply_softening(short_form)
        if soft_short != short_form:
            result_list.append(soft_short)
    
    return result_list
"""

"""
def form_for_adverbial_erek (word, suffix_obj):
    # Erek yumuşamaz (Giderek -> Gidereği X)
    base = "erek"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    if word[-1] in VOWELS:
        base = "y" + base
    return [base]
"""

"""
def form_for_nounifier_iş(word, suffix_obj):

    result_list = []
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        if word[-1] in VOWELS:
            result_list.append('y' + base) # işleyiş
            result_list.append(base[1:])    
        
        if word[-1] == 'n':
            c_form = 'ç' + base # sevinç ? (Aslında inç ayrı ek ama burada variant olarak eklenmiş)
            result_list.append(c_form)
    
    return result_list
"""


# ============================================================================
# VERB TO NOUN SUFFIXES (v2n)
# ============================================================================
    
# --- STANDART YAPIM EKLERİ (DERIVATIONAL - Group 10) ---
# Bunlar isim/sıfat kökü oluşturur, üzerine çoğul/iyelik gelebilir.


##TODO form for form for constofactattive koy, agan eğen biçimleri için



## bu ekin şu anki hali yanlış, aslında bu ek fiilken i eki alıp isim olmuş sözcüklere eklenir. 
# bak-  bakı     bakı yorum.  ölü yorum , ölü oluyorum tarzında 
# ama bekleyorum vs gibi örneklerdeki düzensizliklerden ötürü şimdilik böyle devam edecek 
## olumsuz fiile ekleşebilenler.
continuous_iyor = Suffix(
    "continuous_iyor", 
    "iyor", 
    Type.VERB,  # Fiile gelir
    Type.NOUN,  # İsimleştirir (üzerine şahıs eki alır: geliyor-um)
    form_function=form_for_continuous_iyor, 
    major_harmony=HasMajorHarmony.Yes, # Fonksiyon hallediyor
    minor_harmony=HasMinorHarmony.Yes, # Fonksiyon hallediyor
    group=SuffixGroup.DERIVATIONAL #  derivational değil , değiştirmek için diğer doyaları halletmeli
)
wish_suffix = Suffix("wish_suffix", "se", Type.VERB, Type.NOUN,major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.PREDICATIVE, is_unique=True)

VERB2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
VERB2NOUN = VERB2NOUN + PARICIPLES + INFINITIVES + GERUNDS  +NOUNIFIERS