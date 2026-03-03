from util.suffix import Suffix, Type,  SuffixGroup

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
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)

    base = base+"yor"
    return [base]

# ============================================================================
# VERB TO NOUN SUFFIXES (v2n)
# ============================================================================
    
# --- STANDART YAPIM EKLERİ (DERIVATIONAL - Group 10) ---
# Bunlar isim/sıfat kökü oluşturur, üzerine çoğul/iyelik gelebilir.





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
    has_major_harmony=True, # Fonksiyon hallediyor
    has_minor_harmony=True, # Fonksiyon hallediyor
    group=SuffixGroup.DERIVATIONAL #  derivational değil , değiştirmek için diğer doyaları halletmeli
)
wish_suffix = Suffix("wish_suffix", "se", Type.VERB, Type.NOUN,has_major_harmony=True, has_minor_harmony=False, group=SuffixGroup.PREDICATIVE, is_unique=True)

VERB2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
VERB2NOUN = VERB2NOUN + PARICIPLES + INFINITIVES + GERUNDS  +NOUNIFIERS