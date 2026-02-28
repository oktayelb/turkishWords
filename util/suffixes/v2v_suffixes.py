from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

# ============================================================================
# FORM FUNCTIONS
# ============================================================================

class VerbDerivationalSuffix(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.VERB,
                 major_harmony=HasMajorHarmony.Yes, 
                    minor_harmony=None,  # Set to None to detect if the user passed a value
                 needs_y_buffer=False, 
                 group=SuffixGroup.VERB_DERIVATIONAL, 
                 is_unique=False):
        
        # Dynamic default assignment for minor harmony
        if minor_harmony is None:
            # If the suffix contains any narrow vowel, it defaults to having minor harmony
            if any(vowel in suffix for vowel in ['ı', 'i', 'u', 'ü']): # only i is enough bc of the standart narrow front vowel converntion
                minor_harmony = HasMinorHarmony.Yes
            else:
                minor_harmony = HasMinorHarmony.No

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=comes_to,
            makes=makes,
            form_function=None, # Force the use of the overridden _default_form
            major_harmony=major_harmony,
            minor_harmony=minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )


def form_for_possibiliative_ebil(word, suffix_obj):
    e_base = "e"
    e_base = Suffix._apply_major_harmony(word, e_base, suffix_obj.major_harmony)

    if word and word[-1] not in wrd.VOWELS:
        return [e_base + "bil"]
    else:
        return ["y" + e_base + "bil"]    
# ============================================================================
# VERB TO VERB SUFFIXES (v2v) - Hepsi VERB_DERIVATIONAL (Grup 10)
# ============================================================================



reflexive_ik    = VerbDerivationalSuffix("reflexive_ik"    , "ik" )
reflexive_is    = VerbDerivationalSuffix("reflexive_is"    , "iş" )
active_it       = VerbDerivationalSuffix("active_it"       , "it" ) ## çıkart anlayamaz,
active_dir      = VerbDerivationalSuffix("active_dir"      , "dir")
active_ir       = VerbDerivationalSuffix("active_ir"       , "ir" )
active_er       = VerbDerivationalSuffix("active_er"       , "er" )
passive_il      = VerbDerivationalSuffix("passive_il"      , "il" )
reflexive_in    = VerbDerivationalSuffix("reflexive_in"    , "in" )
randomative_ele = VerbDerivationalSuffix("randomative_ele" , "ele")

### Buranın ayrılması laaızm daha temiz bir mimari... ebilmek evermek eyazmak şeylerini halletmeli.
possibiliative_ebil = Suffix("possibilitative_ebil", "ebil", Type.VERB, Type.VERB, form_function=form_for_possibiliative_ebil, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No,needs_y_buffer=True, group=SuffixGroup.VERB_COMPOUND, is_unique=True)

# Olumsuzluk (Negative): Gel-me.
# Bu ek Yapım eklerinden sonra gelir, ama Çekim eklerinden önce gelir.
# Hiyerarşide VERB_DERIVATIONAL grubunda kalabilir ama is_unique=True olmalı.
negative_me = Suffix("negative_me", "me", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.VERB_NEGATING, is_unique=True)

negative_consto = Suffix("negative_consto", "eme", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes,needs_y_buffer=True, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.VERB_NEGATING, is_unique=True)

VERB2VERB =   [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]