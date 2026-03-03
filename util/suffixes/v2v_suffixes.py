from util.suffix import Suffix, Type,  SuffixGroup
import util.word_methods as wrd


from util.suffixes.v2v.verb_derivationals import VERB_DERIVATIONALS
from util.suffixes.v2v.verb_negative import VERB_NEGATIVES


def form_for_possibiliative_ebil(word, suffix_obj):
    e_base = "e"
    e_base = Suffix._apply_has_major_harmony(word, e_base, suffix_obj.has_major_harmony)

    if word and word[-1] not in wrd.VOWELS:
        return [e_base + "bil"]
    else:
        return ["y" + e_base + "bil"]    
# ============================================================================
# VERB TO VERB SUFFIXES (v2v) - Hepsi VERB_DERIVATIONAL (Grup 10)
# ============================================================================

### Buranın ayrılması laaızm daha temiz bir mimari... ebilmek evermek eyazmak şeylerini halletmeli.
possibiliative_ebil = Suffix("possibilitative_ebil", "ebil", Type.VERB, Type.VERB, form_function=form_for_possibiliative_ebil, has_major_harmony=True, has_minor_harmony=False,needs_y_buffer=True, group=SuffixGroup.VERB_COMPOUND, is_unique=True)

# Olumsuzluk (Negative): Gel-me.
# Bu ek Yapım eklerinden sonra gelir, ama Çekim eklerinden önce gelir.
# Hiyerarşide VERB_DERIVATIONAL grubunda kalabilir ama is_unique=True olmalı.

VERB2VERB =   [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]

VERB2VERB = VERB2VERB + VERB_DERIVATIONALS+ VERB_NEGATIVES