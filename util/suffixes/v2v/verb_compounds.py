from util.suffix import Suffix, Type,  SuffixGroup
import util.word_methods as wrd

class CompoundVerb(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.VERB,
                has_major_harmony=True, 
                has_minor_harmony=False, 
                needs_y_buffer=True, 
                group=SuffixGroup.VERB_COMPOUND, 
                is_unique=False,
                form_function=None):

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=comes_to,
            makes=makes,
            form_function=form_function, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )
    @staticmethod
    
    def _default_form(word, suffix_obj):
        """
        Overridden default form handler specifically for Case Suffixes.
        Consolidates 'n' and 'y' buffer logic for Turkish nominal inflection.
        
        """
        
        ebase = suffix_obj.suffix[0]
        candidates = []
        # Apply standard harmonies using the parent class's static methods
        ebase = Suffix._apply_major_harmony(word, ebase, suffix_obj.has_major_harmony)
        ebase = Suffix._apply_minor_harmony(word, ebase, suffix_obj.has_minor_harmony)
        
        
        if  word[-1] in ["a","e","ı","i","o","ö","u","ü"]:  # If the last character is a vowel, we need to consider buffer consonants
            ebase = "y" + ebase  # Default buffer consonant is 'y'

        
        return [ebase + suffix_obj.suffix[1:]]  # Append the rest of the suffix after the first character
 
# ============================================================================
# VERB TO VERB SUFFIXES (v2v) - Hepsi VERB_DERIVATIONAL (Grup 10)
# ============================================================================

### Buranın ayrılması laaızm daha temiz bir mimari... ebilmek evermek eyazmak şeylerini halletmeli.
possibiliative_ebil   = CompoundVerb("possibilitative_ebil", "ebil")
almostative_eyazmak   = CompoundVerb("almostative_eyazmak", "eyazmak")
continuative_edurmak  = CompoundVerb("continuative_edurmak", "edurmak")
remainmative_kalmak   = CompoundVerb("remainmative_ekalmak", "ekalmak")
persistive_egelmek    = CompoundVerb("persistive_egelmek", "egelmek")
suddenative_ivermek   = CompoundVerb("suddenative_ivermek", "ivermek")

#iyoru buraya bir şekilde koysak mı?


VERB_COMPOUNDS = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]