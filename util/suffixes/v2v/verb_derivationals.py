from util.suffix import Suffix, Type, SuffixGroup

# ============================================================================
# FORM FUNCTIONS
# ============================================================================

class VerbDerivationalSuffix(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.VERB,
                has_major_harmony=True, 
                has_minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=False, 
                group=SuffixGroup.VERB_DERIVATIONAL, 
                is_unique=False):
        
        # Dynamic default assignment for minor harmony
        if minor_harmony is None:
            # If the suffix contains any narrow vowel, it defaults to having minor harmony
            if any(vowel in suffix for vowel in ['ı', 'i', 'u', 'ü']): # only i is enough bc of the standart narrow front vowel converntion
                minor_harmony = True
            else:
                minor_harmony = False

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=comes_to,
            makes=makes,
            form_function=None, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )


reflexive_ik    = VerbDerivationalSuffix("reflexive_ik"    , "ik" )
reflexive_is    = VerbDerivationalSuffix("reflexive_is"    , "iş" )
active_it       = VerbDerivationalSuffix("active_it"       , "it" ) ## çıkart anlayamaz,
active_dir      = VerbDerivationalSuffix("active_dir"      , "dir")
active_ir       = VerbDerivationalSuffix("active_ir"       , "ir" )
active_er       = VerbDerivationalSuffix("active_er"       , "er" )
passive_il      = VerbDerivationalSuffix("passive_il"      , "il" )
reflexive_in    = VerbDerivationalSuffix("reflexive_in"    , "in" )
randomative_ele = VerbDerivationalSuffix("randomative_ele" , "ele")

VERB_DERIVATIONALS = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]