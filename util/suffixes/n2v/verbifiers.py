from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
## COMPLETE
class VerbifierSuffix(Suffix):
    def __init__(self, name, suffix, 
                 comes_to=Type.NOUN,
                 makes=Type.VERB,
                 major_harmony=HasMajorHarmony.Yes, 
                 minor_harmony=None, 
                 needs_y_buffer=False, 
                 form_function=None,
                 group=SuffixGroup.N2V_DERIVATIONAL, 
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
            form_function=form_function, 
            major_harmony=major_harmony,
            minor_harmony=minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )


absentative_se = VerbifierSuffix("absentative_se", "se")
onomatopea_de  = VerbifierSuffix("onomatopea_de",  "de")
verbifier_e    = VerbifierSuffix("verbifier_e",    "e" )
aplicative_le  = VerbifierSuffix("aplicative_le",  "le")
verbifier_ik   = VerbifierSuffix("verbifier_ik",   "ik")


VERBIFIERS = [ 
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]