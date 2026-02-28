from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

class Gerund(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.NOUN,
                major_harmony=HasMajorHarmony.Yes, 
                minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=None, 
                group=SuffixGroup.DERIVATIONAL_LOCKING, 
                is_unique=False):
        
        # Dynamic default assignment for minor harmony
        if minor_harmony is None:
            # If the suffix contains any narrow vowel, it defaults to having minor harmony
            if any(vowel in suffix for vowel in ['ı', 'i', 'u', 'ü']): # only i is enough bc of the standart narrow front vowel converntion
                minor_harmony = HasMinorHarmony.Yes
            else:
                minor_harmony = HasMinorHarmony.No

        if needs_y_buffer is None:
            if suffix[0] in ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']:
                needs_y_buffer = True
            else:
                needs_y_buffer = False
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

#gidereknten kullanımını kabul etmeyeceğiz.
adverbial_erek    = Gerund("adverbial_erek", "erek")
adverbial_ince    = Gerund("adverbial_ince", "ince" )
adverbial_ip      = Gerund("adverbial_ip", "ip")
adverbial_e       = Gerund("adverbial_e", "e")
adverbial_dikçe   = Gerund("adverbial_dikçe", "dikçe")
##Bunu napacağız incele
nondoing_meden = Suffix("adverbial_meden", "meden", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL_LOCKING)


GERUNDS = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]

