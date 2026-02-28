from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

class Plural(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.NOUN,
                makes=Type.NOUN,
                 major_harmony=None, 
                 minor_harmony=None, 
                 needs_y_buffer=False, 
                 form_function=None,
                 group=SuffixGroup.PLURAL, 
                 is_unique=False):
        
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
            form_function=form_function ,
            major_harmony=major_harmony,
            minor_harmony=minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )






plural_ler  = Plural("plural_ler", "ler")

PLURALS = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]