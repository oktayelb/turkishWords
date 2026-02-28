from util import suffix
from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup


class Infinitive(Suffix):
    def __init__(self, name, suffix, 
                 comes_to=Type.VERB,
                 makes=Type.NOUN,
                 major_harmony=HasMajorHarmony.Yes, 
                 minor_harmony=HasMinorHarmony.Yes,  # Set to None to detect if the user passed a value
                 needs_y_buffer=None, 
                 form_function=None,
                 group=SuffixGroup.DERIVATIONAL, 
                 is_unique=False):
        
        if needs_y_buffer is None:
                if suffix[0] in ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']:
                    needs_y_buffer = True
                else:
                    needs_y_buffer = False

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=Type.VERB,
            makes=Type.NOUN,
            form_function=form_function, # Force the use of the overridden _default_form
            major_harmony=major_harmony,
            minor_harmony=minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )



infinitive_me     = Infinitive("infinitive_me"  , "me"  )
infinitive_mek    = Infinitive("infinitive_mek" , "mek" )
nounifier_iş      = Infinitive("nounifier_iş"   , "iş"  )

INFINITIVES = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]