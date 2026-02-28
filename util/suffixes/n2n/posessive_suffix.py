from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup

class PosessiveSuffix(Suffix):
    def __init__(self, name, suffix, 
                 comes_to=Type.NOUN,
                 makes=Type.NOUN,
                 major_harmony=HasMajorHarmony.Yes, 
                 minor_harmony=HasMinorHarmony.Yes,  # Set to None to detect if the user passed a value
                 needs_y_buffer=False, 
                 form_function=None,
                 group=SuffixGroup.POSSESSIVE, 
                 is_unique=False):
        

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=Type.NOUN,
            makes=Type.NOUN,
            form_function=form_function, # Force the use of the overridden _default_form
            major_harmony=major_harmony,
            minor_harmony=minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )

def form_for_posessive_3sg(word, suffix_obj):
    base = "i"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    
    if word[-1] in ["a","e","ı","i","o","ö","u","ü"]:
        base = "s" + base

    return [base]


posessive_1sg = PosessiveSuffix("posessive_1sg", "im")
posessive_2sg = PosessiveSuffix("posessive_2sg", "in")
posessive_3sg = PosessiveSuffix("posessive_3sg", "i", form_function=form_for_posessive_3sg)
posessive_1pl = PosessiveSuffix("posessive_1pl", "imiz")
posessive_2pl = PosessiveSuffix("posessive_2pl", "iniz")
posessive_3pl = PosessiveSuffix("posessive_3pl", "leri")

POSESSIVE_SUFFIX = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]