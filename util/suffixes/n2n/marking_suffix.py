from util.suffix import Suffix, Type, SuffixGroup


class MarkingKi(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.NOUN,
                makes=Type.NOUN,
                 has_major_harmony=True, 
                 has_minor_harmony=True, 
                 needs_y_buffer=False, 
                 form_function=None,
                 group=SuffixGroup.MARKING_KI, 
                 is_unique=True):
        
        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=comes_to,
            makes=makes,
            form_function=form_function ,
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )

###form fonksiyonu mu dizsek?


#### evdeki    TRUE
#### evdenki   FALSE ilginç
#### eviki     FALSE
#### evinki    TRUE
#### eveki     FALSE
## bu yılkiler TRUE
####

marking_ki = MarkingKi("marking_ki", "ki")

MARKINGS = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]