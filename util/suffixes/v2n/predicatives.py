from util.suffix import Suffix, Type,  SuffixGroup

VOWELS = ["a","e","ı","i","o","ö","u","ü"]

class Predicative(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.NOUN,
                makes=Type.NOUN,
                has_major_harmony=True, 
                has_minor_harmony=True,  # Set to None to detect if the user passed a value
                needs_y_buffer=False, 
                form_function=None,
                group=SuffixGroup.PREDICATIVE, 
                is_unique=True):


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



def form_for_continuous_iyor(word, suffix_obj):
    base = "i"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)

    base = base+"yor"
    return [base]

## bu ekin şu anki hali yanlış, aslında bu ek fiilken i eki alıp isim olmuş sözcüklere eklenir. 
# bak-  bakı     bakı yorum.  ölü yorum , ölü oluyorum tarzında 
# ama bekleyorum vs gibi örneklerdeki düzensizliklerden ötürü şimdilik böyle devam edecek 
## olumsuz fiile ekleşebilenler.
continuous_iyor = Predicative("continuous_iyor" , "iyor", form_function=form_for_continuous_iyor,)
wish_suffix     = Predicative("wish_suffix"     , "se")
pasttense_di    = Predicative("pasttense_di"    , "di")


PREDICATIVES = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]