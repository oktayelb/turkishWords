from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd
VOWELS = ["a","e","ı","i","o","ö","u","ü"]

class Copula(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.NOUN,
                makes=Type.NOUN,
                major_harmony=HasMajorHarmony.Yes, 
                minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=True, 
                group=SuffixGroup.PREDICATIVE, 
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
            comes_to=Type.NOUN,
            makes=Type.NOUN,
            form_function=None, # Force the use of the overridden _default_form
            major_harmony=major_harmony,
            minor_harmony=minor_harmony,
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
        base = suffix_obj.suffix
        candidates = []
        # Apply standard harmonies using the parent class's static methods
        base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
        base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
        base = Suffix._apply_consonant_hardening(word, base)
        
        candidates.append(base)  # Always include the base form
        
        if word and word[-1] in ["a","e","ı","i","o","ö","u","ü"]:  # If the last character is a vowel, we need to consider buffer consonants
            if suffix_obj.needs_y_buffer:
                candidates.append('y' + base)
        
        return candidates


def form_for_if_suffix  (word, suffix_obj):

    base= "se"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)

    if(word and word[-1] in VOWELS):
        base = "y" + base
    
    return [base]

def form_for_pasttense_noundi(word,suffix_obj):
    # Hem "doktor-du" (Ek fiil) hem "gel-ecek-ti" (Hikaye birleşik zaman)
    di_base = "di"
    di_base = Suffix._apply_major_harmony(word, di_base, suffix_obj.major_harmony)
    di_base = Suffix._apply_minor_harmony(word, di_base, suffix_obj.minor_harmony)
    di_base = Suffix._apply_consonant_hardening(word, di_base)
    
    if word and word[-1] in VOWELS:
        ydi_base = "y" + di_base
        return [ydi_base, di_base]

    return [di_base]


def form_for_copula_mis (word, suffix_obj):
    base = "miş"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    
    if word and word[-1] in VOWELS:
        base = "y" + base

    return [base]


nounaorist_dir =  Copula("nounaorist_dir", "dir",needs_y_buffer=False)
## bu both ayrılabilir
pasttense_noundi = Copula("pasttense_noundi", "di", comes_to=Type.BOTH)
if_suffix = Copula("if_suffix", "se")
copula_mis = Copula("copula_mis", "miş")