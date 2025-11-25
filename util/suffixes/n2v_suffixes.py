
from util.suffix import Suffix, Type, HasMajorHarmony,HasMinorHarmony



def form_for_verbifier_e(word, suffix_obj):
    """
    Form function for verbifier_e suffix
    - This suffix may change the root
    - Apply standard harmony rules
    """
    from util.suffix import Suffix
    
    result = suffix_obj.suffix
    result = Suffix._apply_major_harmony(word, result, suffix_obj.major_harmony)
    result = Suffix._apply_consonant_hardening(word, result)
    
    result_list = [result]
    
    # Add buffer variants if needed
    if Suffix._should_add_buffer_variants(word, result):
        result_list.append('y' + result)
    
    return result_list



# ============================================================================
# NOUN TO VERB SUFFIXES (n2v)
# ============================================================================

absentative_se = Suffix("absentative_se", "se", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
onomatopea_de  = Suffix("onomatopea_de",  "de", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
verbifier_e    = Suffix("verbifier_e",     "e", Type.NOUN, Type.VERB, form_function=form_for_verbifier_e, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
aplicative_le  = Suffix("aplicative_le",  "le", Type.NOUN, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)


NOUN2VERB = [
    absentative_se,
    onomatopea_de,
    verbifier_e,
    aplicative_le
]