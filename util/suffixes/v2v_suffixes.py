

from util.suffix import Suffix, Type, HasMajorHarmony,HasMinorHarmony



def form_for_active_er(word,suffix_obj):

    pass


def form_for_active_it(word, suffix_obj):
    """
    Form function for active_it suffix
    - Default forms include 'it' with harmony
    - Add 't' when the last letter is a vowel
    - Add 't' when the last 2 letters are ir/ur/ür/ır
    - Apply harmony stages
    """
    result_list = []
    
    # Base form with harmonies
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    # Add 't' variant 
    t_form = 't'
    result_list.append(t_form)
    
    # Add 't' variant when last 2 letters are ir/ur/ür/ır
    if len(word) >= 2:
        last_two = word[-2:]
        if last_two in ['ir', 'ur', 'ür', 'ır']:
            t_form = 't'
            t_form = Suffix._apply_consonant_hardening(word, t_form)
            if t_form not in result_list:
                result_list.append(t_form)
    
    return result_list


def form_for_passive_il(word, suffix_obj):
    """
    Form function for passive_il suffix
    - Default form 'il' with harmony
    - Add 'l' when the last letter is a vowel
    - Becomes 'in' when the last letter of the word is 'n'
    - Apply harmony
    """
    result_list = []
    
    # Base form with harmonies
    
    
    # If last letter is 'n', use 'in' instead
    if  word[-1] == 'l':
        in_form = 'in'
        in_form = Suffix._apply_major_harmony(word, in_form, suffix_obj.major_harmony)
        in_form = Suffix._apply_minor_harmony(word, in_form, suffix_obj.minor_harmony)
        result_list.append(in_form)
    else:
        base = suffix_obj.suffix
        base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
        base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
        base = Suffix._apply_consonant_hardening(word, base)
        result_list.append(base)
    
    # Add 'l' variant when last letter is vowel
    if word and word[-1] in ["a","e","ı","i","o","ö","u","ü"]:
        l_form = 'l'
        result_list.append(l_form)
    
    return result_list


# ============================================================================
# VERB TO VERB SUFFIXES (v2v)
# ============================================================================

reflexive_is = Suffix("reflexive_is", "iş", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
reflexive_ik = Suffix("reflexive_ik", "ik", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
active_it = Suffix("active_it", "it", Type.VERB, Type.VERB, form_function=form_for_active_it, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
active_dir = Suffix("active_dir", "dir", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
active_ir = Suffix("active_ir", "ir", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
passive_il = Suffix("passive_il", "il", Type.VERB, Type.VERB, form_function=form_for_passive_il, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
reflexive_in = Suffix("reflexive_in", "in", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
negative_me = Suffix("negative_me", "me", Type.VERB, Type.VERB, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)



VERB2VERB =   [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]