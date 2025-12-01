
from util.suffix import Suffix, Type, HasMajorHarmony,HasMinorHarmony

VOWELS = ["a","e","ı","i","o","ö","u","ü"]

def form_for_nounifier_ecek(word, suffix_obj):
    """
    Form function for nounifier_ecek suffix
    - Returns harmonized versions of ecek, yecek, cek
    """
    from util.suffix import Suffix
    result_list = []
    
    # Base form: ecek
    base = "ecek"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    # Add buffer variants for vowel-ending words
    if word and word[-1] in VOWELS:
        # yecek variant
        y_form = 'y' + base
        result_list.append(y_form)
        
        # cek variant (vowel dropped)
        short_form = base[1:]  # Remove 'e'
        result_list.append(short_form)
    
    return result_list


def form_for_factative_ir(word, suffix_obj):
    """
    Form function for factative_ir suffix (merges er and ir)
    - Default forms: er, ir
    - Apply harmony and vowel drop
    """
    from util.suffix import Suffix
    import util.word_methods as wrd
    result_list = []
    
    #if last two letters of the word is me or ma
    if word[-2:] in ["ma","me"]:
        if wrd.can_be_verb(word[ :-2]):
                z_base = "z"   #geniş zamanın olumsuzu için ##TODO BURASI BIRAZ SIKINTILI
                result_list.append(z_base)

    # ir form with harmony
    ir_base = 'ir'
    ir_base = Suffix._apply_major_harmony(word, ir_base, suffix_obj.major_harmony)
    ir_base = Suffix._apply_minor_harmony(word, ir_base, suffix_obj.minor_harmony)
    ir_base = Suffix._apply_consonant_hardening(word, ir_base)
    result_list.append(ir_base)
    
    # er form with harmony
    er_base = 'er'
    er_base = Suffix._apply_major_harmony(word, er_base, suffix_obj.major_harmony)
    er_base = Suffix._apply_consonant_hardening(word, er_base)
    if er_base not in result_list:
        result_list.append(er_base)
    
    # Vowel drop variant for vowel-ending words
    if word and word[-1] in VOWELS:
        r_form = 'r'
        if r_form not in result_list:
            result_list.append(r_form)
    
    return result_list


def form_for_toolative_ek(word, suffix_obj):
    """
    Form function for toolative_ek suffix
    - Never drop the vowel
    - Apply vowel harmony only
    """
    from util.suffix import Suffix
    
    result = suffix_obj.suffix
    result = Suffix._apply_major_harmony(word, result, suffix_obj.major_harmony)
    result = Suffix._apply_consonant_hardening(word, result)
    
    return [result]


def form_for_nounifier_iş(word, suffix_obj):
    """
    Form function for nounifier_iş suffix
    - Default: iş with harmony
    - If last letter is consonant: add yiş, ğiş, and ş
    - If last letter is 'n': also append ç
    """
    from util.suffix import Suffix
    result_list = []
    
    # Base form with harmony
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        # If last letter is vowel, add buffer variants
        if word[-1] in VOWELS:
            result_list.append('y' + base)
            result_list.append('ğ' + base)
            result_list.append(base[1:])  # ş variant
        
        # If last letter is 'n', add ç variant
        if word[-1] == 'n':
            c_form = 'ç' + base
            result_list.append(c_form)
    
    return result_list


def form_for_perfectative_ik(word, suffix_obj):
    """
    Form function for perfectative_ik suffix
    - Default form: ik with harmony
    - If last letter is vowel, or has 'n' or 'r': also add 'k'
    - Add yik and ğik variants
    """
    from util.suffix import Suffix
    result_list = []
    
    # Base form with harmony
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        # If last letter is vowel, add buffer variants
        if word[-1] in VOWELS:
            result_list.append('y' + base)
            result_list.append('ğ' + base)
            result_list.append(base[1:])  # k variant
        
        # If last letter is 'n' or 'r', add 'k' variant
        if word[-1] in ['n', 'r']:
            k_form = 'k'
            k_form = Suffix._apply_consonant_hardening(word, k_form)
            if k_form not in result_list:
                result_list.append(k_form)
    
    return result_list


def form_for_nounifier_i(word, suffix_obj):
    """
    Form function for nounifier_i suffix
    - If last letter is 'n', result list should also have empty string
    - Needs y buffer and goes through all harmonies
    """
    from util.suffix import Suffix
    result_list = []
    
    # Base form with harmony
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    # Add buffer variants for vowel-ending words
    if word and word[-1] in VOWELS:
        result_list.append('y' + base)
        result_list.append('ğ' + base)
    
    # If last letter is 'n', add empty string (suffix drops)
    if word and word[-1] == 'n':
        result_list.append('')
    
    return result_list


def form_for_nounifier_inti(word, suffix_obj):
    """
    Form function for nounifier_inti suffix
    - Default: inti with harmony
    - Add vowel-reduced form if last letter is vowel
    - Add 'ti' when last two letters are in/ın/un/ün
    """
    from util.suffix import Suffix
    result_list = []
    
    # Base form with harmony
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        # Vowel-reduced form if last letter is vowel
        if word[-1] in VOWELS:
            reduced = base[1:]  # Remove first vowel
            result_list.append(reduced)
        
        # 'ti' variant when last two letters are in/ın/un/ün
        if len(word) >= 2:
            last_two = word[-2:]
            if last_two in ['in', 'ın', 'un', 'ün']:
                ti_form = 'ti'
                ti_form = Suffix._apply_major_harmony(word, ti_form, suffix_obj.major_harmony)
                ti_form = Suffix._apply_consonant_hardening(word, ti_form)
                if ti_form not in result_list:
                    result_list.append(ti_form)
    
    return result_list



def form_for_toolifier_geç(word, suffix_obj):
    """
    Form function for toolifier_geç suffix (merges geç and eç)
    - Default: geç with harmony
    - Also add eç variant
    """
    from util.suffix import Suffix
    result_list = []
    
    # geç form with harmony
    gec_base = suffix_obj.suffix
    gec_base = Suffix._apply_major_harmony(word, gec_base, suffix_obj.major_harmony)
    gec_base = Suffix._apply_minor_harmony(word, gec_base, suffix_obj.minor_harmony)
    gec_base = Suffix._apply_consonant_hardening(word, gec_base)
    result_list.append(gec_base)
    
    # eç form with harmony
    ec_base = 'eç'
    ec_base = Suffix._apply_major_harmony(word, ec_base, suffix_obj.major_harmony)
    ec_base = Suffix._apply_minor_harmony(word, ec_base, suffix_obj.minor_harmony)
    ec_base = Suffix._apply_consonant_hardening(word, ec_base)
    if ec_base not in result_list:
        result_list.append(ec_base)
    
    return result_list

# ============================================================================
# VERB TO NOUN SUFFIXES (v2n)
# ============================================================================
    
infinitive_me = Suffix("infinitive_me", "me", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
infinitive_mek = Suffix("infinitive_mek", "mek", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
factative_en = Suffix("factative_en", "en", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
toolative_ek = Suffix("toolative_ek", "ek", Type.VERB, Type.NOUN, form_function= form_for_toolative_ek, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
constofactative_gen = Suffix("constofactative_gen", "gen", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
constofactative_gin = Suffix("constofactative_gin", "gin", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_iş = Suffix("nounifier_iş", "iş", Type.VERB, Type.NOUN, form_function= form_for_nounifier_iş, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
perfectative_ik = Suffix("perfectative_ik", "ik", Type.VERB, Type.NOUN, form_function= form_for_perfectative_ik, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
nounifier_i = Suffix("nounifier_i", "i", Type.VERB, Type.NOUN, form_function= form_for_nounifier_i, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True)
nounifier_gi = Suffix("nounifier_gi", "gi", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_ge = Suffix("nounifier_ge", "ge", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
nounifier_im = Suffix("nounifier_im", "im", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_in = Suffix("nounifier_in", "in", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_it = Suffix("nounifier_it", "it", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_inç = Suffix("nounifier_inç", "inç", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
nounifier_inti = Suffix("nounifier_inti", "inti", Type.VERB, Type.NOUN, form_function= form_for_nounifier_inti, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
toolifier_geç = Suffix("toolifier_geç", "geç", Type.VERB, Type.NOUN, form_function= form_for_toolifier_geç, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
adjectiative_ay_v2n = Suffix("adjectiative_ay", "ay", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
nounifier_anak = Suffix("nounifier_anak", "anak", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
nounifier_amak = Suffix("nounifier_amak", "amak", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)

adjectifier_dik = Suffix("adjectifier_dik", "dik", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)
adverbial_e    = Suffix("adverbial_e", "e", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
adverbial_erek = Suffix("adverbial_erek", "erek", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
adverbial_ip   = Suffix("adverbial_ip", "ip", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)

adverbial_esi  = Suffix("adverbial_esi", "esi", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
## bunlar aslında zaman gösterme çekimi olarak değil de isimleştirici olarak analiz edilmeli. belki ir dışındakiler
nounifier_ecek = Suffix("nounifier_ecek", "ecek", Type.VERB, Type.NOUN, form_function= form_for_nounifier_ecek, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True)
##seni bölsek mi
neverfactative_mez = Suffix("neverfactative_mez", "mez", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No)
factative_ir = Suffix("factative_ir", "ir", Type.VERB, Type.NOUN, form_function= form_for_factative_ir, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)

##
pastfactative_miş = Suffix("pastfactative_miş", "miş", Type.BOTH, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes)

VERB2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]