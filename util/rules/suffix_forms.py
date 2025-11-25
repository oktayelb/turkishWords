"""
Template form functions for all suffixes in suffixes.py
Each function can be customized with specific rules for that suffix.
"""
import util.word_methods as wrd

# ============================================================================
# VERB TO VERB SUFFIXES (v2v)
# ============================================================================

def form_for_active_er(word,suffix_obj):
    from util.suffixes import Suffix
    
    base = "er"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)

    return [base]


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
    from util.suffixes import Suffix
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
    from util.suffixes import Suffix
    
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
    if word and word[-1] in wrd.VOWELS:
        l_form = 'l'
        result_list.append(l_form)
    
    return result_list



# ============================================================================
# NOUN TO NOUN SUFFIXES (n2n)
# ============================================================================

# Most n2n suffixes use default behavior, so we'll keep them simple
# or use the default form function


# ============================================================================
# NOUN TO VERB SUFFIXES (n2v)
# ============================================================================

def form_for_verbifier_e(word, suffix_obj):
    """
    Form function for verbifier_e suffix
    - This suffix may change the root
    - Apply standard harmony rules
    """
    from util.suffixes import Suffix
    
    result = suffix_obj.suffix
    result = Suffix._apply_major_harmony(word, result, suffix_obj.major_harmony)
    result = Suffix._apply_consonant_hardening(word, result)
    
    result_list = [result]
    
    # Add buffer variants if needed
    if Suffix._should_add_buffer_variants(word, result):
        result_list.append('y' + result)
    
    return result_list


# ============================================================================
# VERB TO NOUN SUFFIXES (v2n)
# ============================================================================

def form_for_nounifier_ecek(word, suffix_obj):
    """
    Form function for nounifier_ecek suffix
    - Returns harmonized versions of ecek, yecek, cek
    """
    from util.suffixes import Suffix
    result_list = []
    
    # Base form: ecek
    base = "ecek"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    # Add buffer variants for vowel-ending words
    if word and word[-1] in wrd.VOWELS:
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
    from util.suffixes import Suffix
    result_list = []
    
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
    if word and word[-1] in wrd.VOWELS:
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
    from util.suffixes import Suffix
    
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
    from util.suffixes import Suffix
    result_list = []
    
    # Base form with harmony
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        # If last letter is vowel, add buffer variants
        if word[-1] in wrd.VOWELS:
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
    from util.suffixes import Suffix
    result_list = []
    
    # Base form with harmony
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        # If last letter is vowel, add buffer variants
        if word[-1] in wrd.VOWELS:
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
    from util.suffixes import Suffix
    result_list = []
    
    # Base form with harmony
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    # Add buffer variants for vowel-ending words
    if word and word[-1] in wrd.VOWELS:
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
    from util.suffixes import Suffix
    result_list = []
    
    # Base form with harmony
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        # Vowel-reduced form if last letter is vowel
        if word[-1] in wrd.VOWELS:
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
    from util.suffixes import Suffix
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
# NOUN TO NOUN SUFFIXES (n2n)
# ============================================================================

def form_for_pasttense_noundi(word,suffix_obj):

    from util.suffixes import Suffix

    di_base = "di"
    di_base = Suffix._apply_major_harmony(word, di_base, suffix_obj.major_harmony)
    di_base = Suffix._apply_minor_harmony(word, di_base, suffix_obj.minor_harmony)
    di_base = Suffix._apply_consonant_hardening(word, di_base)
    
    dik_base = "dik"
    dik_base = Suffix._apply_major_harmony(word, dik_base, suffix_obj.major_harmony)
    dik_base = Suffix._apply_minor_harmony(word, dik_base, suffix_obj.minor_harmony)
    dik_base = Suffix._apply_consonant_hardening(word, dik_base)

    if word[-1] in wrd.VOWELS:
        di_base = "y" + di_base
        dik_base = "y" + dik_base

    return [dik_base,di_base]

def form_for_abstractifier_iyat(word, suffix_obj):
    result_list = ["iye","iyet","iyat","at","et"]
    
    return result_list


def form_for_adverbial_erek (word, suffix_obj):
    from util.suffixes import Suffix

    base = "erek"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in wrd.VOWELS:
        base = "y" + base

    return [base]


def form_for_dative_archaic_ke(word, suffix_obj):
    """Form function for dative_archaic_ke suffix"""
    pass


def form_for_locative_den(word, suffix_obj):
    """Form function for locative_den suffix"""
    pass


def form_for_dative_e(word, suffix_obj):
    """Form function for dative_e suffix"""
    pass


def form_for_ablative_de(word, suffix_obj):
    """Form function for ablative_de suffix"""
    from util.suffixes import Suffix

    base = "de"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in ["ı","i","u","ü"]:   # akuzatif
        base = "n" + base

    return [base]


def form_for_plural_ler(word, suffix_obj):
    """Form function for plural_ler suffix"""
    pass


def form_for_counting_er(word, suffix_obj):
    """Form function for counting_er suffix"""
    pass


def form_for_cooperative_daş(word, suffix_obj):
    """Form function for cooperative_daş suffix"""
    pass


def form_for_relative_ce(word, suffix_obj):
    """Form function for relative_ce suffix"""
    pass


def form_for_relative_sel(word, suffix_obj):
    """Form function for relative_sel suffix"""
    pass


def form_for_dimunitive_ek_archaic(word, suffix_obj):
    """Form function for dimunitive_ek_archaic suffix"""
    pass


def form_for_approximative_imtrak(word, suffix_obj):
    """Form function for approximative_imtrak suffix"""
    pass


def form_for_accusative(word, suffix_obj):
    """Form function for accusative suffix"""
    pass


def form_for_pluralizer_archaic_iz(word, suffix_obj):
    """Form function for pluralizer_archaic_iz suffix"""
    pass


def form_for_posessive_im(word, suffix_obj):
    """Form function for posessive_im suffix"""
    pass


def form_for_posessive_in(word, suffix_obj):
    """Form function for posessive_in suffix"""
    pass


def form_for_dimunitive_cik(word, suffix_obj):
    """Form function for dimunitive_cik suffix"""
    pass


def form_for_actor_ci(word, suffix_obj):
    """Form function for actor_ci suffix"""
    pass


def form_for_ordinal_inci(word, suffix_obj):
    """Form function for ordinal_inci suffix"""
    pass


def form_for_approximative_si(word, suffix_obj):
    
    from util.suffixes import Suffix
    base1 = "imsi"
    base2 = "si"

    

    base1 = Suffix._apply_major_harmony(word, base1, suffix_obj.major_harmony)
    base1 = Suffix._apply_minor_harmony(word, base1, suffix_obj.minor_harmony)

    base2 = Suffix._apply_major_harmony(word, base2, suffix_obj.major_harmony)
    base2 = Suffix._apply_minor_harmony(word, base2, suffix_obj.minor_harmony)

    if word[-1] in wrd.VOWELS:
        base1 =  base1[1:]

    return [base1, base2]


def form_for_privative_siz(word, suffix_obj):
    """Form function for privative_siz suffix"""
    pass


def form_for_philicative_cil(word, suffix_obj):
    """Form function for philicative_cil suffix"""
    pass


def form_for_composessive_li(word, suffix_obj):
    """Form function for composessive_li suffix"""
    pass


def form_for_suitative_lik(word, suffix_obj):
    """Form function for suitative_lik suffix"""
    pass


def form_for_adjectiative_ay(word, suffix_obj):
    """Form function for adjectiative_ay suffix (n2n)"""
    pass


def form_for_marking_ki(word, suffix_obj):
    """Form function for marking_ki suffix"""
    pass


def form_for_temporative_leyin(word, suffix_obj):
    """Form function for temporative_leyin suffix"""
    pass


def form_for_ideologicative_izm(word, suffix_obj):
    """Form function for ideologicative_izm suffix"""
    pass


def form_for_locative_le(word, suffix_obj):
    """Form function for locative_le suffix"""
    pass


def form_for_eventative_tay(word, suffix_obj):
    """Form function for eventative_tay suffix"""
    pass


def form_for_scientist_olog(word, suffix_obj):
    """Form function for scientist_olog suffix"""
    pass


def form_for_confactuous_le(word, suffix_obj):
    
    from util.suffixes import Suffix

    base = "le"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    
    if word[-1] in wrd.VOWELS:
        base = "y" + base

    return [base]

    pass



def form_for_sg1_conjugation(word, suffix_obj):
    pass

def form_for_sg2_conjugation(word, suffix_obj):
    pass

def form_for_sg3_conjugation(word, suffix_obj):
    pass

def form_for_pl1_conjugation(word, suffix_obj):
    pass

def form_for_pl2_conjugation(word, suffix_obj):
    pass

def form_for_pl3_conjugation(word, suffix_obj):
    pass