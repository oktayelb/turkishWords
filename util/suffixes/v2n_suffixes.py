from util.suffix import Suffix, Type, HasMajorHarmony, HasMinorHarmony, SuffixGroup
import util.word_methods as wrd

VOWELS = ["a","e","ı","i","o","ö","u","ü"]

# ============================================================================
# FORM FUNCTIONS
# ============================================================================

def form_for_continuous_iyor(word, suffix_obj):
    base = "i"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)

    base = base+"yor"
    return [base]

def form_for_nounifier_ecek(word, suffix_obj):
    """
    Form function for nounifier_ecek suffix (Sıfat-Fiil)
    - Returns harmonized versions of ecek, yecek, cek
    - AND softened versions: eceğ, yeceğ, ceğ
    """
    result_list = []
    
    # Base form: ecek
    base = "ecek"   
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    # Softened base: eceğ / acağ
    soft_base = Suffix._apply_softening(base)
    if soft_base != base:
        result_list.append(soft_base)
    
    # Add buffer variants for vowel-ending words
    if word and word[-1] in VOWELS:
        # yecek variant (yıkayacak)
        y_form = 'y' + base
        result_list.append(y_form)
        
        # Softened y_form: yeceğ
        soft_y = Suffix._apply_softening(y_form)
        if soft_y != y_form:
            result_list.append(soft_y)
        
        # cek variant (vowel dropped - uncommon in standard V2N but listed)
        short_form = base[1:] 
        result_list.append(short_form)
        
        # Softened short_form: ceğ
        soft_short = Suffix._apply_softening(short_form)
        if soft_short != short_form:
            result_list.append(soft_short)
    
    return result_list


def form_for_factative_ir(word, suffix_obj):
    """
    Form function for factative_ir suffix (Geniş Zaman Sıfat-Fiil)
    - Default forms: er, ir, z (negative)
    - Note: Does not soften (r and z are soft/continuant).
    """
    result_list = []
    
    # Geniş zamanın olumsuzu (maz/mez) kökü için 'z'
    if len(word) > 2 and word[-2:] in ["ma","me"]:
        if wrd.can_be_verb(word[:-2]):
            z_base = "z"
            result_list.append(z_base)

    # ir form with harmony
    ir_base = 'ir'
    ir_base = Suffix._apply_major_harmony(word, ir_base, suffix_obj.major_harmony)
    ir_base = Suffix._apply_minor_harmony(word, ir_base, suffix_obj.minor_harmony)
    ir_base = Suffix._apply_consonant_hardening(word, ir_base)
    result_list.append(ir_base)
    
    # er form with harmony (Gider, Yapar)
    er_base = 'er'
    er_base = Suffix._apply_major_harmony(word, er_base, suffix_obj.major_harmony)
    er_base = Suffix._apply_consonant_hardening(word, er_base)
    if er_base not in result_list:
        result_list.append(er_base)
    
    # Vowel drop variant for vowel-ending words (Oku-r)
    if word and word[-1] in VOWELS:
        r_form = 'r'
        if r_form not in result_list:
            result_list.append(r_form)
    
    return result_list


def form_for_toolative_ek(word, suffix_obj):
    """
    Form for toolative_ek (e.g. Dur-ak -> Durağı)
    Includes softening (k -> ğ).
    """
    result_list = []
    
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    # Softening: ak -> ağ
    soft_base = Suffix._apply_softening(base)
    if soft_base != base:
        result_list.append(soft_base)
        
    return result_list


def form_for_nounifier_iş(word, suffix_obj):
    """
    Geliş, Gidiş.
    Note: 'iş' suffix usually does not soften in standard morphology.
    (Gelişi, not Gelici). Keeping logic as is.
    """
    result_list = []
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        if word[-1] in VOWELS:
            result_list.append('y' + base) # işleyiş
            result_list.append('ğ' + base) 
            result_list.append(base[1:])   # ş variant (gülüş)
        
        if word[-1] == 'n':
            c_form = 'ç' + base # sevinç ? (Aslında inç ayrı ek ama burada variant olarak eklenmiş)
            result_list.append(c_form)
    
    return result_list


def form_for_perfectative_ik(word, suffix_obj):
    """
    Form for perfectative_ik (e.g. Aç-ık -> Açığı)
    Includes softening (k -> ğ).
    """
    result_list = []
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    

    # Softening base: ik -> iğ
    soft_base = Suffix._apply_softening(base)
    if soft_base != base:
        result_list.append(soft_base)
    
    if word[-2:] in ["me","ma"]:
        result_list.append("y" + base)
        result_list.append("y" + soft_base)

    if word:
        if word[-1] in VOWELS:
            # y buffer
            y_form = 'y' + base
            result_list.append(y_form)
            
            soft_y = Suffix._apply_softening(y_form)
            if soft_y != y_form:
                result_list.append(soft_y)
            
            result_list.append('ğ' + base)
            
            # k variant
            k_form = base[1:]
            result_list.append(k_form)
            
            # Softening k variant: k -> ğ
            soft_k = Suffix._apply_softening(k_form)
            if soft_k != k_form:
                result_list.append(soft_k)
        
        if word[-1] in ['n', 'r']:
            k_form = 'k'
            k_form = Suffix._apply_consonant_hardening(word, k_form)
            if k_form not in result_list:
                result_list.append(k_form)
                
                # Softening k form
                soft_k_form = Suffix._apply_softening(k_form)
                if soft_k_form != k_form:
                    result_list.append(soft_k_form)


    
    return result_list


def form_for_nounifier_i(word, suffix_obj):
    result_list = []
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word and word[-1] in VOWELS:
        result_list.append('y' + base)
        result_list.append('ğ' + base)
    
    
    
    return result_list


def form_for_nounifier_inti(word, suffix_obj):
    result_list = []
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    result_list.append(base)
    
    if word:
        if word[-1] in VOWELS:
            reduced = base[1:]
            result_list.append(reduced)
        
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
    Form for toolifier_geç (Süz-geç -> Süz-gec-i)
    Includes softening (ç -> c).
    """
    result_list = []
    
    # Base: geç
    gec_base = suffix_obj.suffix
    gec_base = Suffix._apply_major_harmony(word, gec_base, suffix_obj.major_harmony)
    gec_base = Suffix._apply_minor_harmony(word, gec_base, suffix_obj.minor_harmony)
    gec_base = Suffix._apply_consonant_hardening(word, gec_base)
    result_list.append(gec_base)
    
    # Softening: geç -> gec / keç -> kec
    soft_gec = Suffix._apply_softening(gec_base)
    if soft_gec != gec_base:
        result_list.append(soft_gec)
    
    # Variant: eç (Utan-gaç -> Utan-ac -> Utan-ac-ı ?) 
    # Not: eç formu genelde 'g' düşmesiyle oluşur, yine de ekliyoruz.
    ec_base = 'eç'
    ec_base = Suffix._apply_major_harmony(word, ec_base, suffix_obj.major_harmony)
    ec_base = Suffix._apply_minor_harmony(word, ec_base, suffix_obj.minor_harmony)
    ec_base = Suffix._apply_consonant_hardening(word, ec_base)
    
    if ec_base not in result_list:
        result_list.append(ec_base)
        
        # Softening variant: eç -> ec
        soft_ec = Suffix._apply_softening(ec_base)
        if soft_ec != ec_base:
            result_list.append(soft_ec)
    
    return result_list

def form_for_adverbial_erek (word, suffix_obj):
    # Erek yumuşamaz (Giderek -> Gidereği X)
    base = "erek"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    if word[-1] in VOWELS:
        base = "y" + base
    return [base]

# ============================================================================
# VERB TO NOUN SUFFIXES (v2n)
# ============================================================================
    
# --- STANDART YAPIM EKLERİ (DERIVATIONAL - Group 10) ---
# Bunlar isim/sıfat kökü oluşturur, üzerine çoğul/iyelik gelebilir.


##TODO form for form for constofactattive koy, agan eğen biçimleri için
infinitive_me = Suffix("infinitive_me", "me", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
infinitive_mek = Suffix("infinitive_mek", "mek", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
factative_en = Suffix("factative_en", "en", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True, group=SuffixGroup.DERIVATIONAL)
toolative_ek = Suffix("toolative_ek", "ek", Type.VERB, Type.NOUN, form_function= form_for_toolative_ek, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
constofactative_gen = Suffix("constofactative_gen", "gen", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
constofactative_gin = Suffix("constofactative_gin", "gin", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
nounifier_iş = Suffix("nounifier_iş", "iş", Type.VERB, Type.NOUN, form_function= form_for_nounifier_iş, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True, group=SuffixGroup.DERIVATIONAL)
perfectative_ik = Suffix("perfectative_ik", "ik", Type.VERB, Type.NOUN, form_function= form_for_perfectative_ik, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True, group=SuffixGroup.DERIVATIONAL)
nounifier_i = Suffix("nounifier_i", "i", Type.VERB, Type.NOUN, form_function= form_for_nounifier_i, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, needs_y_buffer=True, group=SuffixGroup.DERIVATIONAL)
nounifier_gi = Suffix("nounifier_gi", "gi", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
nounifier_ge = Suffix("nounifier_ge", "ge", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
nounifier_im = Suffix("nounifier_im", "im", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
nounifier_in = Suffix("nounifier_in", "in", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
nounifier_it = Suffix("nounifier_it", "it", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
nounifier_inç = Suffix("nounifier_inç", "inç", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
nounifier_inti = Suffix("nounifier_inti", "inti", Type.VERB, Type.NOUN, form_function= form_for_nounifier_inti, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
toolifier_geç = Suffix("toolifier_geç", "geç", Type.VERB, Type.NOUN, form_function= form_for_toolifier_geç, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
nounifier_anak = Suffix("nounifier_anak", "anak", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
nounifier_amak = Suffix("nounifier_amak", "amak", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
nounifier_ce = Suffix("nounifier_ce", "ce", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
subjectifier_men = Suffix("subjectifier_men", "men", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
subjectifier_giç = Suffix("subjectifier_giç", "giç", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
# Sıfat-Fiil (Participles) - Bunlar da isimleşir, çekim alabilir (Bildiğ-im, Yapacağ-ım)
adjectifier_dik = Suffix("adjectifier_dik", "dik", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
nounifier_ecek = Suffix("nounifier_ecek", "ecek", Type.VERB, Type.NOUN, form_function= form_for_nounifier_ecek, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True, group=SuffixGroup.DERIVATIONAL)
neverfactative_mez = Suffix("neverfactative_mez", "mez", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
factative_ir = Suffix("factative_ir", "ir", Type.VERB, Type.NOUN, form_function= form_for_factative_ir, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
pastfactative_miş = Suffix("pastfactative_miş", "miş", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)
undoing_meden = Suffix("undoing_meden", "meden", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)

wish_suffix = Suffix("wish_suffix", "se", Type.NOUN, Type.NOUN,major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.PREDICATIVE, is_unique=True)

## bu ekin şu anki hali yanlış, aslında bu ek fiilken i eki alıp isim olmuş sözcüklere eklenir. 
# bak-  bakı     bakı yorum.  ölü yorum , ölü oluyorum tarzında 
# ama bekleyorum vs gibi örneklerdeki düzensizliklerden ötürü şimdilik böyle devam edecek 
continuous_iyor = Suffix(
    "continuous_iyor", 
    "iyor", 
    Type.VERB,  # Fiile gelir
    Type.NOUN,  # İsimleştirir (üzerine şahıs eki alır: geliyor-um)
    form_function=form_for_continuous_iyor, 
    major_harmony=HasMajorHarmony.Yes, # Fonksiyon hallediyor
    minor_harmony=HasMinorHarmony.Yes, # Fonksiyon hallediyor
    group=SuffixGroup.DERIVATIONAL # Group 10 (En başa yakın)
)

# --- ZARF-FİİLLER (GERUNDS - Group 90) ---
# Bunlar Zarf yapar. Üzerine İyelik (30) veya Hal eki (40) ALAMAZLAR.
# SuffixGroup.GERUND = 90 olduğu için; 30 < 90 kontrolü devreye girer ve zincir kesilir.

adverbial_erek = Suffix("adverbial_erek", "erek", Type.VERB, Type.NOUN, form_function= form_for_adverbial_erek, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, group=SuffixGroup.DERIVATIONAL)
adverbial_ip   = Suffix("adverbial_ip", "ip", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL_LOCKING)
adverbial_e    = Suffix("adverbial_e", "e", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True, group=SuffixGroup.DERIVATIONAL_LOCKING)
adverbial_esi  = Suffix("adverbial_esi", "esi", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, needs_y_buffer=True, group=SuffixGroup.DERIVATIONAL) # Ölesiye (kalıplaşmış olsa da genelde zarftır)
adverbial_ince = Suffix("adverbial_ince", "ince", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.Yes, group=SuffixGroup.DERIVATIONAL)

VERB2NOUN = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]