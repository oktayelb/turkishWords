from enum import Enum
from pathlib import Path
from typing import List, Tuple
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "words.txt"


## Vowel Classes
BACK_FLAT   = ['a','ı']
BACK_ROUND  = ['o','u']

FRONT_FLAT  = ['e','i']
FRONT_ROUND = ['ö','ü']

BACK_VOWELS  = BACK_FLAT  + BACK_ROUND
FRONT_VOWELS = FRONT_FLAT + FRONT_ROUND 

VOWELS = BACK_VOWELS + FRONT_VOWELS


HARD_CONSONANTS = ['f','s','t','k','ç','ş','h','p']  # fıstıkçı şahap



# --- Enums ---
class MajorHarmony(Enum):
    BACK = "back"
    FRONT = "front"

class MinorHarmony(Enum):
    BACK_ROUND = 0
    BACK_WIDE  = 1
    FRONT_ROUND = 2
    FRONT_WIDE  = 3


# --- Load words once ---
with open(DATA_FILE, "r", encoding="utf-8") as f:
    WORDS = set(line.strip() for line in f if line.strip())

def can_be_noun(word: str) -> bool:
    if not word:
        return 0

    # Check basic forms
    if word in WORDS:
        return 1

    # Check soft-l variant if word ends with 'l' because of the convention I imposed
    if word.endswith("l"):
        soft_l = word[:-1] + "ł"
        if soft_l in WORDS:
            return 1

    return 0

def can_be_verb(word: str) -> bool:
    """Checks if a root is a verb by verifying its infinitive form."""
    return can_be_noun(infinitive(word))


# --- Harmony functions ---
def major_harmony(word: str) -> MajorHarmony | None:
    """Determines major vowel harmony based on last vowel"""
    if word.endswith("l"):              
        soft_l = word[:-1] + "ł"
        if can_be_noun(soft_l):
            return MajorHarmony.FRONT
    for ch in reversed(word):
        if ch in VOWELS:
            return MajorHarmony.BACK if ch in BACK_VOWELS else MajorHarmony.FRONT
    return None  # no vowels

def minor_harmony(word: str) -> MinorHarmony | None:
    """Determines minor vowel harmony based on last vowel"""
    for ch in reversed(word):
        if ch not in VOWELS:
            continue
        if ch in ['o', 'u',"ö","ü"]:  
            return MinorHarmony.BACK_ROUND if  major_harmony(word) == MajorHarmony.BACK else MinorHarmony.FRONT_ROUND
        if ch in ['a', 'ı','e', 'i']:
             return MinorHarmony.BACK_WIDE if  major_harmony(word) == MajorHarmony.BACK else MinorHarmony.FRONT_WIDE
    return None


# --- Morphological utilities ---
def infinitive(word: str) -> str:
    """Returns the infinitive form of a verb root."""
    suffix = "mak" if major_harmony(word) == MajorHarmony.BACK else "mek"
    return word + suffix

def ends_with_vowel(word: str) -> bool:
    """Check if word ends with a vowel"""
    return word and word[-1] in VOWELS

def ends_with_consonant(word: str) -> bool:
    """Check if word ends with a consonant"""
    return word and word[-1] not in VOWELS

def has_no_vowels(word: str) -> bool:
    """Return True if the given word contains no vowels."""
    ret = True

    for ch in word:
        if ch in VOWELS:
            ret = False
            break
    return ret

def get_root_candidates(surface_root: str) -> List[Tuple[str, str]]:
    """Analyzes the text segment and returns (Surface Form, Dictionary Lemma)."""
    candidates = [] 

    def check_and_add_softened(form_to_check):
        if not form_to_check: return

        last_char = form_to_check[-1]
        candidate = form_to_check

        if last_char   == 'b':  candidate = form_to_check[:-1] + 'p'
        elif last_char == 'c':  candidate = form_to_check[:-1] + 'ç'
        elif last_char == 'd':  candidate = form_to_check[:-1] + 't'
        elif last_char == 'ğ':  candidate = form_to_check[:-1] + 'k'
        elif last_char == 'g':  candidate = form_to_check[:-1] + 'k'

        # 2. Check & Append
        if can_be_noun(candidate) or can_be_verb(candidate):
            candidates.append(candidate)



    check_and_add_softened(surface_root)


    if 2 < len(surface_root) < 10 and ends_with_consonant(surface_root):
        prefix = surface_root[:-1]
        suffix_char = surface_root[-1]
        
        for vowel in ['ı', 'i', 'u', 'ü']:
            restored = prefix + vowel + suffix_char 
            
            # Case A: Restored form is the root (e.g. burn -> burun)
            if can_be_noun(restored) or can_be_verb(restored):
                candidates.append( restored)
            
            check_and_add_softened(restored)

    # 4. Vowel Narrowing at Root (e.g. diye -> de, yiye -> ye)
    if not can_be_noun(surface_root) and len(surface_root) > 1:
        for terminal_vowel in ['a', 'e']:
            restored = surface_root + terminal_vowel
            if can_be_noun(restored) or can_be_verb(restored):
                 candidates.append( restored)

    return candidates