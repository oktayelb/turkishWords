from enum import Enum
from pathlib import Path

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
    WORDS = list(line.strip() for line in f if line.strip())

def exists(word: str) -> bool:
    if not word:
        return 0

    # Check basic forms
    if word in WORDS:
        return 1
    
    if infinitive(word) in WORDS:
        return 2

    # Check soft-l variant if word ends with 'l' because of the convention I imposed
    if word.endswith("l"):
        soft_l = word[:-1] + "ł"
        if soft_l in WORDS:
            return 1

        if infinitive(soft_l) in WORDS:
            return 2 

    return 0

def can_be_verb(word: str) -> bool:
    """Checks if a root is a verb by verifying its infinitive form."""
    return exists(infinitive(word))


# --- Harmony functions ---
def major_harmony(word: str) -> MajorHarmony | None:
    """Determines major vowel harmony based on last vowel"""
    if word.endswith("l"):              
        soft_l = word[:-1] + "ł"
        if exists(soft_l):
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

