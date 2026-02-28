from enum import Enum
from pathlib import Path
import random
from typing import List, Tuple, Optional

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

# --- Centralized Dictionary State ---
WORDS_LIST: List[str] = []
WORDS_SET: set = set()

def _load_dictionary():
    global WORDS_LIST, WORDS_SET
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            WORDS_LIST = [line.strip() for line in f if line.strip()]
            WORDS_SET = set(WORDS_LIST)
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found")
        WORDS_LIST = []
        WORDS_SET = set()

# Initialize on module load
_load_dictionary()

def delete_word(word: str) -> bool:
    """Removes a word from the in-memory dictionary state."""
    if word in WORDS_SET:
        WORDS_SET.remove(word)
        WORDS_LIST.remove(word)
        return True
    return False

def get_all_words() -> List[str]:
    """Returns the current list of dictionary words."""
    return WORDS_LIST

def get_random_word() -> Optional[str]:
    """Returns a random word from the dictionary."""
    return random.choice(WORDS_LIST) if WORDS_LIST else None

def exists(word: str) -> bool:
    return can_be_noun(word) or can_be_verb(word)

def can_be_noun(word: str) -> bool:
    if not word:
        return False

    if word in WORDS_SET:
        return True

    if word.endswith("l"):
        soft_l = word[:-1] + "ł"
        if soft_l in WORDS_SET:
            return True

    return False

def can_be_verb(word: str) -> bool:
    """Checks if a root is a verb by verifying its infinitive form."""
    if word in ("e", "ha", "da", "ra", "ço"):
        return False
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
            return MinorHarmony.BACK_ROUND if major_harmony(word) == MajorHarmony.BACK else MinorHarmony.FRONT_ROUND
        if ch in ['a', 'ı','e', 'i']:
             return MinorHarmony.BACK_WIDE if major_harmony(word) == MajorHarmony.BACK else MinorHarmony.FRONT_WIDE
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
    for ch in word:
        if ch in VOWELS:
            return False
    return True

def get_root_candidates(surface_root: str) -> List[str]:
    """Analyzes the text segment and returns Surface Forms that are Dictionary Lemmas."""
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

        if can_be_noun(candidate) or can_be_verb(candidate):
            candidates.append(candidate)

    check_and_add_softened(surface_root)

    if 2 < len(surface_root) < 10 and ends_with_consonant(surface_root):
        prefix = surface_root[:-1]
        suffix_char = surface_root[-1]
        
        for vowel in ['ı', 'i', 'u', 'ü']:
            restored = prefix + vowel + suffix_char 
            if can_be_noun(restored) or can_be_verb(restored):
                candidates.append(restored)
            check_and_add_softened(restored)

    if not can_be_noun(surface_root) and len(surface_root) > 1:
        for terminal_vowel in ['a', 'e']:
            restored = surface_root + terminal_vowel
            if can_be_noun(restored) or can_be_verb(restored):
                 candidates.append(restored)

    return candidates