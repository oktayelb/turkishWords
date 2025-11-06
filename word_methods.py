from enum import Enum

# --- Constants ---
VOWELS = ['a','o','u','ı','e','ö','ü','i']
BACK_VOWELS = ['a','ı','o','u']
FRONT_VOWELS = ['e','i','ö','ü']
HARD_CONSONANTS = ['f','s','t','k','ç','ş','h','p']  # fıstıkçı şahap


vowels  = ['a','o','u','ı','e','ö','ü','i']
fistikci_sahap = ['f','s','t','k','ç','ş','h','p']

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
with open("words.txt", "r", encoding="utf-8") as f:
    WORDS = set(line.strip() for line in f if line.strip())

def exists(word: str) -> bool:
    """Checks if a word exists in words.txt"""
    return word in WORDS or infinitive(word) in WORDS


# --- Harmony functions ---
def major_harmony(word: str) -> MajorHarmony | None:
    """Determines major vowel harmony based on last vowel"""
    for ch in reversed(word):
        if ch in VOWELS:
            return MajorHarmony.BACK if ch in BACK_VOWELS else MajorHarmony.FRONT
    return None  # no vowels


def minor_harmony(word: str) -> MinorHarmony | None:
    """Determines minor vowel harmony based on last vowel"""
    for ch in reversed(word):
        if ch not in VOWELS:
            continue
        if ch in ['o', 'u']:  return MinorHarmony.BACK_ROUND
        if ch in ['a', 'ı']:  return MinorHarmony.BACK_WIDE
        if ch in ['ö', 'ü']:  return MinorHarmony.FRONT_ROUND
        if ch in ['e', 'i']:  return MinorHarmony.FRONT_WIDE
    return None


# --- Morphological utilities ---
def infinitive(word: str) -> str:
    """Returns the infinitive form of a verb root."""
    suffix = "mak" if major_harmony(word) == MajorHarmony.BACK else "mek"
    return word + suffix


def can_be_verb(word: str) -> bool:
    """Checks if a root is a verb by verifying its infinitive form."""
    return exists(infinitive(word))


def harden_consonant(ch: str) -> str:
    """Returns the hardened version of a soft consonant if applicable."""
    mapping = {'b': 'p', 'c': 'ç', 'd': 't', 'g': 'k', 'ğ': 'k'}
    return mapping.get(ch, ch)


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

