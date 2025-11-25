import util.suffix as sfx
from functools import lru_cache
from typing import List, Tuple, Optional

class MorphEngine:
    """
    Handles all linguistic logic, suffix operations, and morphological decomposition.
    Acting as a Logic Layer, it has no knowledge of files or ML models.
    """

    def __init__(self):
        self.suffixes = sfx.ALL_SUFFIXES


    @lru_cache(maxsize=10000)
    def decompose(self, word: str) -> List[Tuple]:


        return sfx.decompose(word)

    def get_infinitive(self, word: str) -> str:
        """Generates the infinitive form of a word (word + mek/mak)"""
        suffix_form = self.infinitive_suffix.form(word)
        if suffix_form:
            return word + suffix_form[0]
        return word

    def get_soft_l_variation(self, word: str) -> Optional[str]:
        """Handles the 'l' -> 'ł' check used in specific dictionary cases"""
        if word.endswith("l"):
            return word[:-1] + "ł"
        return None

    def get_suffix_objects(self):
        return self.suffixes