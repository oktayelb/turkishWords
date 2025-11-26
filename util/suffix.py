from enum import Enum
import util.word_methods as wrd



class Type(Enum):
    NOUN = "noun"
    VERB = "verb"
    BOTH = "both"


class HasMajorHarmony(Enum):
    Yes = 0
    No = 1


class HasMinorHarmony(Enum):
    Yes = 0
    No = 1


class Suffix:
    def __init__(self, name, suffix, comes_to, makes, form_function=None, major_harmony=None, minor_harmony=None, needs_y_buffer=False):
        self.name = name
        self.suffix = str(suffix)
        self.comes_to = comes_to
        self.makes = makes
        self.major_harmony = major_harmony
        self.minor_harmony = minor_harmony
        self.needs_y_buffer = needs_y_buffer
        self.form_function = form_function if form_function else self._default_form
    
    def form(self, word):
        return self.form_function(word, self)
    
    @staticmethod
    def _default_form(word, suffix_obj):
        result = suffix_obj.suffix
        
        result = Suffix._apply_major_harmony(word, result, suffix_obj.major_harmony)
        result = Suffix._apply_minor_harmony(word, result, suffix_obj.minor_harmony)
        result = Suffix._apply_consonant_hardening(word, result)
        
        result_list = [result]
        
        if Suffix._should_add_buffer_variants(word, result):
            if suffix_obj.needs_y_buffer:
                result_list.append('y' + result)
                result_list.append('ğ' + result)
            if len(result) > 1:
                result_list.append(result[1:])
        
        return result_list
    
    @staticmethod
    def _apply_major_harmony(word, result, major_harmony):
        if major_harmony != HasMajorHarmony.Yes:
            return result
        
        if wrd.major_harmony(word) == wrd.MajorHarmony.BACK:
            result = result.replace("e", "a")
            result = result.replace("i", "ı")
            result = result.replace("ü", "u")
            result = result.replace("ö", "o")
        
        return result
    
    @staticmethod
    def _apply_minor_harmony(word, result, minor_harmony):
        if minor_harmony != HasMinorHarmony.Yes:
            return result
        
        word_harmony = wrd.minor_harmony(word)
        
        if word_harmony == wrd.MinorHarmony.BACK_ROUND:
            result = result.replace("ı", "u")
        elif word_harmony == wrd.MinorHarmony.FRONT_ROUND:
            result = result.replace("i", "ü")
        
        return result
    
    @staticmethod
    def _apply_consonant_hardening(word, result):
        if not word or not result:
            return result
        
        if word[-1] not in wrd.HARD_CONSONANTS:
            return result
        
        first_char = result[0]
        if first_char not in ['g', 'c', 'd', 'ğ']:
            return result
        
        hardening_map = {'g': 'k', 'd': 't', 'c': 'ç', 'ğ': 'k'}
        return hardening_map.get(first_char, first_char) + result[1:]
    
    @staticmethod
    def _should_add_buffer_variants(word, result):
        return (word and result and 
                word[-1] in wrd.VOWELS and 
                result[0] in wrd.VOWELS)



