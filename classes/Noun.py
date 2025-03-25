import word_methods as wrd
from suffixes import  *
class Root():
    pass
    ##isVerb
    ##suffixes
    def __init__(self,word ):
        self.root = word
        self.harmony = wrd.harmony(word)
        self.isVerb =  (1 if word[len(word)-1] == '-' else 0)


class Word():
    pass
    ##suffixes
    ##length
    ##isVerb
    ##harmony
    ##meaning
    ##isRoot
    ##root

## Noun("adam").cık() = Noun("adamcık")
# Noun("adamcık").laş() = Verb("adamcıklaş")
# 
# 
# 
# 
# 
# 
#  

class Noun(Word):

    def __init__(self, word, suffix_list):
        self.word = word
        self.harmony = wrd.harmony(word)
        self.suffixes = suffix_list

    def print(self):
        pass
    def lar(self):
        form  =("lar" if self.harmony == wrd.Harmony.Back else "ler")
        return     Noun(self.word +form  , self.suffixes + [form])
    def daş(self):
        form =("daş" if self.harmony == wrd.Harmony.Back else "deş")
        return     Noun(self.word + form, self.suffixes + [form])
    def ca(self):
        form = ("ça" if self.harmony == wrd.Harmony.Back else "çe") if self.word[len(self.word)-1] in wrd.fistikci_sahap  else("ca" if self.harmony == wrd.Harmony.Back else "ce")
        return     Noun(        
                                self.word + form,
                                self.suffixes + [form]
                        )