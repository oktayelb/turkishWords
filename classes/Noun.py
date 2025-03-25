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
    pass