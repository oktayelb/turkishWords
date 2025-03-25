from enum import Enum
from suffixes import name2name

class Harmony(Enum):
    Back  = 0
    Front = 1




def exists(word):
    
    with  open("words.txt", "r", encoding="utf-8") as f:
        word_list = [line.strip() for line in f]

    word_list.sort()

    low = 0
    high = len(word_list) - 1

    while low <= high:
        mid = (low + high) // 2
        if word_list[mid] == word:
            return True
        elif word_list[mid] < word:
            low = mid + 1
        else:
            high = mid - 1
    return False



def harmony(word):
    i = len(word) -1
    while(word[i] not in ['a','o','u','ı','e','ö','ü','i',]):
        i = i-1
    ## kuraldışıları da çözüver
    if (word[i] in ['a','ı','o','u']):
        return Harmony.Back
    if (word[i] in ['e','i','ö','ü']):
        return Harmony.Front
    

def infinitive(word):
    return word + ("mak" if harmony(word) == Harmony.Back else "mek")
 

    