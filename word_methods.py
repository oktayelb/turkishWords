from enum import Enum
from suffixes import name2name

vowels = ['a','o','u','ı','e','ö','ü','i',]     ##T.Turkcesindeki tum Unluler

back_vowels = ['a','ı','o','u']                 ## Kalin Unluler

front_vowels =['e','i','ö','ü']                 ## Ince Unluler

fistikci_sahap = ['f','s','t','ç','ş','h','p']  ## Ikonik unsuz sertlestirmesine sokan unsuz harfler


class MajorHarmony(Enum):   ## Buyuk Unlu Uyuymu
    Back  = 0  ##Kalin
    Front = 1  ##ince 

class MinorHarmony(Enum):   ## Kucuk Unlu uyumu

    BackRound   = 0  ##Kalin Yuvarlak 
    BackWide    = 1  ##Kalin Duz
    FrontRound  = 2  ##Ince Yuvarlak 
    FrontWide   = 3  ##Ince Duz




def exists(word):           ## Sozcuk words.txt te var mi onu kontrol ediyor.
    
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



def major_harmony(word):    ## Sozcuge gelecek ekler icin son unlusune bakiyor. Istisnalari saymiyor.
    i = len(word) -1
    while(word[i] not in vowels):
        i = i-1
    ## kuraldışıları da çözüver
    if (word[i] in back_vowels):
        return MajorHarmony.Back
    if (word[i] in front_vowels):
        return MajorHarmony.Front
    

def minor_harmony(word):
    i = len(word) -1
    while(word[i] not in vowels):
        i = i-1
    ## kuraldışıları da çözüver
    temp = word[i]
    if (temp in ['o','u']):
        return MinorHarmony.BackRound
    if (temp in ['ı','a']):
        return MinorHarmony.BackWide
    if (temp in ['ö','ü'] ):
        return MinorHarmony.FrontRound
    if (temp in ['i','e']):
        return MinorHarmony.FrontWide


def infinitive(word):       ## Sozcugun Mastarini veriyor  
    return word + ("mak" if major_harmony(word) == MajorHarmony.Back else "mek")
 

def isVerb(word):           ## girdinin sonuna mastar ekleyip var mi diye bakiyor. 
    return exists(infinitive(word))

    