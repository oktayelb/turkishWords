import word_methods as wrd
from classes import Noun as wrd_type


def decompose(word):
    i = 2
    possible_roots = []
    while (i <= len(word)):
        partition = word[0:i]
        rest = word[i: len(word)]
        i = i + 1
        if(wrd.exists(partition)):
            if(wrd.exists(rest)):
                print("Found Composite word" ,partition,rest )
                
            print("Found one possible root", partition)
            print("Possible partition" ,partition,rest)
            possible_roots.append(partition)
        if(wrd.exists(wrd.infinitive(partition))):
            print("Found one possible verb root", partition + "-")
            print("Possible partition" ,partition+'-',rest)
            possible_roots.append(partition + '-')

    return []



input= input("Sozcuk giriniz: \n")


print(wrd.exists(input))
decompose(input)



    
    