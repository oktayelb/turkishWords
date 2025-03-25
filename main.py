import word_methods as wrd
from classes import Noun as wrd_type


def decompose(word):
    i = 2
    possible_roots = []
    while (i <= len(word)):
        partition = word[0:i]
        i = i + 1
        if(wrd.exists(partition)):
            print("Found one possible root", partition)
            print("Possible partition" ,partition, word[len(partition): len(word)])
            possible_roots.append(partition)
        if(wrd.exists(wrd.infinitive(partition))):
            print("Found one possible root", partition + "-")
            print("Possible partition" ,partition+'-', word[len(partition): len(word)])
            possible_roots.append(partition + '-')



input= input("Sozcuk giriniz: \n")


print(input)
decompose(input)



    
    