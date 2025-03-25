import word_methods as wrd

input_word = input("Sozcuk giriniz: \n")









i = 2
while (i <= len(input_word)):
    partition = input_word[0:i]
    verb_partition = wrd.infinitive(partition) ## ünlü uyumuna göre mastarlamak

    if(wrd.exists(partition)):
        print("Found one possible root", partition)

    if(wrd.exists(verb_partition)):
        print("Found one possible root", partition + "-")
    i = i + 1
    