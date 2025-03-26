from enum import Enum
import word_methods as wrd
class WType(Enum):
    Noun = 0
    Verb = 1

class HasMajorHarmony(Enum):      
    Yes = 0                      
    No = 1                       

class HasMinorHarmony(Enum):
    Yes = 0
    No = 1

class Suffix(): ##
    def __init__(self,suffix,comes_to, makes,major_harmony,minor_harmony):
        self.suffix = str(suffix)
        self.comes_to = comes_to
        self.makes = makes
        self.major_harmony = major_harmony
        self.minor_harmony = minor_harmony

    def form(self,word): 
        result = self.suffix
        if(self.major_harmony == HasMajorHarmony.Yes):
            word_harmony =wrd.major_harmony(word)
            if( word_harmony== wrd.MajorHarmony.Back): 
                i = 0
                while(self.suffix[i] not in wrd.front_vowels):
                    i = i+1
                
                the_vowel = self.suffix[i]

                if( the_vowel is "e" ):
                    result = self.suffix.replace("e","a",4)
                if( the_vowel is "i" ):
                    result = self.suffix.replace("i","ı",4)
                if( the_vowel is "ü" ):
                    result = self.suffix.replace("ü","u",4)
                if( the_vowel is "ü" ):        ## Tabii ki O/Ö ile ek yok (bizziko) ancak gelebilir.
                    result = self.suffix.replace("ü","u",4)
                
            if(word_harmony == wrd.MajorHarmony.Front): pass
                ## if vowels in suffix are already in wrd.front_vowels do nothing
                ## since the standart I have put dictates that the default form
                ## of the suffix will be the front vowel form.

        if(self.minor_harmony == HasMinorHarmony.Yes): ## Buradaki bilgilerin dogrulugu supheli
            word_harmony = wrd.minor_harmony(word)

            if( word_harmony  == wrd.MinorHarmony.BackRound  ): pass
                ## if vowels in suffix are already in ['u'] do nothing
                ## else swap the vowel with 'u'
            if( word_harmony  == wrd.MinorHarmony.BackWide   ): pass
                ## if vowels in suffix are already in ['u'] do nothing
                ## else swap the vowel with 'ı'
            if( word_harmony  == wrd.MinorHarmony.FrontRound ): pass
                ## if vowels in suffix are already in ['ü'] do nothing
                ## else swap the vowel with 'ü'
            if( word_harmony  == wrd.MinorHarmony.FrontWide  ): pass
                ## if vowels in suffix are already in ['e','i'] do nothing
                ## else swap the vowel with 'i'

        if(word[len(word)-1] in wrd.vowels and self.suffix[0] in wrd.vowels):
            
            result = result[0:len(result)]
            ## belk' bunu basha alip optimize edebilirsin
            ## trim the leading vowel from the result 



# V2V suffixes    
reflexive_is          =  Suffix('iş'     , WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
reflexive_ik          =  Suffix("ik"     , WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_t              =  Suffix("it"     , WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_tir            =  Suffix("tir"    , WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes) 
active_ir             =  Suffix("ir"     , WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_il            =  Suffix("il"     , WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_in            =  Suffix("in"     , WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)

#N2N suffixes
plural_ler            =  Suffix("ler"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
counting_er           =  Suffix("er"     , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
cooperative_deş       =  Suffix("daş"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_ce           =  Suffix("ça"     , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_sel          =  Suffix("sel"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
dimunitive_ek         =  Suffix("ak"     , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
approximative_imtrakF =  Suffix("imtırak", WType.Noun, WType.Noun, HasMajorHarmony.No , HasMinorHarmony.No)
approximative_imtrakB =  Suffix("ımtırak", WType.Noun, WType.Noun, HasMajorHarmony.No , HasMinorHarmony.No)
dimunitive_cik        =  Suffix("çik"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
actor_ci              =  Suffix("çi"     , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
ordinal_inci          =  Suffix("inci"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
approximative_si      =  Suffix("si"     , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
approximative_imsi    =  Suffix("imsi"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
privative_siz         =  Suffix("siz"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
philicative_cil       =  Suffix("cil"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
composessive_li       =  Suffix("li"     , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
suitative_lik         =  Suffix("lik"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)

#N2V suffixes
transformative_les    =  Suffix("leş"    , WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
absentative_se        =  Suffix("se"     , WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
acquirative_len       =  Suffix("len"    , WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
verbifier_e           =  Suffix("e"      , WType.Noun, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.No)
aplicative_le         =  Suffix("le"     , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)

#V2N suffixes
infinitive_ma         =  Suffix("ma"     , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
infinitive_mak        =  Suffix("mak"    , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
factative_en          =  Suffix("en"     , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
constofactative_gen   =  Suffix("gen"    , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
infinitive_iş         =  Suffix("iş"     , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
perfectative_ik       =  Suffix("ik"     , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_i           =  Suffix("i"      , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_gi          =  Suffix("gi"     , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_im          =  Suffix("im"     , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
nounifier_in          =  Suffix("in"     , WType.Verb, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
## Çıkarım 1 küçük ünlü  uyumu  yalnızca eki i ı u ü olarak değiştirir
## Çıkarım 2 küçük ünlü  uyumu  e/a lı eklerde görülmez
## Çıkarım 3 eylemden eylem yapan eklerin hepsi küçük ünlü uyumuna girer
## Çıkarım 4 addan    eylem yapan eklerin hiçi  küçük ünlü uyumuna girmez  