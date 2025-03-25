from enum import Enum
import word_methods as wrd
class WType(Enum):
    Noun = 0
    Verb = 1

class HasMajorHarmony(Enum):     ## I have to change this naming 
    Yes = 0                   ## 1- its ugly
    No = 1                 ## second the word_methods already has the samenamed enums

class HasMinorHarmony(Enum):
    Yes = 0
    No = 1

class Suffix():
    def __init__(self,suffix,comes_to, makes,major_harmony,minor_harmony):
        self.suffix = suffix
        self.comes_to = comes_to
        self.makes = makes
        self.major_harmony = major_harmony
        self.minor_harmony = minor_harmony

    def form(self,word):
        if(self.major_harmony == HasMajorHarmony.Yes):
            word_harmony =wrd.major_harmony(word)
            if( word_harmony== wrd.MajorHarmony.Back): pass
                ## if vowels in suffix are already in wrd.back_vowels do nothing
                ## else swap the vowel with its conjugate
            if(word_harmony == wrd.MajorHarmony.Front): pass
                ## if vowels in suffix are already in wrd.front_vowels do nothing
                ## else swap the vowel with its conjugate

        if(self.minor_harmony == HasMinorHarmony.Yes): ## Buradaki bilgilerin dogrulugu supheli
            word_harmony = wrd.minor_harmony(word)

            if( word_harmony  == wrd.MinorHarmony.BackRound  ): pass
                ## if vowels in suffix are already in ['u'] do nothing
                ## else swap the vowel with 'u'
            if( word_harmony  == wrd.MinorHarmony.BackWide   ): pass
                ## if vowels in suffix are already in ['u'] do nothing
                ## else swap the vowel with 'u'
            if( word_harmony  == wrd.MinorHarmony.FrontRound ): pass
                ## if vowels in suffix are already in ['ü'] do nothing
                ## else swap the vowel with 'ü'
            if( word_harmony  == wrd.MinorHarmony.FrontWide  ): pass
                ## if vowels in suffix are already in ['e','i'] do nothing
                ## else swap the vowel with 'u'

        if(word[len(word)-1] in wrd.vowels and self.suffix[0] in wrd.vowels):pass  ## belk' bunu basha alip optimize edebilirsin
            ## trim the leading vowel from the result 



     
# V2V suffixes    
reflexive_is = Suffix('iş', WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
reflexive_ik = Suffix("ik", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_t     = Suffix("it", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_tir   = Suffix("tir",WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
active_ir    = Suffix("ir", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_il   = Suffix("il", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
passive_in   = Suffix("in", WType.Verb, WType.Verb, HasMajorHarmony.Yes, HasMinorHarmony.Yes)


#N2N suffixes
plural                =  Suffix("ler"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
counting_er           =  Suffix("er"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
cooperative_daş       =  Suffix("daş"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_ca           =  Suffix("ça"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
relative_sal          =  Suffix("sel"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.No)
actor                 =  Suffix("çi"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
dimunitive_cik        =  Suffix("çik"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
ordinal               =  Suffix("inci"  , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
approximative_si      =  Suffix("si"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
approximative_imsi    =  Suffix("imsi"  , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
approximative_imtrak  =  Suffix("imtrak", WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
privative             =  Suffix("siz"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
philicative_cil       =  Suffix("cil"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
composessive_li       =  Suffix("li"    , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)
suitative_lik         =  Suffix("lik"   , WType.Noun, WType.Noun, HasMajorHarmony.Yes, HasMinorHarmony.Yes)   