from enum import Enum

class WType(Enum):
    Noun = 0
    Verb = 1

class Suffix():
    def __init__(self,suffix,comes_to, makes):
        self.suffix = suffix
        self.comes_to = comes_to
        self.makes = makes

     
    
reflexive_is = Suffix('iş', WType.Verb, WType.Verb)
reflexive_ik = Suffix("ik", WType.Verb, WType.Verb)
active_t     = Suffix("it", WType.Verb, WType.Verb)
active_tir   = Suffix("tir",WType.Verb, WType.Verb)
active_ir    = Suffix("ir", WType.Verb, WType.Verb)
passive_il   = Suffix("il", WType.Verb, WType.Verb)
passive_in   = Suffix("in", WType.Verb, WType.Verb)


plural                =  Suffix("ler"   , WType.Noun, WType.Noun)
actor                 =  Suffix("çi"    , WType.Noun, WType.Noun)
dimunitive_cik        =  Suffix("çik"   , WType.Noun, WType.Noun)
ordinal               =  Suffix("inci"  , WType.Noun, WType.Noun)
approximative_si      =  Suffix("si"    , WType.Noun, WType.Noun)
approximative_imsi    =  Suffix("imsi"  , WType.Noun, WType.Noun)
approximative_imtrak  =  Suffix("imtrak", WType.Noun, WType.Noun)
privative             =  Suffix("siz"   , WType.Noun, WType.Noun)
counting_er           =  Suffix("er"    , WType.Noun, WType.Noun)
cooperative_daş       =  Suffix("daş"   , WType.Noun, WType.Noun)
philicative_cil       =  Suffix("cil"   , WType.Noun, WType.Noun)
relative_ca           =  Suffix("ça"    , WType.Noun, WType.Noun)
relative_sal          =  Suffix("sel"   , WType.Noun, WType.Noun)
composessive_li       =  Suffix("li"    , WType.Noun, WType.Noun)
suitative_lik         =  Suffix("lik"   , WType.Noun, WType.Noun)   