from enum import Enum, IntEnum
import util.word_methods as wrd

# Eklerin hiyerarşisi.
# Kural: Bir ek, kendinden daha düşük numaralı bir gruptan sonra GELEMEZ.
class SuffixGroup(IntEnum):
    DERIVATIONAL = 10      # Yapım Ekleri (-lik, -ci, -me, -mek)
    PLURAL = 20            # Çoğul Eki (-ler)
    POSSESSIVE = 30        # İyelik Ekleri (-im, -in)
    CASE = 40              # Hal Ekleri (-e, -de)
    POST_CASE = 45         # Hal ekinden sonra gelen istisnalar (-ki)
    PREDICATIVE = 50       # Şahıs / Çekim Ekleri (-im, -sin, -ler)
    TERMINAL = 60          # Zincir sonu Ek-fiiller (-dir, -miş, -di) -> Şahıs ekinden sonra gelebilir
    GERUND = 90            # Zarf Fiiller (-ip, -erek) -> ZİNCİRİ KİLİTLER, üzerine ek almaz.

class Type(Enum):
    NOUN = "noun"
    VERB = "verb"
    BOTH = "both"

class HasMajorHarmony(Enum):
    Yes = 0
    No = 1

class HasMinorHarmony(Enum):
    Yes = 0
    No = 1

class Suffix:
    def __init__(self, name, suffix, comes_to, makes, 
                 form_function=None, major_harmony=None, minor_harmony=None, needs_y_buffer=False,
                 group=SuffixGroup.DERIVATIONAL, is_unique=False):
        
        self.name = name
        self.suffix = str(suffix)
        self.comes_to = comes_to
        self.makes = makes
        self.major_harmony = major_harmony
        self.minor_harmony = minor_harmony
        self.needs_y_buffer = needs_y_buffer
        self.form_function = form_function if form_function else self._default_form
        
        # Hiyerarşi ve Tekrarlama Kontrolü
        self.group = group
        self.is_unique = is_unique
    
    def form(self, word):
        return self.form_function(word, self)
    
    @staticmethod
    def _default_form(word, suffix_obj):
        # 1. Baz formu al
        base = suffix_obj.suffix
        
        # 2. Uyumları uygula
        base = Suffix._apply_major_harmony(word, base, suffix_obj.major_harmony)
        base = Suffix._apply_minor_harmony(word, base, suffix_obj.minor_harmony)
        
        # 3. Sertleşme (Hardening) uygula (Kelime sonuna göre ekin başı değişir: d->t)
        base = Suffix._apply_consonant_hardening(word, base)
        
        # Olası formlar listesi
        candidates = [base]
        
        # 4. Kaynaştırma harfi (Buffer) varyasyonlarını ekle
        if Suffix._should_add_buffer_variants(word, base):
            if suffix_obj.needs_y_buffer:
                candidates.append('y' + base)
                candidates.append('ğ' + base) # Nadir durumlar için
            if len(base) > 1:
                # Ünlü düşmesi/çakışması durumunda ekin ilk harfini at (örn: ne-in -> neyin/nin)
                candidates.append(base[1:])
        
        # 5. Yumuşama (Softening) varyasyonlarını ekle (Ekin sonuna göre: k->ğ)
        # Mevcut adayların her biri için yumuşamış bir versiyon var mı diye bakarız.
        final_results = []
        for cand in candidates:
            final_results.append(cand) # Orijinal hali ekle (örn: ecek)
            
            softened = Suffix._apply_softening(cand)
            if softened != cand:
                final_results.append(softened) # Yumuşamış hali ekle (örn: eceğ)
        
        return final_results
    
    @staticmethod
    def _apply_major_harmony(word, result, major_harmony):
        if major_harmony != HasMajorHarmony.Yes:
            return result
        
        if wrd.major_harmony(word) == wrd.MajorHarmony.BACK:
            result = result.replace("e", "a")
            result = result.replace("i", "ı")
            result = result.replace("ü", "u")
            result = result.replace("ö", "o")
        
        return result
    
    @staticmethod
    def _apply_minor_harmony(word, result, minor_harmony):
        if minor_harmony != HasMinorHarmony.Yes:
            return result
        
        word_harmony = wrd.minor_harmony(word)
        
        if word_harmony == wrd.MinorHarmony.BACK_ROUND:
            result = result.replace("ı", "u")
        elif word_harmony == wrd.MinorHarmony.FRONT_ROUND:
            result = result.replace("i", "ü")
        
        return result
    
    @staticmethod
    def _apply_consonant_hardening(word, result):
        """
        Ünsüz Sertleşmesi (Benzeşmesi):
        Fıstıkçı Şahap ile biten kelimeye 'c, d, g' ile başlayan ek gelirse 'ç, t, k' olur.
        """
        if not word or not result:
            return result
        
        if word[-1] not in wrd.HARD_CONSONANTS:
            return result
        
        first_char = result[0]
        # Yumuşak ünsüz -> Sert ünsüz haritası
        hardening_map = {'g': 'k', 'd': 't', 'c': 'ç', 'ğ': 'k'}
        
        if first_char in hardening_map:
             return hardening_map[first_char] + result[1:]
             
        return result

    @staticmethod
    def _apply_softening(form):
        """
        Ünsüz Yumuşaması (Suffix Softening):
        Ekin kendisi ünlü ile başlayan başka bir ek aldığında sonundaki harf değişebilir.
        Bu metot, ekin son harfini kontrol eder ve yumuşamış halini döndürür.
        
        Örnek: 'ecek' -> 'eceğ', 'dik' -> 'diğ', 'amaç' -> 'amac'
        """
        if not form:
            return form
        
        last_char = form[-1]
        
        # Sık görülen: k -> ğ (Gelecek-im -> Geleceğim)
        if last_char == 'k':
            return form[:-1] + 'ğ'
        
        # Diğer yumuşamalar (Suffixlerde daha nadir ama mümkün)
        elif last_char == 'ç':
            return form[:-1] + 'c'
        elif last_char == 'p':
            return form[:-1] + 'b'
        elif last_char == 't':
            return form[:-1] + 'd'
        
        # Eğer yumuşama yoksa orijinali döndür
        return form

    @staticmethod
    def _should_add_buffer_variants(word, result):
        return (word and result and 
                word[-1] in wrd.VOWELS and 
                result[0] in wrd.VOWELS)