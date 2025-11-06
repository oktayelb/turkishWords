class TurkishDecomposer:
    def __init__(self):
        # Define vowel harmony groups
        self.front_vowels = "eiöü"
        self.back_vowels = "aıou"
        self.rounded_vowels = "oöuü"
        self.unrounded_vowels = "aeıi"
        self.high_vowels = "iıuü"
        self.low_vowels = "eaöo"
        
        # Consonants
        self.voiceless_consonants = "pçtksşhf"
        self.voiced_consonants = "bcdgğzjvl"
        
        # Dictionary of common nominal (noun, adjective, etc.) suffixes with their meanings
        self.nominal_suffixes = {
            "lar": "plural",
            "ler": "plural",
            "da": "locative (at/in)",
            "de": "locative (at/in)",
            "ta": "locative (at/in)",
            "te": "locative (at/in)",
            "dan": "ablative (from)",
            "den": "ablative (from)",
            "tan": "ablative (from)",
            "ten": "ablative (from)",
            "a": "dative (to)",
            "e": "dative (to)",
            "ya": "dative (to)",
            "ye": "dative (to)",
            "ı": "accusative",
            "i": "accusative",
            "u": "accusative",
            "ü": "accusative",
            "yı": "accusative",
            "yi": "accusative",
            "yu": "accusative",
            "yü": "accusative",
            "ın": "genitive (of)",
            "in": "genitive (of)",
            "un": "genitive (of)",
            "ün": "genitive (of)",
            "nın": "genitive (of)",
            "nin": "genitive (of)",
            "nun": "genitive (of)",
            "nün": "genitive (of)",
            "lı": "with",
            "li": "with",
            "lu": "with",
            "lü": "with",
            "sız": "without",
            "siz": "without",
            "suz": "without",
            "süz": "without",
            "cı": "person who does/sells",
            "ci": "person who does/sells",
            "cu": "person who does/sells",
            "cü": "person who does/sells",
            "çı": "person who does/sells",
            "çi": "person who does/sells",
            "çu": "person who does/sells",
            "çü": "person who does/sells",
            "lık": "place/state of being",
            "lik": "place/state of being",
            "luk": "place/state of being",
            "lük": "place/state of being",
            "ki": "that is at",
            "daş": "fellow",
            "deş": "fellow",
            "taş": "fellow",
            "teş": "fellow",
            "la": "with",
            "le": "with",
            "ca": "according to/in the manner of",
            "ce": "according to/in the manner of",
            "ça": "according to/in the manner of",
            "çe": "according to/in the manner of",
            "cık": "diminutive (small)",
            "cik": "diminutive (small)",
            "cuk": "diminutive (small)",
            "cük": "diminutive (small)",
            "çık": "diminutive (small)",
            "çik": "diminutive (small)",
            "çuk": "diminutive (small)",
            "çük": "diminutive (small)",
            "m": "my (1st person possessive)",
            "n": "your (2nd person possessive)",
            "ı": "his/her/its (3rd person possessive)",
            "i": "his/her/its (3rd person possessive)",
            "u": "his/her/its (3rd person possessive)",
            "ü": "his/her/its (3rd person possessive)",
            "sı": "his/her/its (3rd person possessive)",
            "si": "his/her/its (3rd person possessive)",
            "su": "his/her/its (3rd person possessive)",
            "sü": "his/her/its (3rd person possessive)",
            "mız": "our (1st person plural possessive)",
            "miz": "our (1st person plural possessive)",
            "muz": "our (1st person plural possessive)",
            "müz": "our (1st person plural possessive)",
            "ımız": "our (1st person plural possessive)",
            "imiz": "our (1st person plural possessive)",
            "umuz": "our (1st person plural possessive)",
            "ümüz": "our (1st person plural possessive)",
            "nız": "your (2nd person plural possessive)",
            "niz": "your (2nd person plural possessive)",
            "nuz": "your (2nd person plural possessive)",
            "nüz": "your (2nd person plural possessive)",
            "ınız": "your (2nd person plural possessive)",
            "iniz": "your (2nd person plural possessive)",
            "unuz": "your (2nd person plural possessive)",
            "ünüz": "your (2nd person plural possessive)",
            "ları": "their (3rd person plural possessive)",
            "leri": "their (3rd person plural possessive)",
        }
        
        # Dictionary of common verbal suffixes with their meanings
        self.verbal_suffixes = {
            # Infinitive and verbal noun forms
            "mak": "infinitive",
            "mek": "infinitive",
            "ma": "verbal noun/negative",
            "me": "verbal noun/negative",
            
            # Negation
            "ma": "negative",
            "me": "negative",
            
            # Tenses
            "dı": "past definite",
            "di": "past definite",
            "du": "past definite",
            "dü": "past definite",
            "tı": "past definite",
            "ti": "past definite",
            "tu": "past definite",
            "tü": "past definite",
            
            "mış": "reported past/evidential",
            "miş": "reported past/evidential",
            "muş": "reported past/evidential",
            "müş": "reported past/evidential",
            
            "yor": "present continuous",
            "iyor": "present continuous",
            "ıyor": "present continuous",
            "uyor": "present continuous",
            "üyor": "present continuous",
            
            "acak": "future",
            "ecek": "future",
            "yacak": "future",
            "yecek": "future",
            
            "ar": "aorist",
            "er": "aorist",
            "ır": "aorist",
            "ir": "aorist",
            "ur": "aorist",
            "ür": "aorist",
            
            # Modality
            "abil": "ability/possibility",
            "ebil": "ability/possibility",
            "yabil": "ability/possibility",
            "yebil": "ability/possibility",
            
            "malı": "necessity",
            "meli": "necessity",
            
            "sa": "conditional",
            "se": "conditional",
            
            # Voice - change verb's relationship to subject/object
            "n": "passive/reflexive",
            
            "ın": "reflexive",
            "in": "reflexive",
            "un": "reflexive",
            "ün": "reflexive",
            
            "ıl": "passive",
            "il": "passive",
            "ul": "passive",
            "ül": "passive",
            
            "ış": "reciprocal/mutual",
            "iş": "reciprocal/mutual",
            "uş": "reciprocal/mutual",
            "üş": "reciprocal/mutual",
            
            # Causative forms
            "t": "causative",
            "dır": "causative",
            "dir": "causative",
            "dur": "causative", 
            "dür": "causative",
            "tır": "causative",
            "tir": "causative",
            "tur": "causative",
            "tür": "causative",
            
            # Gerunds (verbal adverbs)
            "ip": "by (doing)",
            "ıp": "by (doing)",
            "up": "by (doing)",
            "üp": "by (doing)",
            
            "arak": "by (doing)",
            "erek": "by (doing)",
            
            "ken": "while (doing)",
            "ınca": "when/once (doing)",
            "ince": "when/once (doing)",
            "unca": "when/once (doing)",
            "ünce": "when/once (doing)",
            
            "dıkça": "as long as/the more",
            "dikçe": "as long as/the more",
            "dukça": "as long as/the more",
            "dükçe": "as long as/the more",
            
            # Participles (verbal adjectives)
            "an": "one who (does)",
            "en": "one who (does)",
            "yan": "one who (does)",
            "yen": "one who (does)",
            
            "dık": "that which is (done)",
            "dik": "that which is (done)",
            "duk": "that which is (done)",
            "dük": "that which is (done)",
            
            "acak": "that which will be (done)",
            "ecek": "that which will be (done)",
            
            "mış": "that which has been (done)",
            "miş": "that which has been (done)",
            "muş": "that which has been (done)",
            "müş": "that which has been (done)",
            
            # Question particle
            "mı": "question particle",
            "mi": "question particle",
            "mu": "question particle",
            "mü": "question particle",
        }
        
        # Personal suffixes - added to verbs to indicate person
        # Group 1 (present, future, aorist, conditional)
        self.personal_suffixes_group1 = {
            "m": "1st person singular",
            "n": "2nd person singular",
            # 3rd person singular has no suffix
            "z": "1st person plural",
            "k": "1st person plural",
            "nız": "2nd person plural",
            "niz": "2nd person plural",
            "nuz": "2nd person plural",
            "nüz": "2nd person plural",
            "sınız": "2nd person plural",
            "siniz": "2nd person plural",
            "sunuz": "2nd person plural",
            "sünüz": "2nd person plural",
            "lar": "3rd person plural",
            "ler": "3rd person plural",
        }
        
        # Group 2 (past tense, conditional)
        self.personal_suffixes_group2 = {
            "m": "1st person singular",
            "n": "2nd person singular",
            # 3rd person singular has no suffix
            "k": "1st person plural",
            "nız": "2nd person plural",
            "niz": "2nd person plural",
            "nuz": "2nd person plural",
            "nüz": "2nd person plural",
            "lar": "3rd person plural",
            "ler": "3rd person plural",
        }
        
        # Combined personal suffixes
        self.personal_suffixes = {
            # Group 1 variants with vowels
            "ım": "1st person singular",
            "im": "1st person singular",
            "um": "1st person singular",
            "üm": "1st person singular",
            "yım": "1st person singular",
            "yim": "1st person singular",
            "yum": "1st person singular",
            "yüm": "1st person singular",
            
            "sın": "2nd person singular",
            "sin": "2nd person singular",
            "sun": "2nd person singular",
            "sün": "2nd person singular",
            
            "ız": "1st person plural",
            "iz": "1st person plural",
            "uz": "1st person plural",
            "üz": "1st person plural",
            "yız": "1st person plural",
            "yiz": "1st person plural",
            "yuz": "1st person plural",
            "yüz": "1st person plural",
            
            "sınız": "2nd person plural",
            "siniz": "2nd person plural",
            "sunuz": "2nd person plural",
            "sünüz": "2nd person plural",
            
            "lar": "3rd person plural",
            "ler": "3rd person plural",
            
            # Group 2 variants
            "dım": "1st person singular",
            "dim": "1st person singular",
            "dum": "1st person singular",
            "düm": "1st person singular",
            "tım": "1st person singular",
            "tim": "1st person singular",
            "tum": "1st person singular",
            "tüm": "1st person singular",
            
            "dın": "2nd person singular",
            "din": "2nd person singular",
            "dun": "2nd person singular",
            "dün": "2nd person singular",
            "tın": "2nd person singular",
            "tin": "2nd person singular",
            "tun": "2nd person singular",
            "tün": "2nd person singular",
            
            "dık": "1st person plural",
            "dik": "1st person plural",
            "duk": "1st person plural",
            "dük": "1st person plural",
            "tık": "1st person plural",
            "tik": "1st person plural",
            "tuk": "1st person plural",
            "tük": "1st person plural",
            
            "dınız": "2nd person plural",
            "diniz": "2nd person plural",
            "dunuz": "2nd person plural",
            "dünüz": "2nd person plural",
            "tınız": "2nd person plural",
            "tiniz": "2nd person plural",
            "tunuz": "2nd person plural",
            "tünüz": "2nd person plural",
            
            "dılar": "3rd person plural",
            "diler": "3rd person plural",
            "dular": "3rd person plural",
            "düler": "3rd person plural",
            "tılar": "3rd person plural",
            "tiler": "3rd person plural",
            "tular": "3rd person plural",
            "tüler": "3rd person plural",
        }
        
        # Common Turkish noun stems
        self.common_noun_stems = {
            "ev": "house", "kitap": "book", "araba": "car", "okul": "school", 
            "su": "water", "insan": "human", "çocuk": "child", "masa": "table",
            "kapı": "door", "pencere": "window", "kalem": "pencil", "gün": "day",
            "adam": "man", "kadın": "woman", "arkadaş": "friend", "öğrenci": "student",
            "öğretmen": "teacher", "doktor": "doctor", "köpek": "dog", "kedi": "cat",
            "yol": "road", "göz": "eye", "el": "hand", "ayak": "foot",
            "baş": "head", "anne": "mother", "baba": "father", "kardeş": "sibling",
            "dil": "language", "şehir": "city", "ülke": "country", "dünya": "world"
        }
        
        # Common Turkish verb stems
        self.common_verb_stems = {
            "gel": "come", "git": "go", "yap": "do/make", "ol": "be/become", 
            "gör": "see", "bak": "look", "al": "take/buy", "ver": "give",
            "ye": "eat", "iç": "drink", "oku": "read", "yaz": "write",
            "konuş": "speak", "dinle": "listen", "anla": "understand", "düşün": "think",
            "bil": "know", "iste": "want", "sev": "love", "say": "count/respect",
            "dur": "stop/stand", "otur": "sit", "koş": "run", "yürü": "walk",
            "uyu": "sleep", "çalış": "work", "söyle": "say/tell", "gül": "laugh"
        }

    def get_last_vowel(self, word):
        """Get the last vowel in a word"""
        for char in reversed(word):
            if char.lower() in self.front_vowels + self.back_vowels:
                return char.lower()
        return None

    def binary_search_word(self, word, words_list):
        """Perform binary search on the Turkish dictionary"""
        if not words_list:
            return False
            
        low = 0
        high = len(words_list) - 1
        
        while low <= high:
            mid = (low + high) // 2
            mid_word = words_list[mid].strip()
            
            # Turkish alphabetical order comparison
            comparison = self.compare_turkish_words(word, mid_word)
            
            if comparison == 0:  # Words match
                return True
            elif comparison < 0:  # word comes before mid_word
                high = mid - 1
            else:  # word comes after mid_word
                low = mid + 1
        
        return False

    def compare_turkish_words(self, word1, word2):
        """Compare two words according to Turkish alphabetical order"""
        # Turkish alphabet order: a, b, c, ç, d, e, f, g, ğ, h, ı, i, j, k, l, m, n, o, ö, p, r, s, ş, t, u, ü, v, y, z
        turkish_order = {
            'a': 1, 'b': 2, 'c': 3, 'ç': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'ğ': 9, 'h': 10,
            'ı': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17, 'o': 18, 'ö': 19,
            'p': 20, 'r': 21, 's': 22, 'ş': 23, 't': 24, 'u': 25, 'ü': 26, 'v': 27, 'y': 28, 'z': 29
        }
        
        word1 = word1.lower()
        word2 = word2.lower()
        
        for i in range(min(len(word1), len(word2))):
            c1 = word1[i]
            c2 = word2[i]
            
            if c1 != c2:
                return turkish_order.get(c1, 0) - turkish_order.get(c2, 0)
        
        # If one word is a prefix of the other, the shorter word comes first
        return len(word1) - len(word2)

    def load_dictionary(self, filepath):
        """Load the Turkish dictionary from a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return [line.strip() for line in file]
        except FileNotFoundError:
            print(f"Warning: Dictionary file {filepath} not found.")
            return []

    def check_stem_in_dictionary(self, stem, stem_type):
        """Check if a stem exists in the Turkish dictionary"""
        if not hasattr(self, 'dictionary_words'):
            # If dictionary hasn't been loaded, try to load it
            self.dictionary_words = self.load_dictionary('words.txt')
            
        if not self.dictionary_words:
            return False  # Dictionary not available
        
        # For verb stems, we need to check with infinitive form (mek/mak)
        if stem_type == "verb" or stem_type == "verb (guessed)":
            last_vowel = self.get_last_vowel(stem)
            if last_vowel in self.front_vowels:
                return self.binary_search_word(stem + "mek", self.dictionary_words)
            else:
                return self.binary_search_word(stem + "mak", self.dictionary_words)
        else:
            # For nominal stems, check directly
            return self.binary_search_word(stem, self.dictionary_words)

    def verify_suffix_compatibility(self, stem_type, suffix):
        """Verify that a suffix is compatible with the given stem type"""
        if stem_type.startswith("verb"):
            # Check if suffix is in verbal_suffixes or personal_suffixes
            return suffix in self.verbal_suffixes or suffix in self.personal_suffixes
        elif stem_type.startswith("noun"):
            # Check if suffix is in nominal_suffixes or personal_suffixes 
            return suffix in self.nominal_suffixes or suffix in self.personal_suffixes
        return True  # If stem type is unknown, be permissive

    # Add missing methods
    def check_vowel_harmony(self, stem, suffix):
        """Check if a suffix follows vowel harmony rules with the stem"""
        # This is a simplified implementation
        last_vowel = self.get_last_vowel(stem)
        if not last_vowel:
            return True  # If no vowel in stem, assume harmony is fine
            
        first_suffix_vowel = None
        for char in suffix:
            if char in self.front_vowels + self.back_vowels:
                first_suffix_vowel = char
                break
                
        if not first_suffix_vowel:
            return True  # If no vowel in suffix, assume harmony is fine
            
        # Front/back vowel harmony
        if last_vowel in self.front_vowels and first_suffix_vowel in self.front_vowels:
            return True
        if last_vowel in self.back_vowels and first_suffix_vowel in self.back_vowels:
            return True
            
        # If not following the rule, no harmony
        return False

    def apply_consonant_harmony(self, stem, suffix):
        """Apply consonant harmony rules to a suffix based on the stem"""
        # Simplified implementation
        if not stem or not suffix:
            return suffix
            
        last_char = stem[-1]
        first_suffix_char = suffix[0]
        
        # D/T alternation (voice assimilation)
        if first_suffix_char == 'd' and last_char in self.voiceless_consonants:
            return 't' + suffix[1:]
        
        # No other harmony rules applied for simplicity
        return suffix

    def handle_stem_changes(self, stem, word):
        """Handle potential stem changes like final consonant devoicing, etc."""
        variations = [(stem, "original")]
        
        # Final consonant devoicing (p->b, t->d, k->g/ğ)
        if stem.endswith('p'):
            variations.append((stem[:-1] + 'b', "final p->b"))
        elif stem.endswith('t'):
            variations.append((stem[:-1] + 'd', "final t->d"))
        elif stem.endswith('k'):
            variations.append((stem[:-1] + 'g', "final k->g"))
            variations.append((stem[:-1] + 'ğ', "final k->ğ"))
        elif stem.endswith('ç'):
            variations.append((stem[:-1] + 'c', "final ç->c"))
        
        return variations

    def decompose(self, word):
        """Attempt to decompose a Turkish word into stem and suffixes"""
        word = word.lower()
        possible_decompositions = []
        
        # Load dictionary if not already loaded
        if not hasattr(self, 'dictionary_words'):
            self.dictionary_words = self.load_dictionary('words.txt')
        
        # Try verbal decompositions
        for stem, meaning in self.common_verb_stems.items():
            if word.startswith(stem):
                remaining = word[len(stem):]
                if not remaining:  # Exact match to a verb stem
                    possible_decompositions.append({
                        "stem": stem,
                        "stem_type": "verb",
                        "stem_meaning": meaning,
                        "suffixes": []
                    })
                    continue
                    
                # Try to identify suffixes
                identified_suffixes = self._identify_suffixes(remaining, stem, "verb")
                for suffix_breakdown in identified_suffixes:
                    possible_decompositions.append({
                        "stem": stem,
                        "stem_type": "verb",
                        "stem_meaning": meaning,
                        "suffixes": suffix_breakdown
                    })
                    
            # Check for possible stem variations with consonant changes
            stem_variations = self.handle_stem_changes(stem, word)
            for modified_stem, change_type in stem_variations:
                if word.startswith(modified_stem) and modified_stem != stem:
                    remaining = word[len(modified_stem):]
                    identified_suffixes = self._identify_suffixes(remaining, modified_stem, "verb")
                    for suffix_breakdown in identified_suffixes:
                        possible_decompositions.append({
                            "stem": stem,  # Original stem
                            "modified_stem": modified_stem,
                            "stem_change": change_type,
                            "stem_type": "verb",
                            "stem_meaning": meaning,
                            "suffixes": suffix_breakdown
                        })
        
        # Try nominal decompositions
        for stem, meaning in self.common_noun_stems.items():
            if word.startswith(stem):
                remaining = word[len(stem):]
                if not remaining:  # Exact match to a noun stem
                    possible_decompositions.append({
                        "stem": stem,
                        "stem_type": "noun",
                        "stem_meaning": meaning,
                        "suffixes": []
                    })
                    continue
                    
                # Try to identify suffixes
                identified_suffixes = self._identify_suffixes(remaining, stem, "noun")
                for suffix_breakdown in identified_suffixes:
                    possible_decompositions.append({
                        "stem": stem,
                        "stem_type": "noun",
                        "stem_meaning": meaning,
                        "suffixes": suffix_breakdown
                    })
                    
            # Check for possible stem variations with consonant changes
            stem_variations = self.handle_stem_changes(stem, word)
            for modified_stem, change_type in stem_variations:
                if word.startswith(modified_stem) and modified_stem != stem:
                    remaining = word[len(modified_stem):]
                    identified_suffixes = self._identify_suffixes(remaining, modified_stem, "noun")
                    for suffix_breakdown in identified_suffixes:
                        possible_decompositions.append({
                            "stem": stem,  # Original stem
                            "modified_stem": modified_stem,
                            "stem_change": change_type,
                            "stem_type": "noun",
                            "stem_meaning": meaning,
                            "suffixes": suffix_breakdown
                        })
        
        # If no known stems were found, try to guess the stem using dictionary lookup
        if not possible_decompositions:
            for i in range(1, len(word)):
                potential_stem = word[:i]
                remaining = word[i:]
                
                # Check if potential stem is in dictionary as a verb
                if self.check_stem_in_dictionary(potential_stem, "verb"):
                    identified_suffixes = self._identify_suffixes(remaining, potential_stem, "verb")
                    for suffix_breakdown in identified_suffixes:
                        # Verify suffix compatibility
                        if all(self.verify_suffix_compatibility("verb", suffix['suffix']) for suffix in suffix_breakdown):
                            possible_decompositions.append({
                                "stem": potential_stem,
                                "stem_type": "verb (from dictionary)",
                                "stem_meaning": "found in dictionary",
                                "suffixes": suffix_breakdown
                            })
                
                # Check if potential stem is in dictionary as a noun
                if self.check_stem_in_dictionary(potential_stem, "noun"):
                    identified_suffixes = self._identify_suffixes(remaining, potential_stem, "noun")
                    for suffix_breakdown in identified_suffixes:
                        # Verify suffix compatibility
                        if all(self.verify_suffix_compatibility("noun", suffix['suffix']) for suffix in suffix_breakdown):
                            possible_decompositions.append({
                                "stem": potential_stem,
                                "stem_type": "noun (from dictionary)",
                                "stem_meaning": "found in dictionary",
                                "suffixes": suffix_breakdown
                            })
                    
                # Also try potential stems with consonant changes
                stem_variations = self.handle_stem_changes(potential_stem, word)
                for modified_stem, change_type in stem_variations:
                    if word.startswith(modified_stem) and modified_stem != potential_stem:
                        remaining = word[len(modified_stem):]
                        
                        # Try as verb if in dictionary
                        if self.check_stem_in_dictionary(potential_stem, "verb"):
                            identified_suffixes = self._identify_suffixes(remaining, modified_stem, "verb")
                            for suffix_breakdown in identified_suffixes:
                                if all(self.verify_suffix_compatibility("verb", suffix['suffix']) for suffix in suffix_breakdown):
                                    possible_decompositions.append({
                                        "stem": potential_stem,
                                        "modified_stem": modified_stem,
                                        "stem_change": change_type,
                                        "stem_type": "verb (from dictionary)",
                                        "stem_meaning": "found in dictionary",
                                        "suffixes": suffix_breakdown
                                    })
                        
                        # Try as noun if in dictionary
                        if self.check_stem_in_dictionary(potential_stem, "noun"):
                            identified_suffixes = self._identify_suffixes(remaining, modified_stem, "noun")
                            for suffix_breakdown in identified_suffixes:
                                if all(self.verify_suffix_compatibility("noun", suffix['suffix']) for suffix in suffix_breakdown):
                                    possible_decompositions.append({
                                        "stem": potential_stem,
                                        "modified_stem": modified_stem,
                                        "stem_change": change_type,
                                        "stem_type": "noun (from dictionary)",
                                        "stem_meaning": "found in dictionary",
                                        "suffixes": suffix_breakdown
                                    })
                    
            # If still no decompositions found with dictionary, try as unknown
            if not possible_decompositions:
                for i in range(1, len(word)):
                    potential_stem = word[:i]
                    remaining = word[i:]
                    
                    # Try as verb
                    identified_suffixes = self._identify_suffixes(remaining, potential_stem, "verb")
                    for suffix_breakdown in identified_suffixes:
                        if all(self.verify_suffix_compatibility("verb", suffix['suffix']) for suffix in suffix_breakdown):
                            possible_decompositions.append({
                                "stem": potential_stem,
                                "stem_type": "verb (guessed)",
                                "stem_meaning": "unknown",
                                "suffixes": suffix_breakdown
                            })
                    
                    # Try as noun
                    identified_suffixes = self._identify_suffixes(remaining, potential_stem, "noun")
                    for suffix_breakdown in identified_suffixes:
                        if all(self.verify_suffix_compatibility("noun", suffix['suffix']) for suffix in suffix_breakdown):
                            possible_decompositions.append({
                                "stem": potential_stem,
                                "stem_type": "noun (guessed)",
                                "stem_meaning": "unknown",
                                "suffixes": suffix_breakdown
                            })
                            
                    # Also try potential stems with consonant changes
                    stem_variations = self.handle_stem_changes(potential_stem, word)
                    for modified_stem, change_type in stem_variations:
                        if modified_stem != potential_stem:
                            remaining = word[len(modified_stem):]
                            
                            # Try as verb
                            identified_suffixes = self._identify_suffixes(remaining, modified_stem, "verb")
                            for suffix_breakdown in identified_suffixes:
                                if all(self.verify_suffix_compatibility("verb", suffix['suffix']) for suffix in suffix_breakdown):
                                    possible_decompositions.append({
                                        "stem": potential_stem,
                                        "modified_stem": modified_stem,
                                        "stem_change": change_type,
                                        "stem_type": "verb (guessed)",
                                        "stem_meaning": "unknown",
                                        "suffixes": suffix_breakdown
                                    })
                            
                            # Try as noun
                            identified_suffixes = self._identify_suffixes(remaining, modified_stem, "noun")
                            for suffix_breakdown in identified_suffixes:
                                if all(self.verify_suffix_compatibility("noun", suffix['suffix']) for suffix in suffix_breakdown):
                                    possible_decompositions.append({
                                        "stem": potential_stem,
                                        "modified_stem": modified_stem,
                                        "stem_change": change_type,
                                        "stem_type": "noun (guessed)",
                                        "stem_meaning": "unknown",
                                        "suffixes": suffix_breakdown
                                    })
        
        # Sort decompositions by quality (known stems first, fewer unknown suffixes, etc.)
        possible_decompositions.sort(key=self._rate_decomposition_quality, reverse=True)
        
        return possible_decompositions

    def _rate_decomposition_quality(self, decomp):
        """Rate the quality of a decomposition to sort results"""
        score = 0
        
        # Known stems are better than guessed ones
        if "guessed" not in decomp["stem_type"]:
            score += 100
        if "from dictionary" in decomp["stem_type"]:
            score += 50
        
        # Fewer suffixes is generally better
        score -= len(decomp["suffixes"]) * 2
        
        # Known suffixes are better than unknown ones
        unknown_suffixes = sum(1 for suffix in decomp["suffixes"] if "unknown" in suffix["meaning"])
        score -= unknown_suffixes * 10
        
        # Complete parsing is better than partial
        suffix_coverage = sum(len(suffix["suffix"]) for suffix in decomp["suffixes"])
        if "modified_stem" in decomp:
            stem_len = len(decomp["modified_stem"])
        else:
            stem_len = len(decomp["stem"])
        
        if stem_len + suffix_coverage == len(decomp.get("reconstructed_word", "")):
            score += 25
        
        return score

    def _identify_suffixes(self, suffix_string, stem, stem_type):
        """Recursively identify possible suffix combinations"""
        if not suffix_string:
            return [[]]
        
        results = []
        
        # Check for buffer letters
        # When a suffix starting with a vowel is added to a word ending with a vowel,
        # a buffer consonant (y, n, s) is often inserted
        buffer_letters = ["y", "n", "s"]
        stem_ending = stem[-1] if stem else ""
        stem_has_vowel_ending = stem_ending in self.front_vowels + self.back_vowels
        
        # Apply consonant harmony when needed
        # Check all possible suffixes
        suffix_dict = self.verbal_suffixes if stem_type.startswith("verb") else self.nominal_suffixes
        all_suffixes = {**suffix_dict, **self.personal_suffixes}
        
        # Add the missing suffixes to check
        additional_causative_suffixes = {
            "gı": "causative",
            "gi": "causative",
            "gu": "causative",
            "gü": "causative"
        }
        all_suffixes.update(additional_causative_suffixes)
        
        for suffix, meaning in all_suffixes.items():
            # Skip empty suffixes
            if not suffix:
                continue
                
            # Handle buffer letters
            if stem_has_vowel_ending and suffix[0] in self.front_vowels + self.back_vowels:
                for buffer in buffer_letters:
                    # Check if the buffer + suffix is present
                    if suffix_string.startswith(buffer + suffix):
                        # Apply vowel harmony check
                        if self.check_vowel_harmony(stem, suffix):
                            remaining = suffix_string[len(buffer + suffix):]
                            sub_results = self._identify_suffixes(remaining, stem + buffer + suffix, stem_type)
                            
                            for sub_result in sub_results:
                                results.append([{"suffix": buffer + suffix, "meaning": meaning + " (with buffer letter)"}] + sub_result)
            
            # Normal suffix check (no buffer needed)
            # Apply consonant harmony rules
            harmonized_suffix = self.apply_consonant_harmony(stem, suffix)
            
            if suffix_string.startswith(harmonized_suffix):
                # Check vowel harmony
                if self.check_vowel_harmony(stem, harmonized_suffix):
                    remaining = suffix_string[len(harmonized_suffix):]
                    sub_results = self._identify_suffixes(remaining, stem + harmonized_suffix, stem_type)
                    
                    suffix_info = {"suffix": harmonized_suffix, "meaning": meaning}
                    if harmonized_suffix != suffix:
                        suffix_info["original_suffix"] = suffix
                        suffix_info["meaning"] += " (with consonant harmony)"
                    
                    for sub_result in sub_results:
                        results.append([suffix_info] + sub_result)
        
        # Try to handle vowel dropping (vowel elision)
        # For example: "oğul" + "u" can become "oğlu"
        last_two = stem[-2:] if len(stem) >= 2 else ""
        if len(last_two) == 2 and last_two[0] in self.front_vowels + self.back_vowels and last_two[1] in "lnr":
            # Words with a vowel followed by l, n, or r often drop the vowel when a suffix with a vowel is added
            # We'll try to check if removing the vowel helps identify suffixes
            modified_stem = stem[:-2] + stem[-1]  # Remove the vowel in the last syllable
            for suffix, meaning in all_suffixes.items():
                if suffix_string.startswith(suffix):
                    if self.check_vowel_harmony(modified_stem, suffix):
                        remaining = suffix_string[len(suffix):]
                        sub_results = self._identify_suffixes(remaining, modified_stem + suffix, stem_type)
                        
                        for sub_result in sub_results:
                            results.append([{"suffix": suffix, "meaning": meaning + " (with vowel drop in stem)"}] + sub_result)
        
        # If no complete decomposition found but some suffixes are identified, return partial result
        if not results:
            if len(suffix_string) <= 3:
                results.append([{"suffix": suffix_string, "meaning": "unknown suffix"}])
            else:
                # Try to break down longer unknown segments
                for i in range(1, len(suffix_string)):
                    part1 = suffix_string[:i]
                    part2 = suffix_string[i:]
                    results.append([
                        {"suffix": part1, "meaning": "unknown suffix"}, 
                        {"suffix": part2, "meaning": "unknown suffix"}
                    ])
            
        return results

    def format_decomposition(self, decompositions):
        """Format the decomposition results in a readable way"""
        if not decompositions:
            return "No possible decompositions found."
        
        formatted = []
        for i, decomp in enumerate(decompositions, 1):
            parts = [
                f"Decomposition {i}:",
                f"  Stem: {decomp['stem']} ({decomp['stem_type']}, meaning: {decomp['stem_meaning']})"
            ]
            
            if 'modified_stem' in decomp:
                parts.append(f"  Modified stem: {decomp['modified_stem']} ({decomp['stem_change']})")
            
            if decomp['suffixes']:
                parts.append("  Suffixes:")
                for suffix in decomp['suffixes']:
                    suffix_info = f"    - {suffix['suffix']}: {suffix['meaning']}"
                    if 'original_suffix' in suffix:
                        suffix_info += f" (from {suffix['original_suffix']})"
                    parts.append(suffix_info)
            else:
                parts.append("  No suffixes identified.")
                
            # Build the full word to verify our decomposition
            full_word = decomp.get('modified_stem', decomp['stem'])
            for suffix in decomp['suffixes']:
                full_word += suffix['suffix']
                
            parts.append(f"  Reconstructed word: {full_word}")
            
            # Add dictionary validation info
            if "from dictionary" in decomp["stem_type"]:
                parts.append("  Stem validated from dictionary: Yes")
            elif self.check_stem_in_dictionary(decomp['stem'], decomp['stem_type'].split()[0]):
                parts.append("  Stem found in dictionary: Yes")
            else:
                parts.append("  Stem found in dictionary: No")
                
            formatted.append("\n".join(parts))
        
        return "\n\n".join(formatted)


# Example usage
if __name__ == "__main__":
    decomposer = TurkishDecomposer()
    
    # Test with various examples showcasing different aspects of Turkish morphology
    test_words = [
        # Basic examples
        "evler",       # ev (house) + ler (plural)
        "kitaplar",    # kitap (book) + lar (plural)
        
        # Verb examples with tense and person markers
        "geliyorum",   # gel (come) + iyor (present continuous) + um (1st person singular)
        "gelmiyorum",  # gel (come) + mi (negative) + yor (present continuous) + um (1st person singular)
        
        # Noun with multiple suffixes
        "evlerimizde", # ev (house) + ler (plural) + imiz (1st person plural possessive) + de (locative)
    ]
    
    for word in test_words:
        print(f"\n{'='*60}\nAnalyzing: {word}")
        decompositions = decomposer.decompose(word)
        print(decomposer.format_decomposition(decompositions))

# Interactive usage
def interactive_mode():
    decomposer = TurkishDecomposer()
    
    print("Turkish Word Decomposer")
    print("Enter a Turkish word to decompose, or 'exit' to quit.")
    
    while True:
        word = input("\nEnter a Turkish word: ")
        if word.lower() == "exit":
            break
            
        decompositions = decomposer.decompose(word)
        print(decomposer.format_decomposition(decompositions))

# Uncomment to run interactive mode
# interactive_mode()