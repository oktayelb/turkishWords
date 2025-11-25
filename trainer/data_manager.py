import os
import json
import random
from typing import List, Optional, Tuple, Dict


from data.config import TrainingConfig
import util.suffixes as sfx



class DataManager:
    """Manages file operations for training data, words, and text processing"""

    def __init__(self):
        self.config = TrainingConfig()
        self.words = self._load_words()
        self.suffixes = sfx.ALL_SUFFIXES


    def _load_words(self) -> List[str]:
        """Load words from the dictionary file"""
        try:
            with open(self.config.words_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.config.words_path} not found")
            return []
    
    def _reload_words(self):
        """Reload words from file (useful after deletions)"""
        self.words = self._load_words()

    def load_training_count(self) -> int:
        """Load training count from file"""
        try:
            if os.path.exists(self.config.training_count_file):
                with open(self.config.training_count_file, "r") as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return 0

    def log_decompositions(self, word: str, correct_indices: List[int], 
                          decompositions: List[Tuple]):
        """Save validated decompositions to file"""
        try:
            with open(self.config.valid_decompositions_file, 'a', encoding='utf-8') as f:
                for idx in correct_indices:
                    entry = self._create_log_entry(word, decompositions[idx])
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"📝 Saved {len(correct_indices)} decompositions")
        except Exception as e:
            print(f"⚠️  Could not save: {e}")

    def _create_log_entry(self, word: str, decomp: Tuple) -> Dict:
        """Create a log entry for a decomposition"""
        root, pos, chain, final_pos = decomp
        suffix_info = []
        
        if chain:
            current = root
            for suffix in chain:
                forms = suffix.form(current)
                used_form = forms[0] if forms else ""
                suffix_info.append({
                    'name': suffix.name,
                    'form': used_form,
                    'makes': suffix.makes.name
                })
                current += used_form
        
        return {
            'word': word,
            'root': root,
            'suffixes': suffix_info,
            'final_pos': final_pos
        }

    def get_valid_decomps(self) -> List[Dict]:
        """Load all valid decompositions from log file"""
        entries = []
        try:    
            with open(self.config.valid_decompositions_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"❌ JSON Decode Error on line {i}: {e}. Skipping line.")
                        except Exception as e:
                            print(f"❌ Unexpected Error on line {i}: {e}. Skipping line.")
        except FileNotFoundError:
            print(f"⚠️  No valid decompositions file found")
            return []
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return []
        
        print(f"📖 Loaded {len(entries)} decompositions")
        return entries

    def random_word(self) -> Optional[str]:
        """Get a random word from the dictionary"""
        self._reload_words()
        return random.choice(self.words) if self.words else None
        
    def delete(self, word: str) -> bool:
        """Delete a word from the dictionary file"""
        self._reload_words()
        
        try:
            if word not in self.words:
                return False
            
            self.words.remove(word)
            with open(self.config.words_path, "w", encoding="utf-8") as f:
                for w in self.words:
                    f.write(w + "\n")
            
            return True

        except FileNotFoundError:
            print(f"❌ Error: {self.config.words_path} not found.")
            return False
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            return False

    def exists(self, word: str) -> int:
        """
        Check if word exists in dictionary.
        Returns: 0=no, 1=exists as-is, 2=infinitive exists
        """
        if not word:
            return 0

        infinitive = word + sfx.infinitive_mek.form(word)[0]# word infinitive i suffix ile cekiyoruz
        if infinitive and infinitive in self.words:
            return 2    # better to  return verb first because decomp will still look for nouns.MORE ON DECOMP/
        
        if word in self.words:
            return 1
        


        if word.endswith("l"):

            soft_l_inf = soft_l + sfx.infinitive_mek.form(soft_l)[0]# word infinitive i suffix ile cekiyoruz
            if soft_l_inf and soft_l_inf in self.words:
                return 2                                            # better to  return verb first because decomp will still look for nouns.


            soft_l = word[:-1] + "ł"
            if soft_l in self.words:
                return 1


        return 0

    def delete_word_if_root_exists(self, word: str, correct_indices: List[int], 
                                   decompositions: List[Tuple]) -> bool:
        """
        Check if word should be deleted (root exists in dictionary) and perform deletion.
        Returns True if the word was successfully deleted.
        """
        word_lower = word.lower()
        
        for idx in correct_indices:
            root = decompositions[idx][0].lower()
            
            if root == word_lower:
                continue
            
            if self.exists(root) > 0:
                if self.delete(word_lower):
                    print(f"🗑️  Deleted '{word}' (root '{root}' exists)")
                    return True 
                
                infinitive_form = word_lower + sfx.infinitive_mek.form(word_lower)[0]   # word infinitive i suffix ile cekiyoruz
                if infinitive_form and infinitive_form != word_lower:
                    if self.delete(infinitive_form):
                        print(f"🗑️  Deleted infinitive '{infinitive_form}' for '{word}' (root '{root}' exists)")
                        return True
        
        return False

    def get_text_tokenized(self) -> List[str]:
        """
        Load and tokenize text from sample.txt file.
        Returns a list of words split by whitespace (all lowercase).
        """
        text_path = self.config.sample_text_file
        
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            words = [word.lower() for word in content.split()]
            
            print(f"📄 Loaded {len(words)} words from {text_path}")
            return words
            
        except FileNotFoundError:
            print(f"❌ Error: {text_path} not found")
            return []
        except Exception as e:
            print(f"❌ Error reading text file: {e}")
            return []
    
    def write_decomposed_text(self, text: str):
        """
        Write decomposed text to output file.
        Output file will be named sample_decomposed.txt in data folder.
        """
        output_path = self.config.sample_decomposed_file
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"💾 Decomposed text saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error writing decomposed text: {e}")
            return False
    
    def get_sample_text_path(self) -> str:
        """Get the path to the sample text file"""
        return self.config.sample_text_file
    
    def get_decomposed_text_path(self) -> str:
        """Get the path to the output decomposed text file"""
        return self.config.sample_decomposed_file
    
    def decompose(self, word): ## cheap wrapper to free interactivetrainer from suffix import
        return sfx.decompose(word)