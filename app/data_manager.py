import os
import json
import string
import random
from typing import List, Optional, Dict

from app.file_paths import FilePaths

class DataManager:
    """Manages file operations (I/O only)"""

    def __init__(self):
        self.paths = FilePaths()
        self.words = self._load_words()

    def _load_words(self) -> List[str]:
        """Load words from the dictionary file"""
        try:
            with open(self.paths.words_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.paths.words_path} not found")
            return []
    
    def _reload_words(self):
        self.words = self._load_words()

    def load_training_count(self) -> int:
        try:
            if os.path.exists(self.paths.training_count_path):
                with open(self.paths.training_count_path, "r") as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return 0

    def random_word(self) -> Optional[str]:
        self._reload_words()
        return random.choice(self.words) if self.words else None

    def get_text_tokenized(self) -> List[str]:
        text_path = self.paths.sample_text_path
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("'", "").replace("’", "")
            other_punct = string.punctuation.replace("'", "")
            translator = str.maketrans(other_punct, ' ' * len(other_punct))
            content = content.translate(translator)
            words = [word.lower() for word in content.split()]
            print(f"📄 Loaded {len(words)} words from {text_path}")
            return words
        except FileNotFoundError:
            print(f"❌ Error: {text_path} not found")
            return []
        except Exception as e:
            print(f"❌ Error reading text file: {e}")
            return []
        
    def get_valid_decomps(self) -> List[Dict]:
        entries = []
        try:    
            with open(self.paths.valid_decompositions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
        except FileNotFoundError:
            return []
        return entries

    def log_decompositions(self, log_entries: List[Dict]):
        """
        Pure I/O: Append list of dictionary objects to file.
        Formatting is now done by the caller.
        """
        try:
            with open(self.paths.valid_decompositions_path, 'a', encoding='utf-8') as f:
                for entry in log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"📝 Saved {len(log_entries)} decompositions")
        except Exception as e:
            print(f"⚠️  Could not save: {e}")

    def write_decomposed_text(self, text: str):
        output_path = self.paths.sample_decomposed_path
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"💾 Decomposed text saved to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error writing decomposed text: {e}")
            return False
    
    def delete(self, word: str) -> bool:
        """Delete a word from the dictionary file"""
        self._reload_words()
        
        try:
            if word not in self.words:
                return False
            
            self.words.remove(word)
            with open(self.paths.words_path, "w", encoding="utf-8") as f:
                for w in self.words:
                    f.write(w + "\n")
            return True
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            return False
    def log_sentence_decompositions(self, log_entries: List[Dict], original_sentence: str):
        """
        Logs a full sentence as a single entry in the JSONL file.
        Format: {original_sentence, decomposed_sentence, words: [...]}
        """
        try:
            # Construct the readable decomposed string (e.g., "Ben ev+datif_e git+yor_um")
            # We assume 'morphology_string' is added by the trainer or we fallback to word
            decomposed_str = " ".join([e.get('morphology_string', e['word']) for e in log_entries])
            
            sentence_entry = {
                'type': 'sentence',
                'original_sentence': original_sentence,
                'decomposed_sentence': decomposed_str,
                'words': log_entries
            }
            
            with open(self.paths.valid_decompositions_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sentence_entry, ensure_ascii=False) + '\n')
            print(f"📝 Saved sentence decomposition")
        except Exception as e:
            print(f"⚠️  Could not save sentence: {e}")