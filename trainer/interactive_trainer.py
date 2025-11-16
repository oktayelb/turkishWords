from typing import List, Optional, Tuple, Dict

from trainer.display import TrainerDisplay
from trainer.data_manager import DataManager
from ml_ranking_model import Ranker, Trainer


class InteractiveTrainer:
    """Handles training logic and model management with pre-encoded suffix data"""
    
    def __init__(self, use_triplet_loss: bool = False, use_lstm: bool = False):
        
        self.data_manager = DataManager()
        self.config = self.data_manager.config
        self.display = TrainerDisplay()
        
        # Create suffix mappings for encoding
        self.suffix_to_id = {suffix.name: idx + 1 for idx, suffix in enumerate(self.data_manager.suffixes)}
        self.id_to_suffix = {idx + 1: suffix.name for idx, suffix in enumerate(self.data_manager.suffixes)}
        self.category_to_id = {'Noun': 0, 'Verb': 1}
        
        self.model = Ranker(
            suffix_vocab_size=len(self.data_manager.suffixes),
            num_categories=2,
            embed_dim=128,
            num_layers=4,
            num_heads=8,
            use_lstm=use_lstm
        )

        self.trainer = Trainer(
            model=self.model,
            model_path=self.config.model_path,
            lr=1e-4,
            batch_size=16,
            use_triplet_loss=use_triplet_loss,
            patience=10
        )
        
        self.training_count = self.data_manager.load_training_count()
        self.auto_stats = {
            'words_processed': 0,
            'words_deleted': 0,
            'words_skipped': 0
        }
    
    def encode_suffix_chain(self, suffix_objects: List) -> List[Tuple[int, int]]:
        """
        Convert suffix objects to (suffix_id, category_id) tuples for ML model.
        
        Args:
            suffix_objects: List of Suffix objects
            
        Returns:
            List of (suffix_id, category_id) tuples
        """
        if not suffix_objects:
            return []
        
        encoded = []
        for suffix_obj in suffix_objects:
            suffix_id = self.suffix_to_id.get(suffix_obj.name, 0)
            category_id = self.category_to_id.get(suffix_obj.makes.name, 0)
            encoded.append((suffix_id, category_id))
        
        return encoded
    


    def save(self):
        """Save model and training count"""
        self.trainer.save_checkpoint(self.config.model_path)
        with open(self.config.training_count_file, "w") as f:
            f.write(str(self.training_count))
        print(f"✅ Model saved")
    
    def _train_on_choices(self, suffix_chains: List[List], 
                         correct_indices: List[int]) -> float:
        """
        Train on user's choices with pre-encoded data.
        
        Args:
            suffix_chains: List of suffix object chains
            correct_indices: Indices of correct decompositions
        """
        # Encode all suffix chains to (suffix_id, category_id) format
        encoded_chains = [self.encode_suffix_chain(chain) for chain in suffix_chains]
        
        # Create training data in new format: (root_info, candidates, correct_idx)
        training_data = [
            ([], encoded_chains, idx) for idx in correct_indices
        ]
        
        return self.trainer.train_epoch(training_data)

    def train_on_word(self, word: str) -> Optional[bool]:
        """
        Train on a single word.
        Returns: True if trained, False if skipped, None if quit
        """
        decompositions = self.data_manager.decompose(word)
        if not decompositions:
            print(f"\n⚠️  No decompositions found for '{word}'")
            return False
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        if len(suffix_chains) == 1:
            print(f"\n✅ Only one decomposition for '{word}' - skipping")
            return False
        
        scores = None
        if self.training_count > 0:
            try:
                # Encode chains for prediction
                encoded_chains = [self.encode_suffix_chain(chain) for chain in suffix_chains]
                _, scores = self.trainer.predict(encoded_chains)
                print(f"\n🤖 ML Model predictions shown")
            except Exception:
                pass
        
        index_mapping = self.display.show_decompositions(word, decompositions, scores)
        choices = self.display.get_user_choices(len(suffix_chains))
        
        if choices is None:
            return None
        if choices == [-1]:
            return False
        
        correct_indices = [index_mapping[c + 1] for c in choices]
        
        self.data_manager.log_decompositions(word, correct_indices, decompositions)
        self.data_manager.delete_word_if_root_exists(word, correct_indices, decompositions)
        
        loss = self._train_on_choices(suffix_chains, correct_indices)
        print(f"\n✅ Training complete. Loss: {loss:.4f}")
        print(f"Total examples: {self.training_count}")
        
        self.training_count += 1
        if self.training_count % self.config.checkpoint_frequency == 0:
            self.save()
        
        return True
    
    def evaluate_word(self, word: str):
        """Evaluate model on a word without training"""
        decompositions = self.data_manager.decompose(word)
        if not decompositions:
            print(f"\n⚠️  No decompositions found")
            return
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        try:
            # Encode chains for prediction
            encoded_chains = [self.encode_suffix_chain(chain) for chain in suffix_chains]
            pred_idx, scores = self.trainer.predict(encoded_chains)
            
            print(f"\n🤖 ML Model's top prediction:")
            self.display._show_single_decomposition(
                word, decompositions[pred_idx], scores[pred_idx], 1, pred_idx
            )
        except Exception as e:
            print(f"\n❌ Error: {e}")

    def batch_train_from_file(self, filepath: Optional[str] = None):
        """Train on all logged decompositions"""
        entries = self.data_manager.get_valid_decomps()
        
        word_groups = {}
        for entry in entries:
            word = entry['word']
            word_groups.setdefault(word, []).append(entry)
        
        training_data = []
        skipped = 0
        
        for word, correct_entries in word_groups.items():
            try:
                decompositions = self.data_manager.decompose(word)
                if not decompositions or len(decompositions) == 1:
                    skipped += 1
                    continue
                
                suffix_chains = [chain for _, _, chain, _ in decompositions]
                
                correct_indices = self._match_decompositions(correct_entries, decompositions)
                if not correct_indices:
                    skipped += 1
                    continue
                
                # Encode chains once per word
                encoded_chains = [self.encode_suffix_chain(chain) for chain in suffix_chains]
                
                # Add multiple training examples (one per correct index)
                for idx in correct_indices:
                    training_data.append(([], encoded_chains, idx))
            
            except Exception:
                skipped += 1
        
        if not training_data:
            print("⚠️  No valid training data")
            return
        
        print(f"\n🚀 Training on {len(training_data)} examples...")
        
        history = self.trainer.train(training_data, num_epochs=5, verbose=True)
        final_loss = history['train_history'][-1] if history['train_history'] else 0.0
        
        self.training_count += len(training_data)
        self.display.show_batch_summary(len(training_data), skipped, final_loss, self.training_count)
        self.save()
    
    def _match_decompositions(self, entries: List[Dict], 
                             decompositions: List[Tuple]) -> List[int]:
        """Match logged entries to decomposition indices"""
        indices = []
        for entry in entries:
            entry_root = entry['root']
            entry_suffixes = [s['name'] for s in entry.get('suffixes', [])]
            
            for idx, (root, _, chain, _) in enumerate(decompositions):
                if root != entry_root:
                    continue
                chain_suffixes = [s.name for s in chain] if chain else []
                if chain_suffixes == entry_suffixes and idx not in indices:
                    indices.append(idx)
                    break
        
        return indices

    def auto_mode(self):
        """Continuously train on random words from dictionary"""
        print(f"💡 Words deleted if root exists in dictionary")
        print(f"   Press 'q' to exit\n")
        
        consecutive_errors = 0
        
        while True:
            try:
                word = self.data_manager.random_word()
                if not word:
                    print("\n✅ No more words!")
                    break
                
                self.auto_stats['words_processed'] += 1
                consecutive_errors = 0
                
                result = self.train_on_word(word)
                
                if result is None:
                    self.auto_stats['words_processed'] -= 1 
                    print("\n👋 Exiting auto mode...")
                    break
                
                if result is False:
                    self.auto_stats['words_skipped'] += 1
                
                word_exists_after = self.data_manager.exists(word.lower())
                if word_exists_after == 0: 
                    self.auto_stats['words_deleted'] += 1
                    
                if self.training_count % self.config.checkpoint_frequency == 0:
                    self.save()
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted!")
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"\n❌ Error: {e}")
                if consecutive_errors >= 5:
                    print("\n❌ Too many errors. Exiting...")
                    break
        
        self.auto_stats['words_processed'] -= self.auto_stats['words_skipped'] + self.auto_stats['words_deleted']
        self.display.show_auto_summary(self.auto_stats)

    def text_mode(self):
        """Process a text file and output morphologically decomposed words"""
        print("\n📄 Text Mode - Processing text file...")
        
        text = self.data_manager.get_text_tokenized()
        
        if not text:
            print("⚠️  No text found or file is empty")
            return
        
        print(f"📊 Processing {len(text)} words...")
        
        decomposed_words = []
        processed_count = 0
        unchanged_count = 0
        error_count = 0
        
        for idx, word in enumerate(text, 1):
            try:
                decompositions = self.data_manager.decompose(word)
                
                if not decompositions:
                    decomposed_words.append(word)
                    unchanged_count += 1
                    continue
                
                if len(decompositions) == 1:
                    decomposed_word = self._format_decomposition(word, decompositions[0], simple=True)
                    decomposed_words.append(decomposed_word)
                    processed_count += 1
                    continue
                
                suffix_chains = [chain for _, _, chain, _ in decompositions]
                
                try:
                    # Encode chains for prediction
                    encoded_chains = [self.encode_suffix_chain(chain) for chain in suffix_chains]
                    pred_idx, scores = self.trainer.predict(encoded_chains)
                    best_decomposition = decompositions[pred_idx]
                    decomposed_word = self._format_decomposition(word, best_decomposition, simple=True)
                    decomposed_words.append(decomposed_word)
                    processed_count += 1
                except Exception:
                    decomposed_word = self._format_decomposition(word, decompositions[0], simple=True)
                    decomposed_words.append(decomposed_word)
                    processed_count += 1
                
                if idx % 100 == 0:
                    print(f"   Progress: {idx}/{len(text)} words...")
            
            except Exception as e:
                decomposed_words.append(word)
                unchanged_count += 1
                error_count += 1
                if error_count <= 5:
                    print(f"   ⚠️  Error processing '{word}': {e}")
        
        output_text = ' '.join(decomposed_words)
        self.data_manager.write_decomposed_text(output_text)
        
        print(f"\n✅ Text processing complete!")
        print(f"   📝 Total words: {len(text)}")
        print(f"   🔄 Decomposed: {processed_count}")
        print(f"   ➡️  Unchanged: {unchanged_count}")
        if error_count > 0:
            print(f"   ⚠️  Errors: {error_count}")
        print(f"   💾 Output saved to: {self.data_manager.get_decomposed_text_path()}")

    def _format_decomposition(self, word: str, decomposition: Tuple, simple: bool = False) -> str:
        """
        Format a decomposition into readable text.
        
        Args:
            word: The original word
            decomposition: Tuple of (root, pos, chain, final_pos)
            simple: If True, only output suffix names. If False, include forms.
        
        Examples:
            simple=True:  ('git', 'verb', [past_tense], 'verb') -> 'git+pastfactative_miş'
            simple=False: ('git', 'verb', [past_tense], 'verb') -> 'git+pastfactative_miş_miş'
        """
        root, pos, chain, final_pos = decomposition
        
        if not chain:
            return root
        
        if simple:
            suffix_names = [suffix.name for suffix in chain]
            result = root + '+' + '+'.join(suffix_names)
            return result
        
        suffix_parts = []
        current_word = root
        
        for suffix_obj in chain:
            forms = suffix_obj.form(current_word)
            used_form = forms[0] if forms else suffix_obj.suffix
            
            suffix_str = f"{suffix_obj.name}_{used_form}"
            suffix_parts.append(suffix_str)
            
            current_word += used_form
        
        result = root + '+' + '+'.join(suffix_parts)
        return result

    def interactive_loop(self):
        """Main interactive training loop"""
        self.display.welcome()
        
        while True:
            try:
                cmd = input("\n📤 Enter word or command: ").strip().lower()
                
                if not cmd:
                    continue
                
                if cmd == 'quit':
                    if self.training_count > 0 and self.display.confirm_save():
                        self.save()
                    print("Goodbye!")
                    break
                elif cmd == 'save':
                    self.save()
                elif cmd == 'stats':
                    self.display.show_statistics(self)
                elif cmd == 'auto':
                    self.auto_mode()
                elif cmd == 'text':
                    self.text_mode()
                elif cmd == 'batch':
                    self.batch_train_from_file()
                elif cmd.startswith('batch '):
                    self.batch_train_from_file(cmd[6:].strip())
                elif cmd.startswith('eval '):
                    self.evaluate_word(cmd[5:].strip())
                else:
                    result = self.train_on_word(cmd)
                    if result is None and self.display.confirm_save():
                        self.save()
                        break
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted!")
                if self.display.confirm_save():
                    self.save()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")