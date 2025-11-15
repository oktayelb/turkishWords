from typing import List, Optional, Tuple, Dict

from data.config import TrainingConfig
import util.word_methods as wrd
from ml_ranking_model import ImprovedRanker, ImprovedTrainer, DataAugmenter
import util.suffixes as sfx

from InteractiveTrainer.display import TrainerDisplay
from InteractiveTrainer.file_manager import FileManager


class InteractiveTrainer:
    """Handles training logic and model management"""
    
    def __init__(self, use_triplet_loss: bool = False, use_lstm: bool = False):
        self.config = TrainingConfig()
        self.display = TrainerDisplay()
        self.file_manager = FileManager()
        
        # Initialize model
        self.model = ImprovedRanker(
            embed_dim=128,
            num_layers=4,
            num_heads=8,
            use_lstm=use_lstm
        )
        
        self.trainer = ImprovedTrainer(
            model=self.model,
            lr=1e-4,
            batch_size=16,
            use_triplet_loss=use_triplet_loss,
            patience=10
        )
        
        self.training_count = self.file_manager.load_training_count()
        self.auto_stats = {
            'words_processed': 0,
            'words_deleted': 0,
            'words_skipped': 0
        }
    
    # ==================== CORE DECOMPOSITION ====================
    
    def decompose(self, word: str) -> List[Tuple]:
        """Find all possible legal decompositions of a word."""
        if not word:
            return []
        
        analyses = []
        for i in range(1, len(word) + 1):
            root = word[:i]
            #if not wrd.exists(root):
            #    continue

            pos = "verb" #if wrd.can_be_verb(root) else "noun"

            # Try both POS routes if it's potentially a verb
            chains = (sfx.find_suffix_chain(word, "verb", root) +
                    sfx.find_suffix_chain(word, "noun", root)) if pos == "verb" \
                    else sfx.find_suffix_chain(word, "noun", root)

            for chain, final_pos in chains:
                analyses.append((root, pos, chain, final_pos))
        
        return analyses
    
    # ==================== MODEL OPERATIONS ====================

    def save(self):
        """Save model and training count"""
        self.trainer.save_checkpoint(self.config.model_path)
        with open(self.config.training_count_file, "w") as f:
            f.write(str(self.training_count))
        print(f"✅ Model saved")
    
    def _train_on_choices(self, root: str, suffix_chains: List[List], 
                         correct_indices: List[int]) -> float:
        """Train on user's choices"""
        training_data = [(root, suffix_chains, idx) for idx in correct_indices]
        return self.trainer.train_epoch(training_data)
    
    # ==================== SINGLE WORD TRAINING ====================
    
    def train_on_word(self, word: str) -> Optional[bool]:
        """
        Train on a single word.
        Returns: True if trained, False if skipped, None if quit
        """
        # Get decompositions
        decompositions = self.decompose(word)
        if not decompositions:
            print(f"\n⚠️  No decompositions found for '{word}'")
            return False
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        if len(suffix_chains) == 1:
            print(f"\n✅ Only one decomposition for '{word}' - skipping")
            return False
        
        # Get predictions
        root = decompositions[0][0]
        scores = None
        if self.training_count > 0:
            try:
                _, scores = self.trainer.predict(root, suffix_chains)
                print(f"\n🤖 ML Model predictions shown")
            except Exception:
                pass
        
        # Display and get user choice
        index_mapping = self.display.show_decompositions(word, decompositions, scores)
        choices = self.display.get_user_choices(len(suffix_chains))
        
        if choices is None:  # Quit
            return None
        if choices == [-1]:  # Skip
            return False
        
        # Map display choices to original indices
        correct_indices = [index_mapping[c + 1] for c in choices]
        
        # Log valid decompositions
        self.file_manager.log_decompositions(word, correct_indices, decompositions)
        
        # Delete word if its root exists in dictionary
        self.file_manager.delete_word_if_root_exists(word, correct_indices, decompositions)
        
        # Train the model
        loss = self._train_on_choices(root, suffix_chains, correct_indices)
        print(f"\n✅ Training complete. Loss: {loss:.4f}")
        print(f"Total examples: {self.training_count}")
        
        # Periodic save
        self.training_count += 1
        if self.training_count % self.config.checkpoint_frequency == 0:
            self.save()
        
        return True
    
    def evaluate_word(self, word: str):
        """Evaluate model on a word without training"""
        decompositions = self.decompose(word)
        if not decompositions:
            print(f"\n⚠️  No decompositions found")
            return
        
        root = decompositions[0][0]
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        try:
            pred_idx, scores = self.trainer.predict(root, suffix_chains)
            print(f"\n🤖 ML Model's top prediction:")
            self.display._show_single_decomposition(
                word, decompositions[pred_idx], scores[pred_idx], 1, pred_idx
            )
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    # ==================== BATCH TRAINING ====================
    
    def batch_train_from_file(self, filepath: Optional[str] = None):
        """Train on all logged decompositions"""
        # Load entries
        entries = self.file_manager.get_valid_decomps()
        
        # Group by word and prepare training data
        word_groups = {}
        for entry in entries:
            word = entry['word']
            word_groups.setdefault(word, []).append(entry)
        
        training_data = []
        skipped = 0
        
        for word, correct_entries in word_groups.items():
            try:
                decompositions = self.decompose(word)
                if not decompositions or len(decompositions) == 1:
                    skipped += 1
                    continue
                
                root = decompositions[0][0]
                suffix_chains = [chain for _, _, chain, _ in decompositions]
                
                # Match logged entries to decompositions
                correct_indices = self._match_decompositions(correct_entries, decompositions)
                if not correct_indices:
                    skipped += 1
                    continue
                
                # Add to training data
                for idx in correct_indices:
                    training_data.append((root, suffix_chains, idx))
            
            except Exception:
                skipped += 1
        
        if not training_data:
            print("⚠️  No valid training data")
            return
        
        print(f"\n🚀 Training on {len(training_data)} examples...")
        
        # Train
        history = self.trainer.train(training_data, num_epochs=5, verbose=True)
        final_loss = history['training_history'][-1] if history['training_history'] else 0.0
        
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
    
    # ==================== AUTO MODE ====================
    
    def auto_mode(self):
        """Continuously train on random words from dictionary"""
        print(f"💡 Words deleted if root exists in dictionary")
        print(f"   Press 'q' to exit\n")
        
        consecutive_errors = 0
        
        while True:
            try:
                word = self.file_manager.random_word()
                if not word:
                    print("\n✅ No more words!")
                    break
                
                self.auto_stats['words_processed'] += 1
                consecutive_errors = 0
                
                result = self.train_on_word(word)
                
                if result is None:  # Quit
                    self.auto_stats['words_processed'] -= 1 
                    print("\n👋 Exiting auto mode...")
                    break
                
                if result is False:  # Skipped
                    self.auto_stats['words_skipped'] += 1
                
                # Check for deletion
                if wrd.exists(word.lower()) == 0: 
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
    
    # ==================== TEXT MODE ====================
    
    def text_mode(self):
        """Process a text file and output morphologically decomposed words"""
        print("\n📄 Text Mode - Processing text file...")
        
        # Initialize list with tokenized text
        text = self.file_manager.get_text_tokenized()
        
        if not text:
            print("⚠️  No text found or file is empty")
            return
        
        print(f"📊 Processing {len(text)} words...")
        
        # Store decomposed results
        decomposed_words = []
        processed_count = 0
        unchanged_count = 0
        error_count = 0
        
        for idx, word in enumerate(text, 1):
            try:
                # Get all possible decompositions for this word
                decompositions = self.decompose(word)
                
                # If no decompositions found, keep word as-is
                if not decompositions:
                    decomposed_words.append(word)
                    unchanged_count += 1
                    continue
                
                # If only one decomposition, use it
                if len(decompositions) == 1:
                    decomposed_word = self._format_decomposition(word, decompositions[0], simple=True)
                    decomposed_words.append(decomposed_word)
                    processed_count += 1
                    continue
                
                # Multiple decompositions - use ML prediction to get the best one
                root = decompositions[0][0]
                suffix_chains = [chain for _, _, chain, _ in decompositions]
                
                try:
                    # Get ML model prediction
                    pred_idx, scores = self.trainer.predict(root, suffix_chains)
                    best_decomposition = decompositions[pred_idx]
                    decomposed_word = self._format_decomposition(word, best_decomposition, simple=True)
                    decomposed_words.append(decomposed_word)
                    processed_count += 1
                except Exception:
                    # If prediction fails, use first decomposition
                    decomposed_word = self._format_decomposition(word, decompositions[0], simple=True)
                    decomposed_words.append(decomposed_word)
                    processed_count += 1
                
                # Show progress every 100 words
                if idx % 100 == 0:
                    print(f"   Progress: {idx}/{len(text)} words...")
            
            except Exception as e:
                # If any error occurs, keep word unchanged
                decomposed_words.append(word)
                unchanged_count += 1
                error_count += 1
                if error_count <= 5:  # Only show first 5 errors
                    print(f"   ⚠️  Error processing '{word}': {e}")
        
        # Write results to output file
        output_text = ' '.join(decomposed_words)
        self.file_manager.write_decomposed_text(output_text)
        
        # Show summary
        print(f"\n✅ Text processing complete!")
        print(f"   📝 Total words: {len(text)}")
        print(f"   🔄 Decomposed: {processed_count}")
        print(f"   ➡️  Unchanged: {unchanged_count}")
        if error_count > 0:
            print(f"   ⚠️  Errors: {error_count}")
        print(f"   💾 Output saved to: {self.file_manager.get_decomposed_text_path()}")

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
            # Simple format: just suffix names
            suffix_names = [suffix.name for suffix in chain]
            result = root + '+' + '+'.join(suffix_names)
            return result
        
        # Detailed format: suffix names + actual forms used
        suffix_parts = []
        current_word = root
        
        for suffix_obj in chain:
            # Get the actual form that this suffix takes after the current base
            forms = suffix_obj.form(current_word)
            used_form = forms[0] if forms else suffix_obj.suffix
            
            # Format: suffix_name + underscore + used_form
            suffix_str = f"{suffix_obj.name}_{used_form}"
            suffix_parts.append(suffix_str)
            
            # Update current base for next suffix
            current_word += used_form
        
        # Combine: root + '+' + suffixes
        result = root + '+' + '+'.join(suffix_parts)
        return result
    # ==================== INTERACTIVE LOOP ====================
    
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