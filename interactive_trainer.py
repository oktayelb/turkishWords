"""
Refactored interactive training pipeline for Turkish morphological analyzer
Split into two focused classes: InteractiveTrainer (logic) and TrainerDisplay (UI)
"""

import os
import json
from typing import List, Optional, Tuple, Dict

from data.config import TrainingConfig
from util.decomposition import decompose
import util.word_methods as wrd
from ml_ranking_model import ImprovedRanker, ImprovedTrainer, DataAugmenter


class TrainerDisplay:
    """Handles all user interaction and display formatting"""
    
    @staticmethod
    def welcome():
        """Show welcome message and available commands"""
        print("\n💡 Commands:")
        print("  - Enter a word to analyze and train")
        print("  - 'auto' - Start auto mode (random words from dictionary)")
        print("  - 'eval <word>' - Evaluate model on a word")
        print("  - 'batch' - Train on all logged decompositions")
        print("  - 'stats' - Show training statistics")
        print("  - 'save' - Save model")
        print("  - 'quit' - Exit")
    
    @staticmethod
    def show_config(mode: str, use_lstm: bool):
        """Display model configuration"""
        print("\n" + "=" * 60)
        print(f"Configuration:")
        print(f"  - Loss: {mode}")
        print(f"  - Architecture: {'LSTM' if use_lstm else 'Transformer'}")
        print("=" * 60 + "\n")
    
    @staticmethod
    def show_decompositions(word: str, decompositions: List[Tuple], 
                           scores: Optional[List[float]] = None) -> Dict[int, int]:
        """
        Display all decompositions sorted by score.
        Returns mapping from display index to original index.
        """
        print(f"\n{word}")
        
        # Sort by score (lower is better)
        if scores:
            indexed = list(enumerate(zip(decompositions, scores)))
            indexed.sort(key=lambda x: x[1][1])
        else:
            indexed = [(i, (d, None)) for i, d in enumerate(decompositions)]
        
        # Display each option
        index_mapping = {}
        for display_idx, (orig_idx, (decomp, score)) in enumerate(indexed, 1):
            TrainerDisplay._show_single_decomposition(
                word, decomp, score, display_idx, orig_idx
            )
            index_mapping[display_idx] = orig_idx
        
        return index_mapping
    
    @staticmethod
    def _show_single_decomposition(word: str, decomp: Tuple, score: Optional[float],
                                   display_idx: int, orig_idx: int):
        """Format and print a single decomposition"""
        root, pos, chain, final_pos = decomp
        
        print(f"\n[Option {display_idx}] (Original index: {orig_idx + 1})")
        if score is not None:
            print(f"ML Score: {score:.4f}")
        
        print(f"Root:      {root} ({pos})")
        
        if chain:
            suffix_forms, suffix_names, formation = TrainerDisplay._build_suffix_chain(
                word, root, pos, chain
            )
            print(f"Suffixes:  {' + '.join(suffix_forms)}")
            print(f"Names:     {' + '.join(suffix_names)}")
            print(f"Formation: {' → '.join(formation)}")
        else:
            verb_marker = "-" if pos == "verb" else ""
            print(f"Formation: {root}{verb_marker} (no suffixes)")
        
        print(f"Final POS: {final_pos}")
        print("-" * 70)
    
    @staticmethod
    def _build_suffix_chain(word: str, root: str, pos: str, 
                           chain: List) -> Tuple[List[str], List[str], List[str]]:
        """Build suffix forms, names, and formation steps"""
        current_stem = root
        suffix_forms = []
        suffix_names = []
        formation = [root + ("-" if pos == "verb" else "")]
        cursor = len(root)
        
        for suffix_obj in chain:
            # Find matching suffix form
            found_form = None
            for form in suffix_obj.form(current_stem):
                if word.startswith(form, cursor):
                    found_form = form
                    break
            
            if not found_form:
                print(f"[Warning: Could not match suffix '{suffix_obj.name}']")
                break
            
            suffix_forms.append(found_form)
            suffix_names.append(suffix_obj.name)
            current_stem += found_form
            cursor += len(found_form)
            
            # Add formation step
            verb_marker = "-" if suffix_obj.makes.name == "Verb" else ""
            formation.append(current_stem + verb_marker)
        
        return suffix_forms, suffix_names, formation
    
    @staticmethod
    def get_user_choices(num_options: int) -> Optional[List[int]]:
        """Get user's choice of correct decomposition(s)"""
        while True:
            choice = input(
                f"\nSelect correct (1-{num_options}, 's'=skip, 'q'=quit): "
            ).strip().lower()
            
            if choice == 'q':
                return None
            if choice == 's':
                return [-1]
            
            parsed = TrainerDisplay._parse_numbers(choice, num_options)
            if parsed is not None:
                return [c - 1 for c in parsed]  # Convert to 0-indexed
    
    @staticmethod
    def _parse_numbers(input_str: str, max_value: int) -> Optional[List[int]]:
        """Parse comma/space separated numbers"""
        normalized = input_str.replace(',', ' ').replace('-', ' ').replace('/', ' ')
        parts = normalized.split()
        
        if not parts:
            print("No input provided.")
            return None
        
        try:
            choices = [int(p) for p in parts]
        except ValueError:
            print("Invalid number in input.")
            return None
        
        invalid = [c for c in choices if c < 1 or c > max_value]
        if invalid:
            print(f"Invalid choices: {invalid}. Use 1-{max_value}")
            return None
        
        return list(dict.fromkeys(choices))  # Remove duplicates
    
    @staticmethod
    def confirm_save() -> bool:
        """Ask if user wants to save before quitting"""
        return input("Save model before quitting? (y/n): ").strip().lower() == 'y'
    
    @staticmethod
    def get_training_mode() -> str:
        """Ask user to choose training mode"""
        print("\nSelect training mode:")
        print("  1. Contrastive learning (default)")
        print("  2. Triplet loss (better margins)")
        choice = input("Enter 1 or 2 (default: 1): ").strip()
        return 'triplet' if choice == '2' else 'contrastive'
    
    @staticmethod
    def get_architecture() -> bool:
        """Ask user to choose architecture (returns True for LSTM)"""
        print("\nSelect model architecture:")
        print("  1. Transformer (more accurate, slower)")
        print("  2. LSTM (faster, good accuracy)")
        choice = input("Enter 1 or 2 (default: 1): ").strip()
        return choice == '2'
    
    @staticmethod
    def show_statistics(trainer: 'InteractiveTrainer'):
        """Display comprehensive training statistics"""
        print(f"\n📊 Training Statistics:")
        print(f"  Total examples: {trainer.training_count}")
        
        if trainer.trainer.training_history:
            recent = trainer.trainer.training_history[-20:]
            print(f"  Recent avg loss: {sum(recent)/len(recent):.4f}")
            print(f"  Latest loss: {trainer.trainer.training_history[-1]:.4f}")
        
        if trainer.trainer.validation_history:
            print(f"  Best validation: {trainer.trainer.best_val_loss:.4f}")
        
        if trainer.auto_stats['words_processed'] > 0:
            print(f"\n  Auto mode:")
            print(f"    - Processed: {trainer.auto_stats['words_processed']}")
            print(f"    - Deleted: {trainer.auto_stats['words_deleted']}")
            print(f"    - Skipped: {trainer.auto_stats['words_skipped']}")
        
        print(f"\n  Model config:")
        print(f"    - Architecture: {'LSTM' if trainer.model.use_lstm else 'Transformer'}")
        print(f"    - Loss: {'Triplet' if trainer.trainer.use_triplet_loss else 'Contrastive'}")
        print(f"    - Batch size: {trainer.trainer.batch_size}")
    
    @staticmethod
    def show_auto_summary(stats: Dict):
        """Show auto mode summary"""
        print(f"\n{'='*70}")
        print(f"AUTO MODE SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Processed: {stats['words_processed']}")
        print(f"🗑️  Deleted: {stats['words_deleted']}")
        print(f"⭐️  Skipped: {stats['words_skipped']}")
        print(f"{'='*70}\n")
    
    @staticmethod
    def show_batch_summary(trained: int, skipped: int, final_loss: float, total: int):
        """Show batch training summary"""
        print(f"\n{'='*70}")
        print(f"BATCH TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Trained: {trained} examples")
        print(f"⭐️  Skipped: {skipped} words")
        print(f"📊 Final loss: {final_loss:.4f}")
        print(f"📈 Total examples: {total}")
        print(f"{'='*70}\n")


class InteractiveTrainer:
    """Handles training logic and model management"""
    
    def __init__(self, use_triplet_loss: bool = False, use_lstm: bool = False):
        self.config = TrainingConfig()
        self.display = TrainerDisplay()
        
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
        
        self.training_count = self._load_training_count()
        self._load_checkpoint()
        
        self.auto_stats = {
            'words_processed': 0,
            'words_deleted': 0,
            'words_skipped': 0
        }
    
    def _load_checkpoint(self):
        """Load existing model if available"""
        if os.path.exists(self.config.model_path):
            try:
                self.trainer.load_checkpoint(self.config.model_path)
                print(f"✅ Loaded model from {self.config.model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
        else:
            print("Starting with fresh model")
    
    def _load_training_count(self) -> int:
        """Load training count from file"""
        try:
            if os.path.exists(self.config.training_count_file):
                with open(self.config.training_count_file, "r") as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return 0
    
    def save(self):
        """Save model and training count"""
        self.trainer.save_checkpoint(self.config.model_path)
        with open(self.config.training_count_file, "w") as f:
            f.write(str(self.training_count))
        print(f"✅ Model saved")
    
    def train_on_word(self, word: str) -> Optional[bool]:
        """
        Train on a single word.
        Returns: True if trained, False if skipped, None if quit
        """
        # Get decompositions
        decompositions = decompose(word)
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
        self._log_decompositions(word, correct_indices, decompositions)
        
        # Delete word if its root exists in dictionary
        if self._delete_word_if_root_exists(word, correct_indices, decompositions):
            # If deleted, we don't want to process the word further in auto_mode stats
            # This logic is handled by the caller (auto_mode)
            pass 
        
        # Train the model
        loss = self._train_on_choices(root, suffix_chains, correct_indices)
        print(f"\n✅ Training complete. Loss: {loss:.4f}")
        print(f"Total examples: {self.training_count}")
        
        # Periodic save
        self.training_count += 1
        if self.training_count % self.config.checkpoint_frequency == 0:
            self.save()
        
        return True
    
    def _train_on_choices(self, root: str, suffix_chains: List[List], 
                         correct_indices: List[int]) -> float:
        """Train on user's choices"""
        training_data = [(root, suffix_chains, idx) for idx in correct_indices]
        return self.trainer.train_epoch(training_data)
    
    def _log_decompositions(self, word: str, correct_indices: List[int], 
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
    
    def _delete_word_if_root_exists(self, word: str, correct_indices: List[int], 
                           decompositions: List[Tuple]) -> bool:
        """
        Check if word should be deleted (root exists in dictionary) and perform deletion.
        Returns True if the word was successfully deleted.
        """
        word_lower = word.lower()
        
        for idx in correct_indices:
            root = decompositions[idx][0].lower()
            
            if root == word_lower:  # No suffixes
                continue
            
            # wrd.exists returns: 0=no, 1=infinitive, 2=non-infinitive
            if wrd.exists(root) > 0:
                # The word to be deleted is the word itself if its root is a non-infinitive form,
                # otherwise delete the infinitive form.
                ## bruada hata var ex'sts sira onceigi veriyo noun a Bunu duzelt.
                to_be_deleted = word_lower if wrd.exists(word_lower) == 1 else wrd.infinitive(word_lower)
                
                if to_be_deleted and wrd.delete(to_be_deleted):
                    print(f"🗑️  Deleted '{word}' (root '{root}' exists)")
                    return True 
                else:
                    print(f"Could not delete word '{to_be_deleted or word_lower}'")
        
        return False
    
    def evaluate_word(self, word: str):
        """Evaluate model on a word without training"""
        decompositions = decompose(word)
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
    
    def batch_train_from_file(self, filepath: Optional[str] = None):
        """Train on all logged decompositions"""
        # ... (lines before remain the same) ...
        
        # Load entries
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
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return
        
        if not entries:
            print("⚠️  No entries found")
            return
        
        print(f"📖 Loaded {len(entries)} decompositions")
        
        # Group by word and prepare training data
        word_groups = {}
        for entry in entries:
            word = entry['word']
            word_groups.setdefault(word, []).append(entry)
        
        training_data = []
        skipped = 0
        
        for word, correct_entries in word_groups.items():
            try:
                decompositions = decompose(word)
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
    
    def auto_mode(self):
        """Continuously train on random words from dictionary"""
        print(f"💡 Words deleted if root exists in dictionary")
        print(f"   Press 'q' to exit\n")
        
        consecutive_errors = 0
        
        while True:
            try:
                word = wrd.random_word()
                if not word:
                    print("\n✅ No more words!")
                    break
                
                self.auto_stats['words_processed'] += 1
                consecutive_errors = 0
                
                result = self.train_on_word(word)
                
                if result is None:  # Quit
                    # Already counted as processed, but not trained/skipped
                    self.auto_stats['words_processed'] -= 1 
                    print("\n👋 Exiting auto mode...")
                    break
                
                if result is False: # Skipped
                    # If train_on_word skipped, it means no decomposition or only one.
                    # If it was deleted, result would have been True (training happened).
                    # This is for words that could not be processed due to decomposition issues.
                    self.auto_stats['words_skipped'] += 1
                
                # Check for deletion by seeing if the word still exists in the dictionary.
                # If the word was deleted by _delete_word_if_root_exists, it's counted here.
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
        
        # Processed words count should only include successfully checked words
        self.auto_stats['words_processed'] -= self.auto_stats['words_skipped'] + self.auto_stats['words_deleted']
        self.display.show_auto_summary(self.auto_stats)
    
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


def main():
    """Entry point"""
    display = TrainerDisplay()
    mode = display.get_training_mode()
    use_lstm = display.get_architecture()
    display.show_config(mode, use_lstm)
    
    trainer = InteractiveTrainer(
        use_triplet_loss=(mode == 'triplet'),
        use_lstm=use_lstm
    )
    trainer.interactive_loop()


if __name__ == "__main__":
    main()