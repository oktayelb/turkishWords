"""
Interactive training pipeline for Turkish morphological analyzer
Updated to work with improved ML ranking model
Added AUTO MODE for random word training
"""

import os
import json

from typing import List, Optional, Tuple, Dict


# in project imports
from data.config import TrainingConfig
from util.decomposition import decompose
import  util.print as display 
import util.word_methods as wrd

from ml_ranking_model import (
    ImprovedRanker, 
    ImprovedTrainer,
    DataAugmenter
)

class UserInputHandler:
    """Handles user input and validation"""
    
    @staticmethod
    def get_correct_choices(num_options: int) -> Optional[List[int]]:
        """
        Get user's choice(s) of correct decomposition(s)
        Returns: List of 0-indexed choices, [-1] for skip, or None for quit
        """
        while True:
            try:
                choice = input(
                    f"\nSelect correct decomposition(s) (1-{num_options}, 's' to skip, 'q' to quit): "
                ).strip().lower()
                
                if choice == 'q':
                    return None
                elif choice == 's':
                    return [-1]
                
                # Parse multiple numbers
                choices = UserInputHandler._parse_number_input(choice, num_options)
                
                if choices is not None:
                    return [c - 1 for c in choices]  # Convert to 0-indexed
                    
            except Exception as e:
                print(f"Error parsing input: {e}. Please try again.")
    
    @staticmethod
    def _parse_number_input(input_str: str, max_value: int) -> Optional[List[int]]:
        """Parse comma/space separated numbers from input string"""
        # Replace common separators with spaces
        normalized = input_str.replace(',', ' ').replace('-', ' ').replace('/', ' ').replace(';', ' ')
        parts = [p.strip() for p in normalized.split() if p.strip()]
        
        if not parts:
            print("No input provided. Please try again.")
            return None
        
        # Convert to integers
        try:
            choices = [int(part) for part in parts]
        except ValueError as e:
            print(f"Invalid number in input: {e}")
            return None
        
        # Validate range
        invalid = [c for c in choices if c < 1 or c > max_value]
        if invalid:
            print(f"Invalid choice(s): {invalid}. Please enter numbers between 1 and {max_value}")
            return None
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(choices))
    
    @staticmethod
    def confirm_save() -> bool:
        """Ask user if they want to save before quitting"""
        choice = input("Save model before quitting? (y/n): ").strip().lower()
        return choice == 'y'
    
    @staticmethod
    def get_training_mode() -> str:
        """Ask user which training mode to use"""
        print("\nSelect training mode:")
        print("  1. Contrastive learning (default, good for single correct answer)")
        print("  2. Triplet loss (better margins, good for ranking)")
        choice = input("Enter 1 or 2 (default: 1): ").strip()
        return 'triplet' if choice == '2' else 'contrastive'
    
    @staticmethod
    def get_arch_mode():
        print("\nSelect model architecture:")
        print("  1. Transformer (more accurate, slower)")
        print("  2. LSTM (faster, good accuracy)")
        arch_choice = input("Enter 1 or 2 (default: 1): ").strip()
        return  (arch_choice == '2')



class DecompositionLogger:
    """Handles logging of valid decompositions"""
    
    filepath = TrainingConfig.valid_decompositions_file
    
    def log_decompositions(self, word: str, correct_indices: List[int], 
                          decompositions: List[Tuple]) -> None:
        """Save validated decomposition(s) to file in JSONL format"""
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                for idx in correct_indices:
                    entry = self._create_entry(word, decompositions[idx])
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"📝 Saved {len(correct_indices)} valid decomposition(s) to {self.filepath}")
        
        except Exception as e:
            print(f"⚠️  Could not save valid decomposition: {e}")
    
    def _create_entry(self, word: str, decomp: Tuple) -> Dict:
        """Create a log entry for a decomposition"""
        root, pos, chain, final_pos = decomp
        
        suffix_info = []
        if chain:
            current_word = root
            for suffix_obj in chain:
                suffix_form = suffix_obj.form(current_word)
                suffix_info.append({
                    'name': suffix_obj.name,
                    'form': suffix_form,
                    'makes': suffix_obj.makes.name
                })
                current_word += suffix_form
        
        return {
            'word': word,
            'root': root,
            'suffixes': suffix_info,
            'final_pos': final_pos
        }
    
    def count_entries(self) -> int:
        """Count total logged entries"""
        if not os.path.exists(self.filepath):
            return 0
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0


class InteractiveTrainer:
    """Main interactive training interface with improved ML model"""
    
    def __init__(self, use_triplet_loss: bool = False, use_lstm: bool = False):
        """
        Initialize interactive trainer with improved model.
        
        Args:
            use_triplet_loss: Use triplet loss instead of contrastive
            use_lstm: Use LSTM encoder instead of Transformer (faster)
        """
        self.config = TrainingConfig()
        
        # Create improved model
        self.model = ImprovedRanker(
            embed_dim=128,
            num_layers=4,
            num_heads=8,
            use_lstm=use_lstm
        )
        
        # Create improved trainer with batching
        self.trainer = ImprovedTrainer(
            model=self.model,
            lr=1e-4,
            batch_size=16,  # Smaller batch for interactive mode
            use_triplet_loss=use_triplet_loss,
            patience=10
        )
        
        self.logger = DecompositionLogger()
        self.input_handler = UserInputHandler()
        self.augmenter = DataAugmenter()

        
        # Load state
        self.training_count = self._load_training_count()
        self._load_checkpoint_if_exists()
        
        # Track training examples for batch training
        self.pending_examples: List[Tuple[str, List[List], int]] = []
        
        # Auto mode statistics
        self.auto_mode_stats = {
            'words_processed': 0,
            'words_deleted': 0,
            'words_skipped': 0
        }
    
    def _load_checkpoint_if_exists(self) -> None:
        """Load existing model checkpoint if available"""
        if os.path.exists(self.config.model_path):
            try:
                self.trainer.load_checkpoint(self.config.model_path)
                print(f"✅ Loaded existing model from {self.config.model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print("Starting with fresh model")
        else:
            print("Starting with fresh model")
    
    def _load_training_count(self) -> int:
        """Load training count from file"""
        try:
            if os.path.exists(self.config.training_count_file):
                with open(self.config.training_count_file, "r") as f:
                    return int(f.read().strip())
        except Exception as e:
            print(f"⚠️  Could not read training_count.txt: {e}")
        return 0
    
    def _save_training_count(self) -> None:
        """Save current training count to file"""
        try:
            with open(self.config.training_count_file, "w") as f:
                f.write(str(self.training_count))
        except Exception as e:
            print(f"⚠️  Could not save training_count.txt: {e}")
    
    def _match_decompositions(self, correct_entries: List[Dict], 
                             decompositions: List[Tuple]) -> List[int]:
        """
        Match logged decompositions to actual decomposition tuples
        Returns list of indices in decompositions that match correct_entries
        """
        correct_indices = []
        
        for entry in correct_entries:
            entry_root = entry['root']
            entry_suffix_names = [s['name'] for s in entry.get('suffixes', [])]
            
            # Find matching decomposition
            for idx, decomp in enumerate(decompositions):
                root, pos, chain, final_pos = decomp
                
                if root != entry_root:
                    continue
                
                chain_suffix_names = [s.name for s in chain] if chain else []
                
                if chain_suffix_names == entry_suffix_names:
                    if idx not in correct_indices:
                        correct_indices.append(idx)
                    break
        
        return correct_indices
   
    def save(self) -> None:
        """Save model and training count"""
        self.trainer.save_checkpoint(self.config.model_path)
        self._save_training_count()
        print(f"✅ Model and training count saved")
    
    def _add_to_batch(self, root: str, suffix_chains: List[List], correct_idx: int) -> None:
        """Add training example to pending batch"""
        self.pending_examples.append((root, suffix_chains, correct_idx))
    
    def _train_batch(self) -> Optional[float]:
        """Train on accumulated batch of examples"""
        if not self.pending_examples:
            return None
        
        # Train one epoch on the batch
        avg_loss = self.trainer.train_epoch(self.pending_examples)
        
        # Clear pending examples
        num_examples = len(self.pending_examples)
        self.pending_examples = []
        
        return avg_loss
    
    def _train_on_single_example(self, root: str, suffix_chains: List[List], 
                                 correct_idx: int) -> float:
        """
        Train on a single example immediately (for interactive feedback).
        Uses the improved trainer's batch processing with batch_size=1.
        """
        # Create a mini-batch with just this example
        training_data = [(root, suffix_chains, correct_idx)]
        
        # Train one epoch on this single example
        loss = self.trainer.train_epoch(training_data)
        
        return loss
    
    def _train_on_choices(self, root: str, suffix_chains: List[List], 
                         correct_indices: List[int], interactive: bool = True) -> float:
        """
        Train model on user's choices.
        
        Args:
            root: Word root
            suffix_chains: All candidate suffix chains
            correct_indices: Indices of correct decompositions
            interactive: If True, train immediately for feedback; if False, add to batch
        
        Returns:
            Loss value (or 0 if added to batch)
        """
        if len(correct_indices) == 1:
            # Single correct answer
            correct_idx = correct_indices[0]
            
            if interactive:
                # Train immediately for user feedback
                return self._train_on_single_example(root, suffix_chains, correct_idx)
            else:
                # Add to batch for later training
                self._add_to_batch(root, suffix_chains, correct_idx)
                return 0.0
        
        else:
            # Multiple correct answers - train on each as separate example
            total_loss = 0.0
            
            for correct_idx in correct_indices:
                if interactive:
                    loss = self._train_on_single_example(root, suffix_chains, correct_idx)
                    total_loss += loss
                else:
                    self._add_to_batch(root, suffix_chains, correct_idx)
            
            return total_loss / len(correct_indices) if interactive else 0.0
    
    def _check_and_delete_word(self, word: str, correct_indices: List[int], 
                               decompositions: List[Tuple]) -> bool:
        """
        Check if the word should be deleted from words.txt based on chosen decompositions.
        Delete if ANY of the chosen roots exist in the dictionary.
        
        Returns: True if word was deleted, False otherwise
        """
        word_lower = word.lower()
        
        # Check each correct decomposition
        for idx in correct_indices:
            root, pos, chain, final_pos = decompositions[idx]
            root_lower = root.lower()   
            
            # Skip if root is the same as the word (no suffixes)
            if root_lower == word_lower:
                continue
            
            # Check if root exists in dictionary
            if wrd.exists(root_lower):
                # Delete the word from dictionary

                deleted_elemet = word_lower if wrd.exists(root_lower) == 1 else wrd.infinitive(word_lower)

                if wrd.delete(deleted_elemet):
                    print(f"🗑️  Deleted '{word}' from words.txt (root '{root}' exists in dictionary)")
                    return True
                else:
                    print(f"⚠️  Could not delete '{word}' from words.txt")
                    return False
        
        return False
    
    def train_on_word(self, word: str, auto_mode: bool = False) -> Optional[bool]:
        """
        Interactive training on a single word
        
        Args:
            word: Word to analyze
            auto_mode: If True, handle auto mode logic (deletion, etc.)
        
        Returns: True if trained, False if skipped, None if quit
        """
        # Get decompositions
        suffix_chains = []
        decompositions = decompose(word)
        
        # Extract root from first decomposition for feature extraction
        root = decompositions[0][0] if decompositions else word

        for root_d, pos, suffix_chain, final_pos in decompositions:
            # suffix_chain already contains Suffix objects
            suffix_chains.append(suffix_chain)
        
        # Handle edge cases
        if not suffix_chains:
            print(f"\n⚠️  No valid decompositions found for '{word}'")
            if auto_mode:
                self.auto_mode_stats['words_skipped'] += 1
            return False
        
        if len(suffix_chains) == 1:
            print(f"\n✅ Only one decomposition exists for '{word}' - no training needed")
            if auto_mode:
                self.auto_mode_stats['words_skipped'] += 1
            return False
        
        # Get ML predictions if model has been trained
        scores = None
        if self.training_count > 0:
            try:
                predicted_idx, scores = self.trainer.predict(root, suffix_chains)
                print(f"\n🤖 ML Model predicts option with highest score")
            except Exception as e:
                print(f"\n⚠️  Could not get model predictions: {e}")
                scores = None
        
        # Display options and get user choice
        index_mapping = display.DecompositionDisplay.display_all(word, decompositions, scores)
        choices = self.input_handler.get_correct_choices(len(suffix_chains))
        
        if choices is None:  # Quit
            return None
        elif choices == [-1]:  # Skip
            if auto_mode:
                self.auto_mode_stats['words_skipped'] += 1
            return False
        
        # Map display choices to original indices
        correct_indices = [index_mapping[c + 1] for c in choices]
        
        # Log valid decompositions
        self.logger.log_decompositions(word, correct_indices, decompositions)
        
        # Check if word should be deleted (auto mode only)
        if auto_mode:
            self.auto_mode_stats['words_processed'] += 1
            if self._check_and_delete_word(word, correct_indices, decompositions):
                self.auto_mode_stats['words_deleted'] += 1
        
        # Train the model (interactive mode = immediate feedback)
        loss = self._train_on_choices(root, suffix_chains, correct_indices, interactive=True)
        self._print_training_result(loss, len(correct_indices))
        
        # Update and save periodically
        self.training_count += 1
        if self.training_count % self.config.checkpoint_frequency == 0:
            self.save()
        
        return True

    def _print_training_result(self, loss: float, num_choices: int) -> None:
        """Print training results"""
        if num_choices == 1:
            print(f"\n✅ Training step completed. Loss: {loss:.4f}")
        else:
            print(f"\n✅ Training completed on {num_choices} examples. Avg Loss: {loss:.4f}")
        
        print(f"Total training examples: {self.training_count}")
    
    def batch_train_from_file(self, filepath: Optional[str] = None) -> None:
        """
        Load all valid decompositions from file and train on them in batch.
        Uses true batch processing for efficiency.
        """
        filepath = filepath or self.config.valid_decompositions_file
        
        if not os.path.exists(filepath):
            print(f"\n⚠️  No valid decompositions file found at {filepath}")
            return
        
   
        
        # Load all entries
        entries = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return
        
        if not entries:
            print("⚠️  No entries found in file")
            return
        
        print(f"📖 Loaded {len(entries)} valid decompositions")
        
        # Group by word to find correct decompositions
        word_decompositions = {}
        for entry in entries:
            word = entry['word']
            if word not in word_decompositions:
                word_decompositions[word] = []
            word_decompositions[word].append(entry)
        
        print(f"📊 Found {len(word_decompositions)} unique words")
        
        # Collect all training examples
        training_data = []
        skipped = 0
        
        for word, correct_entries in word_decompositions.items():
            try:
                # Get all possible decompositions for this word
                suffix_chains = []
                decompositions = decompose(word)
                
                # Extract root
                root = decompositions[0][0] if decompositions else word

                for root_d, pos, suffix_chain, final_pos in decompositions:
                    suffix_chains.append(suffix_chain)
                
                if not suffix_chains or len(suffix_chains) == 1:
                    skipped += 1
                    continue
                
                # Find indices of correct decompositions
                correct_indices = self._match_decompositions(correct_entries, decompositions)
                
                if not correct_indices:
                    print(f"⚠️  Could not match decomposition for '{word}' - skipping")
                    skipped += 1
                    continue
                
                # Add each correct decomposition as a training example
                for correct_idx in correct_indices:
                    training_data.append((root, suffix_chains, correct_idx))
            
            except Exception as e:
                print(f"❌ Error preparing '{word}': {e}")
                skipped += 1
                continue
        
        if not training_data:
            print("⚠️  No valid training data collected")
            return
        
        print(f"\n🚀 Training on {len(training_data)} examples in batches of {self.trainer.batch_size}...")
        
        # Train using the improved trainer's batch processing
        try:
            history = self.trainer.train(
                training_data=training_data,
                num_epochs=5,  # Multiple passes over the data
                verbose=True
            )
            
            avg_loss = history['training_history'][-1] if history['training_history'] else 0.0
            
            # Update training count
            self.training_count += len(training_data)
            
            # Final summary
            print(f"\n{'='*70}")
            print(f"BATCH TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"✅ Successfully trained on: {len(training_data)} examples")
            print(f"⭐️  Skipped: {skipped} words")
            print(f"📊 Final loss: {avg_loss:.4f}")
            print(f"📈 Total training examples: {self.training_count}")
            print(f"{'='*70}\n")
            
            # Save the updated model
            self.save()
        
        except Exception as e:
            print(f"\n❌ Error during batch training: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate_word(self, word: str) -> None:
        """Evaluate model on a word without training - show only best prediction"""
        suffix_chains = []
        decompositions = decompose(word)
        
        # Extract root
        root = decompositions[0][0] if decompositions else word

        for root_d, pos, suffix_chain, final_pos in decompositions:
            suffix_chains.append(suffix_chain)
        
        if not suffix_chains:
            print(f"\n⚠️  No valid decompositions found for '{word}'")
            return
        
        # Get predictions
        try:
            predicted_idx, scores = self.trainer.predict(root, suffix_chains)
            
            print(f"\n🤖 ML Model's top prediction:")
            display.DecompositionDisplay.format_decomposition(
                word, decompositions[predicted_idx], scores[predicted_idx], 1, predicted_idx
            )
            
        
        except Exception as e:
            print(f"\n❌ Error getting predictions: {e}")
            import traceback
            traceback.print_exc()
    
    def show_statistics(self) -> None:
        """Display training statistics"""
        print(f"\n📊 Training Statistics:")
        print(f"  Total examples trained: {self.training_count}")
        
        if self.trainer.training_history:
            recent_losses = self.trainer.training_history[-20:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f"  Recent average loss: {avg_loss:.4f}")
            print(f"  Latest loss: {self.trainer.training_history[-1]:.4f}")
        
        if self.trainer.validation_history:
            print(f"  Best validation loss: {self.trainer.best_val_loss:.4f}")
        
        logged_count = self.logger.count_entries()
        print(f"  Valid decompositions logged: {logged_count}")
        

        
        # Auto mode stats
        if self.auto_mode_stats['words_processed'] > 0:
            print(f"\n  Auto mode statistics:")
            print(f"    - Words processed: {self.auto_mode_stats['words_processed']}")
            print(f"    - Words deleted: {self.auto_mode_stats['words_deleted']}")
            print(f"    - Words skipped: {self.auto_mode_stats['words_skipped']}")
        
        print(f"\n  Model configuration:")
        print(f"    - Architecture: {'LSTM' if self.model.use_lstm else 'Transformer'}")
        print(f"    - Loss function: {'Triplet' if self.trainer.use_triplet_loss else 'Contrastive'}")
        print(f"    - Batch size: {self.trainer.batch_size}")
        print(f"    - Embedding dim: {self.model.embed_dim}")
    
    def _handle_quit(self) -> bool:
        """Handle quit command. Returns True if should exit"""
        # Train any pending batch examples before quitting
        if self.pending_examples:
            print(f"\n🚀 Training on {len(self.pending_examples)} pending examples...")
            self._train_batch()
        
        if self.training_count > 0 and self.input_handler.confirm_save():
            self.save()
        print("Goodbye!")
        return True
    
    def auto_mode(self) -> None:
        """
        Auto mode: continuously present random words from words.txt for training.
        Deletes words from dictionary if their root exists in it.
        """

        
        print(f"💡 Words will be deleted if their root exists in the dictionary")
        print(f"   Press 'q' during selection to exit auto mode\n")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                # Get random word
                word = wrd.random_word()
                
                if not word:
                    print("\n✅ No more words in dictionary!")
                    break
                
                # Reset error counter on successful word retrieval
                consecutive_errors = 0
                
                # Show progress

                # Train on this word
                result = self.train_on_word(word, auto_mode=True)
                
                if result is None:  # User quit
                    print("\n👋 Exiting auto mode...")
                    break
                
                # Periodic save
                if self.training_count % self.config.checkpoint_frequency == 0:
                    self.save()
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted! Exiting auto mode...")
                break
            
            except Exception as e:
                consecutive_errors += 1
                print(f"\n❌ Error in auto mode: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n❌ Too many consecutive errors ({max_consecutive_errors}). Exiting auto mode...")
                    break
                
                # Continue to next word
                continue
        
        # Show final statistics
        print(f"\n{'='*70}")
        print(f"AUTO MODE SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Words processed: {self.auto_mode_stats['words_processed']}")
        print(f"🗑️  Words deleted: {self.auto_mode_stats['words_deleted']}")
        print(f"⏭️  Words skipped: {self.auto_mode_stats['words_skipped']}")
        print(f"{'='*70}\n")

    def interactive_loop(self) -> None:
        """Main interactive training loop"""
        display.welcome()
        
        print("\n💡 Commands:")
        print("  - Enter a word to analyze and train")
        print("  - 'auto' - Start auto mode (random words from dictionary)")
        print("  - 'eval <word>' - Evaluate model on a word")
        print("  - 'batch' - Train on all logged decompositions")
        print("  - 'stats' - Show training statistics")
        print("  - 'save' - Save model")
        print("  - 'quit' - Exit")
        
        while True:
            try:
                user_input = input("\n🔤 Enter word or command: ").strip().lower()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input == 'quit':
                    if self._handle_quit():
                        break
                
                elif user_input == 'save':
                    self.save()
                
                elif user_input == 'stats':
                    self.show_statistics()
                
                elif user_input == 'auto':
                    self.auto_mode()
                
                elif user_input == 'batch':
                    self.batch_train_from_file()
                
                elif user_input.startswith('batch '):
                    filepath = user_input[6:].strip()
                    if filepath:
                        self.batch_train_from_file(filepath)
                
                elif user_input.startswith('eval '):
                    word = user_input[5:].strip()
                    if word:
                        self.evaluate_word(word)
                
                else:
                    # Treat as word to analyze and train
                    result = self.train_on_word(user_input, auto_mode=False)
                    
                    if result is None:  # User chose to quit during training
                        if self._handle_quit():
                            break
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted!")
                if self._handle_quit():
                    break
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Entry point for interactive training"""
    # Ask user for training configuration


    print(decompose("güneş"))
    handler = UserInputHandler()
    # Get training mode preference
    mode = handler.get_training_mode()
    use_triplet = (mode == 'triplet')
    
    # Ask about architecture

    use_lstm = handler.get_arch_mode()

    display.ml_choices(mode,use_lstm)
    # Create and run trainer
    trainer = InteractiveTrainer(
        use_triplet_loss=use_triplet,
        use_lstm=use_lstm
    )
    trainer.interactive_loop()


if __name__ == "__main__":
    main()