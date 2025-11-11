"""
Interactive training pipeline for Turkish morphological analyzer
Integrates main.py decomposition with ML ranking model
"""

import torch
import os
import json
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

# Import your modules
from util.decomposition import decompose
import util.print as display
from ml_ranking_model import (
    SuffixVocabulary, 
    DecompositionRanker, 
    DecompositionTrainer
)


@dataclass
class TrainingConfig:
    """Configuration for the interactive trainer"""
    model_path: str = "data/turkish_morph_model.pt"
    vocab_path: str = "data/suffix_vocab.json"
    training_count_file: str = "data/training_count.txt"
    valid_decompositions_file: str = "data/valid_decompositions.jsonl"
    checkpoint_frequency: int = 10  # Save every N examples
    
    # Model hyperparameters
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    learning_rate: float = 1e-4


class DecompositionDisplay:
    """Handles formatting and display of decompositions"""
    
    @staticmethod
    def format_decomposition(word: str, decomp: Tuple, score: Optional[float] = None, 
                            display_idx: int = 1, original_idx: int = 0) -> str:
        """Format a single decomposition for display"""
        root, pos, chain, final_pos = decomp
        
        lines = []
        lines.append(f"\n[Option {display_idx}] (Original index: {original_idx + 1})")
        
        if score is not None:
            lines.append(f"ML Score: {score:.4f}")
        
        lines.append(f"Root:      {root} ({pos})")
        
        if chain:
            suffix_forms, suffix_names, formation_steps = DecompositionDisplay._build_suffix_info(
                root, pos, chain
            )
            lines.append(f"Suffixes:  {' + '.join(suffix_forms)}")
            lines.append(f"Names:     {' + '.join(suffix_names)}")
            lines.append(f"Formation: {' → '.join(formation_steps)}")
        else:
            lines.append(f"Formation: {root + '-' if pos == 'verb' else root} (no suffixes)")
        
        lines.append(f"Final POS: {final_pos}")
        lines.append("-" * 70)
        
        return display.header("\n".join(lines))
    
    @staticmethod
    def _build_suffix_info(root: str, pos: str, chain: List) -> Tuple[List[str], List[str], List[str]]:
        """Build suffix forms, names, and formation steps"""
        current_word = root
        suffix_forms = []
        suffix_names = []
        formation_steps = [root + ("-" if pos == "verb" else "")]
        
        for suffix_obj in chain:
            suffix_form = suffix_obj.form(current_word)
            suffix_forms.append(suffix_form)
            suffix_names.append(suffix_obj.name)
            
            current_word += suffix_form
            target_pos = "verb" if suffix_obj.makes.name == "Verb" else "noun"
            word_display = current_word + ("-" if target_pos == "verb" else "")
            formation_steps.append(word_display)
        
        return suffix_forms, suffix_names, formation_steps
    
    @staticmethod
    def display_all(word: str, decompositions: List[Tuple], 
                   scores: Optional[List[float]] = None) -> Dict[int, int]:
        """
        Display all decompositions sorted by score
        Returns mapping from display index to original index
        """

        display.header(word)

        # Sort by score if available
        if scores:
            indexed_decomps = list(enumerate(zip(decompositions, scores)))
            indexed_decomps.sort(key=lambda x: x[1][1])  # Sort by score ascending
        else:
            indexed_decomps = [(i, (d, None)) for i, d in enumerate(decompositions)]
        
        # Display each decomposition
        index_mapping = {}
        for display_idx, (original_idx, (decomp, score)) in enumerate(indexed_decomps, 1):
            DecompositionDisplay.format_decomposition(
                word, decomp, score, display_idx, original_idx
            )
            index_mapping[display_idx] = original_idx
        
        return index_mapping


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


class DecompositionLogger:
    """Handles logging of valid decompositions"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
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
    """Main interactive training interface"""
    
    def __init__(self, config: Optional[TrainingConfig] = None, auto_batch_train: bool = False):
        self.config = config or TrainingConfig()
        
        # Initialize components
        self.vocab = self._initialize_vocabulary()
        self.model = self._initialize_model()
        self.trainer = self._initialize_trainer()
        self.logger = DecompositionLogger(self.config.valid_decompositions_file)
        self.display = DecompositionDisplay()
        self.input_handler = UserInputHandler()
        
        # Load state
        self.training_count = self._load_training_count()
        self._load_checkpoint_if_exists()
        
        # Batch train if requested
        if auto_batch_train:
            self.batch_train_from_file()
    
    def _initialize_vocabulary(self) -> SuffixVocabulary:
        """Initialize and save vocabulary"""
        print("Initializing vocabulary from suffixes.py...")
        vocab = SuffixVocabulary()
        vocab.save(self.config.vocab_path)
        print(f"✅ Vocabulary saved to {self.config.vocab_path}")
        return vocab
    
    def _initialize_model(self) -> DecompositionRanker:
        """Initialize the ranking model"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        return DecompositionRanker(
            vocab_size=self.vocab.num_suffixes,
            num_categories=2,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads
        )
    
    def _initialize_trainer(self) -> DecompositionTrainer:
        """Initialize the trainer"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return DecompositionTrainer(
            self.model, 
            self.vocab, 
            lr=self.config.learning_rate, 
            device=device
        )
    
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
        """Save model, vocabulary, and training count"""
        self.trainer.save_checkpoint(self.config.model_path)
        self.vocab.save(self.config.vocab_path)
        self._save_training_count()
        print(f"✅ Model, vocabulary, and training count saved")
    
    
    def _train_on_choices(self, suffix_chains: List[List], correct_indices: List[int]) -> float:
        """Train model on user's choices"""
        if len(correct_indices) == 1:
            # Single correct answer
            return self.trainer.train_step(suffix_chains, correct_indices[0])
        else:
            # Multiple correct answers - train on preference pairs
            total_loss = 0
            pair_count = 0
            
            for i in range(len(correct_indices)):
                for j in range(i + 1, len(correct_indices)):
                    better_idx = correct_indices[i]
                    worse_idx = correct_indices[j]
                    loss = self.trainer.train_step_pairwise(suffix_chains, better_idx, worse_idx)
                    total_loss += loss
                    pair_count += 1
            
            return total_loss / pair_count if pair_count > 0 else 0
    
    def train_on_word(self, word: str) -> Optional[bool]:
        """
        Interactive training on a single word
        Returns: True if trained, False if skipped, None if quit
        """
        # Get decompositions
        suffix_chains = []
        decompositions = decompose(word)

        for root, pos, suffix_chain, final_pos in decompositions:
        # suffix_chain already contains Suffix objects
            suffix_chains.append(suffix_chain)
        
        # Handle edge cases
        if not suffix_chains:
            print(f"\n⚠️  No valid decompositions found for '{word}'")
            return False
        
        if len(suffix_chains) == 1:
            print(f"\n✅ Only one decomposition exists for '{word}' - no training needed")
            return False
        
        # Get ML predictions if model has been trained
        scores = None
        if self.training_count > 0:
            predicted_idx, scores = self.trainer.predict(suffix_chains)
            print(f"\n🤖 ML Model predicts option with highest score")
        
        # Display options and get user choice
        index_mapping = self.display.display_all(word, decompositions, scores)
        choices = self.input_handler.get_correct_choices(len(suffix_chains))
        
        if choices is None:  # Quit
            return None
        elif choices == [-1]:  # Skip
            return False
        
        # Map display choices to original indices
        correct_indices = [index_mapping[c + 1] for c in choices]
        
        # Log valid decompositions
        self.logger.log_decompositions(word, correct_indices, decompositions)
        
        # Train the model
        loss = self._train_on_choices(suffix_chains, correct_indices)
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
            pair_count = num_choices * (num_choices - 1) // 2
            print(f"\n✅ Training completed on {pair_count} preference pairs. Avg Loss: {loss:.4f}")
        
        print(f"Total training examples: {self.training_count}")
    
    def batch_train_from_file(self, filepath: Optional[str] = None) -> None:
        """
        Load all valid decompositions from file and train on them in batch
        """
        filepath = filepath or self.config.valid_decompositions_file
        
        if not os.path.exists(filepath):
            print(f"\n⚠️  No valid decompositions file found at {filepath}")
            return
        
        display.header(filepath)

        
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
        
        # Train on each word
        total_trained = 0
        total_loss = 0.0
        skipped = 0
        
        for word_idx, (word, correct_entries) in enumerate(word_decompositions.items(), 1):
            try:
                # Get all possible decompositions for this word
                suffix_chains = []
                decompositions = decompose(word)

                for root, pos, suffix_chain, final_pos in decompositions:
                # suffix_chain already contains Suffix objects
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
                
                # Train on this word
                loss = self._train_on_choices(suffix_chains, correct_indices)
                total_loss += loss
                total_trained += 1
                self.training_count += 1
                
                # Print progress every 10 words
                if word_idx % 10 == 0:
                    avg_loss = total_loss / total_trained if total_trained > 0 else 0
                    print(f"Progress: {word_idx}/{len(word_decompositions)} words | "
                          f"Trained: {total_trained} | Avg Loss: {avg_loss:.4f}")
            
            except Exception as e:
                print(f"❌ Error training on '{word}': {e}")
                skipped += 1
                continue
        
        # Final summary
        avg_loss = total_loss / total_trained if total_trained > 0 else 0
        print(f"\n{'='*70}")
        print(f"BATCH TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successfully trained on: {total_trained} words")
        print(f"⏭️  Skipped: {skipped} words")
        print(f"📊 Average loss: {avg_loss:.4f}")
        print(f"📈 Total training examples: {self.training_count}")
        print(f"{'='*70}\n")
        
        # Save the updated model
        if total_trained > 0:
            self.save()
    
    def evaluate_word(self, word: str) -> None:
        """Evaluate model on a word without training - show only best prediction"""
        suffix_chains = []
        decompositions = decompose(word)

        for root, pos, suffix_chain, final_pos in decompositions:
        # suffix_chain already contains Suffix objects
            suffix_chains.append(suffix_chain)
        
        if not suffix_chains:
            print(f"\n⚠️  No valid decompositions found for '{word}'")
            return
        
        
        # Get predictions
        predicted_idx, scores = self.trainer.predict(suffix_chains)
        
        print(f"\n🤖 ML Model's top prediction:")

        self.display.format_decomposition(
            word, decompositions[predicted_idx], scores[predicted_idx], 1, predicted_idx
        )
    
    def show_statistics(self) -> None:
        """Display training statistics"""
        print(f"\n📊 Training Statistics:")
        print(f"  Total examples trained: {self.training_count}")
        
        if self.trainer.training_history:
            recent_losses = self.trainer.training_history[-20:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f"  Recent average loss: {avg_loss:.4f}")
            print(f"  Latest loss: {self.trainer.training_history[-1]:.4f}")
        
        logged_count = self.logger.count_entries()
        print(f"  Valid decompositions logged: {logged_count}")
    
    def _handle_quit(self) -> bool:
        """Handle quit command. Returns True if should exit"""
        if self.training_count > 0 and self.input_handler.confirm_save():
            self.save()
        print("Goodbye!")
        return True

    def interactive_loop(self) -> None:
        """Main interactive training loop"""
        display.welcome()
        
        while True:
            try:
                user_input = input("\n📤 Enter word or command: ").strip().lower()
                
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
                    result = self.train_on_word(user_input)
                    
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

    config = TrainingConfig(
        model_path="data/turkish_morph_model.pt",
        vocab_path="data/suffix_vocab.json"
    )
    
    trainer = InteractiveTrainer(config)
    trainer.interactive_loop()


if __name__ == "__main__":
    main()