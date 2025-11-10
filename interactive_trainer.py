"""
Interactive training pipeline for Turkish morphological analyzer
Integrates main.py decomposition with ML ranking model
"""

import torch
import os
import json
from typing import List, Optional

# Import your modules
from decomposition import decompose, get_suffix_object_lists
from ml_ranking_model import (
    SuffixVocabulary, 
    DecompositionRanker, 
    DecompositionTrainer
)


class InteractiveTrainer:
    def __init__(self, model_path: str = "model_checkpoint.pt", vocab_path: str = "vocab.json"):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.training_count_file = "training_count.txt"
        self.valid_decompositions_file = "valid_decompositions.jsonl"  # New file for logging
        
        # Initialize vocabulary and auto-save to JSON
        print("Initializing vocabulary from suffixes.py...")
        self.vocab = SuffixVocabulary()
        self.vocab.save(vocab_path)  # Always save/update the vocabulary file
        print(f"✓ Vocabulary saved to {vocab_path}")
        
        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.model = DecompositionRanker(
            vocab_size=self.vocab.num_suffixes,
            num_categories=2,
            embed_dim=128,
            num_layers=4,
            num_heads=8
        )
        
        # Initialize trainer
        self.trainer = DecompositionTrainer(self.model, self.vocab, lr=1e-4, device=device)
        
        # Load existing model if available
        if os.path.exists(model_path):
            try:
                self.trainer.load_checkpoint(model_path)
                print(f"✓ Loaded existing model from {model_path}")
            except Exception as e:
                print(f"⚠  Could not load model: {e}")
                print("Starting with fresh model")
        else:
            print("Starting with fresh model")
        
        self.training_count = self.load_training_count()
    
    def display_decompositions(self, word: str, suffix_chains: List[List], 
                            scores: Optional[List[float]] = None):
        """Display all decompositions with optional ML scores"""
        print("\n" + "="*70)
        print(f"DECOMPOSITION CANDIDATES FOR: {word}")
        print("="*70)
        
        decompositions = decompose(word)
        
        # Create list of (index, decomposition, score) and sort by score
        if scores:
            indexed_decomps = list(enumerate(zip(decompositions, scores)))
            # Sort by score (ascending), so highest score is last
            indexed_decomps.sort(key=lambda x: x[1][1])
        else:
            indexed_decomps = [(i, (d, None)) for i, d in enumerate(decompositions)]
        
        for display_idx, (original_idx, (decomp, score)) in enumerate(indexed_decomps, 1):
            root, pos, chain, final_pos = decomp
            
            print(f"\n[Option {display_idx}] (Original index: {original_idx + 1})")
            if score is not None:
                print(f"ML Score: {score:.4f}")
            
            print(f"Root:      {root} ({pos})")
            
            if chain:
                # Generate suffix forms progressively
                current_word = root
                suffix_forms = []
                suffix_names = []
                
                for suffix_obj in chain:
                    suffix_form = suffix_obj.form(current_word)
                    suffix_forms.append(suffix_form)
                    suffix_names.append(suffix_obj.name)
                    current_word += suffix_form
                
                print(f"Suffixes:  {' + '.join(suffix_forms)}")
                print(f"Names:     {' + '.join(suffix_names)}")
                
                # Show progressive formation
                steps = [root + ("-" if pos == "verb" else "")]
                current_word = root
                for suffix_obj in chain:
                    suffix_form = suffix_obj.form(current_word)
                    current_word += suffix_form
                    target_pos = "verb" if suffix_obj.makes.name == "Verb" else "noun"
                    word_display = current_word + ("-" if target_pos == "verb" else "")
                    steps.append(word_display)
                
                print(f"Formation: {' → '.join(steps)}")
            else:
                print(f"Formation: {root + '-' if pos == 'verb' else root} (no suffixes)")
            
            print(f"Final POS: {final_pos}")
            print("-" * 70)
        
        # Return mapping from display index to original index for user input
        return {i+1: original_idx for i, (original_idx, _) in enumerate(indexed_decomps)}
    
    def get_user_choice(self, num_options: int) -> Optional[List[int]]:
        """Get user's choice(s) of correct decomposition(s)"""
        while True:
            try:
                choice = input(f"\nSelect correct decomposition(s) (1-{num_options}, 's' to skip, 'q' to quit): ").strip().lower()
                
                if choice == 'q':
                    return None
                elif choice == 's':
                    return [-1]
                
                # Parse multiple numbers - handle spaces, commas, hyphens, etc.
                # Replace common separators with spaces
                choice = choice.replace(',', ' ').replace('-', ' ').replace('/', ' ').replace(';', ' ')
                # Split and filter out empty strings
                parts = [p.strip() for p in choice.split() if p.strip()]
                
                if not parts:
                    print("No input provided. Please try again.")
                    continue
                
                # Convert to integers
                choices = []
                for part in parts:
                    try:
                        num = int(part)
                        choices.append(num)
                    except ValueError:
                        print(f"Invalid number: '{part}'")
                        choices = None
                        break
                
                if choices is None:
                    continue
                
                # Validate all choices are in range
                invalid = [c for c in choices if c < 1 or c > num_options]
                if invalid:
                    print(f"Invalid choice(s): {invalid}. Please enter numbers between 1 and {num_options}")
                    continue
                
                # Remove duplicates while preserving order
                seen = set()
                unique_choices = []
                for c in choices:
                    if c not in seen:
                        seen.add(c)
                        unique_choices.append(c - 1)  # Convert to 0-indexed
                
                return unique_choices
                
            except Exception as e:
                print(f"Error parsing input: {e}. Please try again.")

    def save_valid_decomposition(self, word: str, correct_indices: List[int], decompositions: List):
        """Save validated decomposition(s) to file in JSONL format"""
        try:
            with open(self.valid_decompositions_file, 'a', encoding='utf-8') as f:
                for idx in correct_indices:
                    root, pos, chain, final_pos = decompositions[idx]
                    
                    # Build suffix information
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
                    
                    # Create the entry
                    entry = {
                        'word': word,
                        'root': root,
                        'root_pos': pos,
                        'suffixes': suffix_info,
                        'final_pos': final_pos
                    }
                    
                    # Write as single line JSON
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"📝 Saved {len(correct_indices)} valid decomposition(s) to {self.valid_decompositions_file}")
        
        except Exception as e:
            print(f"⚠  Could not save valid decomposition: {e}")

    def train_on_word(self, word: str):
        """Interactive training on a single word"""
        # Get all decompositions
        suffix_chains = get_suffix_object_lists(word)
        decompositions = decompose(word)
        
        if not suffix_chains:
            print(f"\n⚠  No valid decompositions found for '{word}'")
            return False
        
        if len(suffix_chains) == 1:
            print(f"\n✓ Only one decomposition exists for '{word}' - no training needed")
            return False
        
        # Get ML predictions if model has been trained
        if self.training_count > 0:
            predicted_idx, scores = self.trainer.predict(suffix_chains)
            print(f"\n🤖 ML Model predicts option with highest score")
        else:
            scores = None
        
        # Display options (returns mapping from display to original indices)
        index_mapping = self.display_decompositions(word, suffix_chains, scores)
        
        # Get user choice(s)
        choices = self.get_user_choice(len(suffix_chains))
        
        if choices is None:  # Quit
            return None
        elif choices == [-1]:  # Skip
            return False
        
        # Map display choices to original indices
        correct_indices = [index_mapping[c + 1] for c in choices]
        
        # Save the valid decomposition(s)
        self.save_valid_decomposition(word, correct_indices, decompositions)
        
        # Train on the example(s)
        if len(correct_indices) == 1:
            # Single correct answer
            loss = self.trainer.train_step(suffix_chains, correct_indices[0])
            print(f"\n✓ Training step completed. Loss: {loss:.4f}")
        else:
            # Multiple correct answers with preference order
            # Train on pairs: each earlier choice is preferred over later ones
            total_loss = 0
            pair_count = 0
            for i in range(len(correct_indices)):
                for j in range(i + 1, len(correct_indices)):
                    better_idx = correct_indices[i]
                    worse_idx = correct_indices[j]
                    loss = self.trainer.train_step_pairwise(suffix_chains, better_idx, worse_idx)
                    total_loss += loss
                    pair_count += 1
            
            avg_loss = total_loss / pair_count if pair_count > 0 else 0
            print(f"\n✓ Training completed on {pair_count} preference pairs. Avg Loss: {avg_loss:.4f}")
        
        self.training_count += 1
        print(f"Total training examples: {self.training_count}")
        
        # Save checkpoint every 10 examples
        if self.training_count % 10 == 0:
            self.save()
        
        return True

    def evaluate_word(self, word: str):
        """Evaluate model on a word without training - show only best prediction"""
        suffix_chains = get_suffix_object_lists(word)
        
        if not suffix_chains:
            print(f"\n⚠  No valid decompositions found for '{word}'")
            return
        
        if len(suffix_chains) == 1:
            print(f"\n✓ Only one decomposition exists for '{word}'")
            self.display_decompositions(word, suffix_chains)
            return
        
        # Get predictions
        predicted_idx, scores = self.trainer.predict(suffix_chains)
        
        print(f"\n🤖 ML Model's top prediction:")
        print("="*70)
        
        # Get the decomposition with highest score
        decompositions = decompose(word)
        root, pos, chain, final_pos = decompositions[predicted_idx]
        
        print(f"Root:      {root} ({pos})")
        print(f"ML Score:  {scores[predicted_idx]:.4f}")
        
        if chain:
            current_word = root
            suffix_forms = []
            suffix_names = []
            
            for suffix_obj in chain:
                suffix_form = suffix_obj.form(current_word)
                suffix_forms.append(suffix_form)
                suffix_names.append(suffix_obj.name)
                current_word += suffix_form
            
            print(f"Suffixes:  {' + '.join(suffix_forms)}")
            print(f"Names:     {' + '.join(suffix_names)}")
            
            steps = [root + ("-" if pos == "verb" else "")]
            current_word = root
            for suffix_obj in chain:
                suffix_form = suffix_obj.form(current_word)
                current_word += suffix_form
                target_pos = "verb" if suffix_obj.makes.name == "Verb" else "noun"
                word_display = current_word + ("-" if target_pos == "verb" else "")
                steps.append(word_display)
            
            print(f"Formation: {' → '.join(steps)}")
        else:
            print(f"Formation: {root + '-' if pos == 'verb' else root} (no suffixes)")
        
        print(f"Final POS: {final_pos}")
        print("="*70)
        
    def load_training_count(self) -> int:
        """Load training count from file, or start at 0 if not found."""
        try:
            if os.path.exists(self.training_count_file):
                with open(self.training_count_file, "r") as f:
                    return int(f.read().strip())
        except Exception as e:
            print(f"⚠  Could not read training_count.txt: {e}")
        return 0  # default if missing or invalid

    def save_training_count(self):
        """Save current training count to file."""
        try:
            with open(self.training_count_file, "w") as f:
                f.write(str(self.training_count))
        except Exception as e:
            print(f"⚠  Could not save training_count.txt: {e}")

    def save(self):
        """Save model, vocabulary, and training count"""
        self.trainer.save_checkpoint(self.model_path)
        self.vocab.save(self.vocab_path)
        self.save_training_count()
        print(f"✓ Model, vocabulary, and training count saved")

    def interactive_loop(self):
        """Main interactive training loop"""
        print("\n" + "=" * 70)
        print("INTERACTIVE MORPHOLOGICAL ANALYZER TRAINER")
        print( "=" * 70)
        print("\nCommands:")
        print("  - Enter a word to analyze and train")
        print("  - 'eval <word>' to evaluate without training")
        print("  - 'save' to save the model")
        print("  - 'stats' to see training statistics")
        print("  - 'quit' to exit")
        print("="*70)
        
        while True:
            try:
                user_input = input("\n🔤 Enter word or command: ").strip().lower()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input == 'quit':
                    if self.training_count > 0:
                        save_choice = input("Save model before quitting? (y/n): ").strip().lower()
                        if save_choice == 'y':
                            self.save()
                    print("Goodbye!")
                    break
                
                elif user_input == 'save':
                    self.save()
                
                elif user_input == 'stats':
                    print(f"\n📊 Training Statistics:")
                    print(f"  Total examples trained: {self.training_count}")
                    if self.trainer.training_history:
                        recent_losses = self.trainer.training_history[-20:]
                        avg_loss = sum(recent_losses) / len(recent_losses)
                        print(f"  Recent average loss: {avg_loss:.4f}")
                        print(f"  Latest loss: {self.trainer.training_history[-1]:.4f}")
                    
                    # Show count of valid decompositions logged
                    if os.path.exists(self.valid_decompositions_file):
                        try:
                            with open(self.valid_decompositions_file, 'r', encoding='utf-8') as f:
                                line_count = sum(1 for _ in f)
                            print(f"  Valid decompositions logged: {line_count}")
                        except:
                            pass
                
                elif user_input.startswith('eval '):
                    word = user_input[5:].strip()
                    if word:
                        self.evaluate_word(word)
                
                else:
                    # Treat as word to analyze and train
                    word = user_input
                    result = self.train_on_word(word)
                    
                    if result is None:  # User chose to quit
                        if self.training_count > 0:
                            save_choice = input("Save model before quitting? (y/n): ").strip().lower()
                            if save_choice == 'y':
                                self.save()
                        print("Goodbye!")
                        break
            
            except KeyboardInterrupt:
                print("\n\n⚠  Interrupted!")
                if self.training_count > 0:
                    save_choice = input("Save model before quitting? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        self.save()
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Entry point for interactive training"""
    trainer = InteractiveTrainer(
        model_path="turkish_morph_model.pt",
        vocab_path="suffix_vocab.json"
    )
    trainer.interactive_loop()


if __name__ == "__main__":
    main()