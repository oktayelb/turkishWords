import concurrent.futures
import multiprocessing
from typing import List, Optional, Tuple, Dict

# Internal Project Imports
import app.display as Display
from app.data_manager import DataManager
from ml.ml_ranking_model import Ranker, Trainer
import util.decomposer as sfx
from util.suffix import Suffix

def _worker_decompose(word):
    return sfx.decompose(word)

class InteractiveTrainer:
    """
    Handles the interactive training loop.
    Delegates ML logic and configuration to the ml.ranker module.
    """
    
    def __init__(self):
        # 1. Initialize Components
        self.data_manager = DataManager()
    
        self.model = Ranker(suffix_vocab_size=len(sfx.ALL_SUFFIXES))
        self.trainer = Trainer(model=self.model)
        
        self.training_count = self.data_manager.load_training_count()

    def save(self):
        """
        Trigger a save. The Trainer handles the model weights; 
        we handle the training count file.
        """
        # The trainer knows its own save path from ml.config
        self.trainer.save_checkpoint()
        
        # We manually save the counter to the file path defined in DataManager
        # (Assuming DataManager still holds the path to training_count.txt)
        with open(self.data_manager.paths.training_count_path, "w") as f:
            f.write(str(self.training_count))
        print(f"Model saved")
    
    def _train_on_choices(self, suffix_chains: List[List], correct_indices: List[int]) -> float:
        """
        Format the user's choice into a training example and update the model.
        """
        # Encode all candidate chains
        encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
        
        # Format: (root_info, candidates, correct_index)
        # Note: root_info is empty [] because we currently don't use root embeddings
        training_data = [
            ([], encoded_chains, idx) for idx in correct_indices
        ]
        
        # Use persistent training to force the model to memorize this decision
        return self.trainer.train_persistent(training_data)

    def train_mode(self, word: str) -> Optional[bool]:
        """
        Main logic for processing a single word:
        Decompose -> Predict -> User Choice -> Train -> Save
        """
        decompositions = self.data_manager.decompose(word)

        if not decompositions:
            print(f"\n  No decompositions found for '{word}'")
            return False
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        if len(suffix_chains) == 1:
            print(f"\n Only one decomposition for '{word}' - skipping")
            return False
        
        # AI Suggestion (Optional)
        scores = None
        if self.training_count > 0:
            try:
                encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
                _, scores = self.trainer.predict(encoded_chains)
                print(f"\n ML Model predictions shown")
            except Exception:
                pass
        
        # User Interaction
        index_mapping = Display.show_decompositions(word, decompositions, scores)
        choices = Display.get_user_choices(len(suffix_chains))
        
        if choices is None: # User typed 'q'
            return None
        if choices == [-1]: # User typed 's' (skip)
            return False
        
        correct_indices = [index_mapping[c + 1] for c in choices]
        
        # Logging & Cleanup
        self.data_manager.log_decompositions(word, correct_indices, decompositions)
        self.data_manager.delete_word_if_root_exists(word, correct_indices, decompositions)
        
        # Training
        loss = self._train_on_choices(suffix_chains, correct_indices)
        print(f"\n Training complete. Loss: {loss:.4f}")
        
        self.training_count += 1
        print(f"Total examples: {self.training_count}")
        
        # Auto-Save: Access frequency from the Trainer instance
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
        
        return True
    
    def evaluation_mode(self, word: str):
        """Evaluate model on a word without training"""
        decompositions = self.data_manager.decompose(word)
        if not decompositions:
            print(f"\n  No decompositions found")
            return
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        try:
            encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
            pred_idx, scores = self.trainer.predict(encoded_chains)
            
            print(f"\n🤖 ML Model's top prediction:")
            Display.show_single_decomposition(
                word, decompositions[pred_idx], scores[pred_idx], 1, pred_idx
            )
        except Exception as e:
            print(f"\n Error: {e}")

    def relearn_mode(self, filepath: Optional[str] = None):
        """Train on all logged valid decompositions."""
        entries = self.data_manager.get_valid_decomps()
        
        # Group by word to handle multiple correct decompositions per word
        word_groups = {}
        for entry in entries:
            word_groups.setdefault(entry['word'], []).append(entry)
        
        training_data = []
        skipped = 0
        
        # Re-verify decomps and prepare data
        for word, correct_entries in word_groups.items():
            try:
                decompositions = self.data_manager.decompose(word)
                if not decompositions:
                    skipped += 1
                    continue
                
                correct_indices = Translation._match_decompositions(correct_entries, decompositions)
                if not correct_indices:
                    skipped += 1
                    continue
                
                suffix_chains = [chain for _, _, chain, _ in decompositions]
                encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
                
                for idx in correct_indices:
                    training_data.append(([], encoded_chains, idx))
            
            except Exception:
                skipped += 1
        
        if not training_data:
            print("  No valid training data found.")
            return
        
        print(f"\n Training on {len(training_data)} examples...")
        
        # Delegate batch training to Trainer
        history = self.trainer.train(training_data, num_epochs=5, verbose=True)
        final_loss = history['train_history'][-1] if history['train_history'] else 0.0
        
        self.training_count += len(training_data)
        Display.show_batch_summary(len(training_data), skipped, final_loss, self.training_count)
        self.save()

    def auto_mode(self):
        """Loop that picks random words and prompts the user."""
        print(f"   Words deleted if root exists in dictionary")
        print(f"   Press 'q' to exit\n")
        auto_stats = {
            'words_processed': 0,
            'words_deleted': 0,
            'words_skipped': 0
        }
        while True:
            try:
                word = self.data_manager.random_word()
                if not word:
                    print("\n✅ No more words!")
                    break
                
                auto_stats['words_processed'] += 1
                result = self.train_on_word(word)
                
                if result is None: # User Quit
                    auto_stats['words_processed'] -= 1 
                    print("\n Exiting auto mode...")
                    break
                
                if result is False: # Skipped
                    auto_stats['words_skipped'] += 1
                
                # Check deletion status
                if self.data_manager.exists(word.lower()) == 0: 
                    auto_stats['words_deleted'] += 1
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n Error: {e}")
                break
        
        Display.show_auto_summary(auto_stats)

    def text_mode(self):
        """
        Decompose a full text file using Multiprocessing, then use Batch AI to rank.
        """
        print(f"\nExample: 'sample.txt'")
        filename = input("Enter filename (default: sample.txt): ").strip()
        if not filename: filename = "sample.txt" # Defaults should be handled by DataManager if specific logic exists
        
        # 1. Load
        try:
            # Assuming DataManager has a method to get tokens, or we use the default
            text = self.data_manager.get_text_tokenized()
        except Exception as e:
            print(f" Error loading text: {e}")
            return

        if not text:
            print("  No text found")
            return
            
        print(f"    Input: {len(text)} tokens")
        
        # 2. Dedup
        unique_words = list(set(text))
        print(f"    Unique words: {len(unique_words)}")
        
        # 3. Parallel Decomposition
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"    Spawning {num_cores} worker processes...")
        
        word_results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Use top-level function _worker_decompose to ensure pickle compatibility
            results = executor.map(_worker_decompose, unique_words, chunksize=100)
            
            for word, decompositions in zip(unique_words, results):
                word_results[word] = decompositions

        # 4. Prepare for AI
        print("    Preparing AI Batch...")
        cache = {}
        ambiguous_batches = [] # Tuple: (original_word_index, encoded_chains)
        
        for i, word in enumerate(unique_words):
            decomps = word_results[word]
            if not decomps:
                cache[word] = word
            elif len(decomps) == 1:
                cache[word] = Display.format_decomposition(word, decomps[0], simple=True)
            else:
                # Encode for ML
                suffix_chains = [chain for _, _, chain, _ in decomps]
                encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
                ambiguous_batches.append((i, encoded_chains))
                cache[word] = ("__WAITING__", decomps)

        # 5. Batch Inference
        if ambiguous_batches:
            print(f"    Ranking {len(ambiguous_batches)} ambiguous words...")
            just_chains = [x[1] for x in ambiguous_batches]
            
            # Using the optimized batch_predict from Trainer
            batch_predictions = self.trainer.batch_predict(just_chains)

            # Apply results
            for idx, (original_idx, _) in enumerate(ambiguous_batches):
                best_idx_in_decomp = batch_predictions[idx][0]
                word = unique_words[original_idx]
                _, decomps = cache[word]
                
                if best_idx_in_decomp >= len(decomps): best_idx_in_decomp = 0
                best_decomp = decomps[best_idx_in_decomp]
                
                cache[word] = Display.format_decomposition(word, best_decomp, simple=True)

        # 6. Reconstruct
        print("    Reconstructing...")
        final_output = []
        for word in text:
            final_output.append(cache.get(word, word))
            
        output_text = '\n'.join(final_output)
        self.data_manager.write_decomposed_text(output_text)
        print("    Done.")

    def menu(self):

        """Main entry point."""
        Display.welcome()
        
        while True:
            try:
                cmd = input("\n Enter word or command: ").strip().lower()
                if not cmd: continue

                if cmd == 'quit':
                    if self.training_count > 0 and Display.confirm_save():
                        self.save()
                    break
                elif cmd == 'save':
                    self.save()
                elif cmd == 'stats':
                    Display.show_statistics(self)
                elif cmd == 'auto':
                    self.auto_mode()
                elif cmd == 'text':
                    self.text_mode()
                elif cmd == 'relearn':
                    self.relearn_mode()
                elif cmd.startswith('eval '):
                    self.evaluation_mode(cmd[5:].strip())
                else:
                    result = self.train_mode(cmd)
                    if result is None and Display.confirm_save():
                        self.save()
                        break
            
            except KeyboardInterrupt:
                if Display.confirm_save():
                    self.save()
                break
            except Exception as e:
                print(f"\n Error: {e}")
    

        ##abstract
    
    
class Translation:   
    def _match_decompositions(self, entries: List[Dict], decompositions: List[Tuple]) -> List[int]:
        """Helper to match logged decomposition strings back to current decomposition indices."""
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

    def encode_suffix_chain(self, suffix_chain: List[Suffix]) -> List[Tuple[int, int]]:
        """
        Convert a list of Suffix objects into (suffix_id, category_id) tuples 
        that the ML model can understand.
        """
        suffix_to_id = {
            suffix.name: idx + 1  
            for idx, suffix in enumerate(sfx.ALL_SUFFIXES)
        }
        category_to_id = {'Noun': 0, 'Verb': 1}
        if not suffix_chain:
            return []
        
        encoded = []
        for suffix_obj in suffix_chain:
            suffix_id = suffix_to_id.get(suffix_obj.name, 0)
            category_id = category_to_id.get(suffix_obj.makes.name, 0)
            encoded.append((suffix_id, category_id))
        
        return encoded
    