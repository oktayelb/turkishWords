from typing import List, Optional, Tuple, Dict

from interactive.display import TrainerDisplay
from data.data_manager import DataManager
from ml_ranking_model import Ranker, Trainer
from interactive.config import TrainingConfig
import concurrent.futures
import multiprocessing
class InteractiveTrainer:
    """Handles training logic and model management with pre-encoded suffix data"""
    
    def __init__(self):
        
        self.data_manager = DataManager()
        self.config = TrainingConfig()
        self.display = TrainerDisplay()
        
        # Create suffix and category mappings for encoding
        self.category_to_id = {'Noun': 0, 'Verb': 1}
        
        # FIX: Initialize suffix_to_id mapping from data_manager.suffixes
        self.suffix_to_id = {
            suffix.name: idx + 1  # Start from 1, reserve 0 for padding
            for idx, suffix in enumerate(self.data_manager.suffixes)
        }
        
        self.model =  Ranker(
            suffix_vocab_size=len(self.data_manager.suffixes),  # Use actual vocab size
            num_categories=self.config.category_num,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
        )
        self.trainer = Trainer(
            model= self.model,
            model_path=self.data_manager.config.model_path,
            lr=self.config.learning_rate,
            batch_size=self.config.batch_size,
            patience=self.config.checkpoint_frequency
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
            # FIX: Handle unknown suffixes gracefully with padding index
            suffix_id = self.suffix_to_id.get(suffix_obj.name, 0)
            category_id = self.category_to_id.get(suffix_obj.makes.name, 0)
            encoded.append((suffix_id, category_id))
        
        return encoded
    


    def save(self):
        """Save model and training count"""
        ##TODO ## https://gemini.google.com/app/7b96b246a19d322b
        
        self.trainer.save_checkpoint(self.data_manager.config.model_path)
        with open(self.data_manager.config.training_count_file, "w") as f:
            f.write(str(self.training_count))
        print(f"✅ Model saved")
    
    def _train_on_choices(self, suffix_chains: List[List], 
                         correct_indices: List[int]) -> float:
        """
        Train on user's choices with pre-encoded data.
        
        Args:
            suffix_chains: List of suffix object chains
            correct_indices: Indices of correct decompositions
            
        Returns:
            Average training loss
        """
        # Encode all suffix chains to (suffix_id, category_id) format
        encoded_chains = [self.encode_suffix_chain(chain) for chain in suffix_chains]
        
        # Create training data in new format: (root_info, candidates, correct_idx)
        training_data = [
            ([], encoded_chains, idx) for idx in correct_indices
        ]
        # CHANGED: Use the new persistent training method
        # This will loop internally until the model learns this specific word
        return self.trainer.train_persistent(training_data)

    def train_on_word(self, word: str) -> Optional[bool]:
        """
        Train on a single word.
        Returns: True if trained, False if skipped, None if quit
        """
        decompositions = self.data_manager.decompose(word)

        ## non decomposable word, the word is also not in dictionary.
        if not decompositions:
            print(f"\n  No decompositions found for '{word}'")
            return False
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        ## nnon decomposable word, the word is in dictionary.
        if len(suffix_chains) == 1:
            print(f"\n Only one decomposition for '{word}' - skipping")
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
            self.display.show_single_decomposition(
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
        skip_logs = {
            'no_decomp': [],
            'single_decomp': [],
            'match_failed': [],
            'exception': []
        }
        
        for word, correct_entries in word_groups.items():
            try:
                decompositions = self.data_manager.decompose(word)
                
                if not decompositions:
                    skipped += 1
                    skip_logs['no_decomp'].append({
                        'word': word,
                        'logged_entries': correct_entries
                    })
                    continue
                """     
                if len(decompositions) == 1:
                    skipped += 1
                    skip_logs['single_decomp'].append({
                        'word': word,
                        'decomposition': {
                            'root': decompositions[0][0],
                            'suffixes': [s.name for s in decompositions[0][2]] if decompositions[0][2] else []
                        },
                        'logged_entries': correct_entries
                    })
                    continue
                """ 
                suffix_chains = [chain for _, _, chain, _ in decompositions]
                
                correct_indices = self._match_decompositions(correct_entries, decompositions)
                if not correct_indices:
                    skipped += 1
                    skip_logs['match_failed'].append({
                        'word': word,
                        'logged_entries': [{
                            'root': e['root'],
                            'suffixes': [s['name'] for s in e.get('suffixes', [])]
                        } for e in correct_entries],
                        'current_decompositions': [{
                            'root': root,
                            'suffixes': [s.name for s in chain] if chain else []
                        } for root, _, chain, _ in decompositions]
                    })
                    continue
                
                # Encode chains once per word
                encoded_chains = [self.encode_suffix_chain(chain) for chain in suffix_chains]
                
                # Add multiple training examples (one per correct index)
                for idx in correct_indices:
                    training_data.append(([], encoded_chains, idx))
            
            except Exception as e:
                skipped += 1
                skip_logs['exception'].append({
                    'word': word,
                    'error': str(e),
                    'logged_entries': correct_entries
                })
        
        # Write skip logs to files
        import json
        import os
        
        logs_dir = 'skip_logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        for reason, logs in skip_logs.items():
            if logs:
                filepath = os.path.join(logs_dir, f'{reason}.jsonl')
                with open(filepath, 'w', encoding='utf-8') as f:
                    for log_entry in logs:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                print(f"📝 {reason}: {len(logs)} entries logged to {filepath}")
        
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

    # In interactive_trainer.py, replace the text_mode method:

    def text_mode(self):
        """
        🚀 Multiprocessing Text Mode 
        Uses ALL CPU cores to decompose words, then GPU/CPU for Batch AI ranking.
        """
        print(f"\nExample: 'sample.txt'")
        filename = input("Enter filename (default: sample.txt): ").strip()
        if not filename: filename = "sample.txt"
        
        # 1. Load Text
        try:
            # Assuming get_text_tokenized can take a filename or uses a default
            # If your data_manager doesn't support args, you might need to set it first
            text = self.data_manager.get_text_tokenized() 
        except Exception as e:
            print(f"❌ Error loading text: {e}")
            return

        if not text:
            print("⚠️  No text found")
            return
            
        print(f"   📊 Input: {len(text)} tokens")
        
        # 2. Deduplication (Critical optimization)
        unique_words = list(set(text))
        print(f"   🔍 Unique words: {len(unique_words)} (Reduction: {100 - len(unique_words)/len(text)*100:.1f}%)")
        
        # 3. Parallel Decomposition (The Speedup)
        # We use all available cores minus 1 (to keep system responsive)
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"   🔥 Spawning {num_cores} worker processes for decomposition...")
        
        word_results = {} # Maps word -> decompositions
        
        # ProcessPoolExecutor handles the heavy lifting
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # map returns results in the same order as inputs
            # chunksize helps reduce communication overhead
            results = executor.map(self.data_manager.decompose, unique_words, chunksize=100)
            
            for word, decompositions in zip(unique_words, results):
                word_results[word] = decompositions

        # 4. Filter Ambiguous Words for AI
        print("   🤖 Preparing AI Batch...")
        
        cache = {} # Final lookup table {word: formatted_string}
        ambiguous_batches = [] # [ (word_index, encoded_chains) ]
        
        for i, word in enumerate(unique_words):
            decomps = word_results[word]
            
            if not decomps:
                cache[word] = word
            elif len(decomps) == 1:
                cache[word] = self.display.format_decomposition(word, decomps[0], simple=True)
            else:
                # Prepare for ML
                suffix_chains = [chain for _, _, chain, _ in decomps]
                encoded_chains = [self.encode_suffix_chain(chain) for chain in suffix_chains]
                
                # Store tuple: (index_in_unique_words, chains)
                ambiguous_batches.append((i, encoded_chains))
                # Placeholder
                cache[word] = ("__WAITING__", decomps)

        # 5. Batch AI Inference
        if ambiguous_batches:
            print(f"   ⚡ Ranking {len(ambiguous_batches)} ambiguous words on device...")
            
            # Extract just the chains for the predictor
            just_chains = [x[1] for x in ambiguous_batches]
            
            # Use the batch_predict method we added to ml_ranking_model.py
            # If you haven't added it yet, standard predict loop is fine here 
            # but batch_predict is 10x faster.
            try:
                batch_predictions = self.trainer.batch_predict(just_chains)
            except AttributeError:
                print("   ⚠️  Trainer missing batch_predict, falling back to slow loop...")
                batch_predictions = []
                for chains in just_chains:
                     batch_predictions.append(self.trainer.predict(chains))

            # Apply predictions
            for idx, (original_idx, _) in enumerate(ambiguous_batches):
                best_idx_in_decomp = batch_predictions[idx][0]
                
                word = unique_words[original_idx]
                _, decomps = cache[word] # Retrieve waiting data
                
                # Safety check
                if best_idx_in_decomp >= len(decomps): best_idx_in_decomp = 0
                
                best_decomp = decomps[best_idx_in_decomp]
                cache[word] = self.display.format_decomposition(word, best_decomp, simple=True)

        # 6. Reconstruct Text
        print("   📝 Reconstructing full text...")
        final_output = []
        for word in text:
            final_output.append(cache.get(word, word))
            
        output_text = '\n'.join(final_output)
        self.data_manager.write_decomposed_text(output_text)
        
        print(f"\n✅ Done! Processed {len(text)} words.")
        print(f"   💾 Saved to: {self.data_manager.get_decomposed_text_path()}")

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