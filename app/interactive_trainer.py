import concurrent.futures
import multiprocessing
from typing import List, Optional, Tuple, Dict, Any

# Internal Project Imports
import app.display as display
from app.data_manager import DataManager
from ml.ml_ranking_model import Ranker, Trainer
import util.decomposer as sfx
from util.suffix import Suffix


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
        """Trigger a save."""
        self.trainer.save_checkpoint()
        with open(self.data_manager.paths.training_count_path, "w") as f:
            f.write(str(self.training_count))
        print(f"Model saved")
    
    def train_mode(self, word: str) -> Optional[bool]:
        """
        Main logic for processing a single word:
        Decompose -> Predict -> User Choice -> Train -> Save
        """
        decompositions = sfx.decompose(word)

        if not decompositions:
            print(f"\n  No decompositions found for '{word}'")
            return False
        
        # 1. Get Predictions if available
        scores = None
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        if len(suffix_chains) == 1:
            print(f"\n Only one decomposition for '{word}' - skipping")
            return False

        if self.training_count > 0:
            try:
                encoded_chains_pred = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
                _, scores = self.trainer.predict(encoded_chains_pred)
                print(f"\n ML Model predictions shown")
            except Exception:
                pass
        
        # 2. Sort and Prepare Data for Display
        # Combine decomps with scores (default to 0.0 if None)
        scored_decomps = []
        for i, decomp in enumerate(decompositions):
            score = scores[i] if scores else 0.0
            scored_decomps.append((score, decomp))
        
        # Sort by score (ascending: lower is better usually, or descending depending on loss)
        # Assuming lower score/distance is better
        if scores:
            scored_decomps.sort(key=lambda x: x[0])

        # Prepare View Models (Strings) using Translation class
        view_models = []
        sorted_decomps = [] # Keep track of the actual objects in sorted order
        
        translator = Translation()
        for score, decomp in scored_decomps:
            vm = translator.reconstruct_morphology(word, decomp)
            vm['score'] = score if scores else None
            view_models.append(vm)
            sorted_decomps.append(decomp)

        # 3. Show Display
        display.show_decompositions(word, view_models)
        
        # 4. Get User Input
        # choices are 0-based indices referring to the *sorted* list we just displayed
        choices = display.get_user_choices(len(sorted_decomps))
        
        if choices is None: # User typed 'q'
            return None
        if choices == [-1]: # User typed 's'
            return False
        
        # Map sorted indices back to actual decompositions
        correct_decomps = [sorted_decomps[i] for i in choices]
        
        # We need original indices for DataManager? 
        # Actually DataManager.log_decompositions usually expects the full list and indices, 
        # but simpler to just pass the objects if we refactor DataManager, 
        # OR we just find the index of the chosen object in the original unsorted list.
        # Let's map back to original indices for consistency with existing data manager
        original_indices = []
        for correct_d in correct_decomps:
            original_indices.append(decompositions.index(correct_d))

        # Logging & Cleanup
        self.data_manager.log_decompositions(word, original_indices, decompositions)
        self.data_manager.delete_word_if_root_exists(word, original_indices, decompositions)
        
        # --- Training Logic ---
        encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
        
        # Train on the Original Indices (because encoded_chains matches original order)
        training_data = [
            ([], encoded_chains, idx) for idx in original_indices
        ]
        
        loss = self.trainer.train_persistent(training_data)
        print(f"\n Training complete. Loss: {loss:.4f}")

        self.training_count += 1
        print(f"Total examples: {self.training_count}")
        
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
        
        return True
    
    def evaluation_mode(self, word: str):
        """Evaluate model on a word without training"""
        decompositions = sfx.decompose(word)
        if not decompositions:
            print(f"\n  No decompositions found")
            return
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        try:
            encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
            pred_idx, scores = self.trainer.predict(encoded_chains)
            
            print(f"\n🤖 ML Model's top prediction:")
            
            # Prepare view model for the single best result
            best_decomp = decompositions[pred_idx]
            vm = Translation().reconstruct_morphology(word, best_decomp)
            vm['score'] = scores[pred_idx]
            
            # Use display's single option printer (accessed via list for simplicity)
            display.show_decompositions(word, [vm])

        except Exception as e:
            print(f"\n Error: {e}")

    def relearn_mode(self):
        """Train on all logged valid decompositions."""
        entries = self.data_manager.get_valid_decomps()
        
        word_groups = {}
        for entry in entries:
            word_groups.setdefault(entry['word'], []).append(entry)
        
        training_data = []
        skipped = 0
        
        for word, correct_entries in word_groups.items():
            try:
                decompositions = sfx.decompose(word)
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
        
        history = self.trainer.train(training_data, num_epochs=5, verbose=True)
        final_loss = history['train_history'][-1] if history['train_history'] else 0.0
        
        self.training_count += len(training_data)
        display.show_batch_summary(len(training_data), skipped, final_loss, self.training_count)
        self.save()

    def auto_words_mode(self):
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
                result = self.train_mode(word)
                
                if result is None: # User Quit
                    auto_stats['words_processed'] -= 1 
                    print("\n Exiting auto mode...")
                    break
                
                if result is False: # Skipped
                    auto_stats['words_skipped'] += 1
                
                if self.data_manager.exists(word.lower()) == 0: 
                    auto_stats['words_deleted'] += 1
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n Error: {e}")
                break
        
        display.show_auto_summary(auto_stats)

    def sample_mode(self):
        """
        Decompose a full text file using Multiprocessing, then use Batch AI to rank.
        """
        print(f"\nExample: 'sample.txt'")
        filename = input("Enter filename (default: sample.txt): ").strip()
        if not filename: filename = "sample.txt"
        
        try:
            text = self.data_manager.get_text_tokenized()
        except Exception as e:
            print(f" Error loading text: {e}")
            return

        if not text:
            print("  No text found")
            return
            
        print(f"    Input: {len(text)} tokens")
        
        unique_words = list(set(text))
        print(f"    Unique words: {len(unique_words)}")
        
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"    Spawning {num_cores} worker processes...")
        
        word_results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = executor.map(sfx.decompose, unique_words, chunksize=100)
            for word, decompositions in zip(unique_words, results):
                word_results[word] = decompositions

        print("    Preparing AI Batch...")
        cache = {}
        ambiguous_batches = [] 
        
        for i, word in enumerate(unique_words):
            decomps = word_results[word]
            if not decomps:
                cache[word] = word
            elif len(decomps) == 1:
                # Use simplified format logic here or call Translation if strictly needed
                cache[word] = display.format_decomposition(word, decomps[0], simple=True)
            else:
                suffix_chains = [chain for _, _, chain, _ in decomps]
                encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
                ambiguous_batches.append((i, encoded_chains))
                cache[word] = ("__WAITING__", decomps)

        if ambiguous_batches:
            print(f"    Ranking {len(ambiguous_batches)} ambiguous words...")
            just_chains = [x[1] for x in ambiguous_batches]
            
            batch_predictions = self.trainer.batch_predict(just_chains)

            for idx, (original_idx, _) in enumerate(ambiguous_batches):
                best_idx_in_decomp = batch_predictions[idx][0]
                word = unique_words[original_idx]
                _, decomps = cache[word]
                
                if best_idx_in_decomp >= len(decomps): best_idx_in_decomp = 0
                best_decomp = decomps[best_idx_in_decomp]
                
                cache[word] = display.format_decomposition(word, best_decomp, simple=True)

        print("    Reconstructing...")
        final_output = []
        for word in text:
            final_output.append(cache.get(word, word))
            
        output_text = '\n'.join(final_output)
        self.data_manager.write_decomposed_text(output_text)
        print("    Done.")

    def stat_mode(self):
        """Display comprehensive training statistics"""
        print(f"\n Training Statistics:")
        print(f"  Total examples: {self.trainer.training_count}")
        
        if self.trainer.trainer.training_history:
            recent = self.trainer.trainer.training_history[-20:]
            print(f"  Recent avg loss: {sum(recent)/len(recent):.4f}")
            print(f"  Latest loss: {self.trainer.trainer.training_history[-1]:.4f}")
        
        if self.trainer.trainer.validation_history:
            print(f"  Best validation: {self.trainer.trainer.best_val_loss:.4f}")
        
        if self.trainer.auto_stats['words_processed'] > 0:
            print(f"\n  Auto mode:")
            print(f"    - Processed: {self.trainer.auto_stats['words_processed']}")
            print(f"    - Deleted: {self.trainer.auto_stats['words_deleted']}")
            print(f"    - Skipped: {self.trainer.auto_stats['words_skipped']}")
        
        print(f"\n  Model config:")
        print(f"    - Architecture: {'LSTM' if self.trainer.model.use_lstm else 'Transformer'}")
        print(f"    - Loss: {'Triplet' if self.trainer.trainer.use_triplet_loss else 'Contrastive'}")
        print(f"    - Batch size: {self.trainer.trainer.batch_size}")

    def menu(self):
        """Main entry point."""
        display.welcome()
        
        while True:
            try:
                cmd = input("\n Enter word or command: ").strip().lower()
                if not cmd: continue

                if cmd == 'quit':
                    if self.training_count > 0 and display.confirm_save():
                        self.save()
                    break
                elif cmd == 'save':
                    self.save()
                elif cmd == 'stats':
                    self.stat_mode()
                elif cmd == 'auto':
                    self.auto_words_mode()
                elif cmd == 'sample':
                    self.sample_mode()
                elif cmd == 'relearn':
                    self.relearn_mode()
                elif cmd.startswith('eval '):
                    self.evaluation_mode(cmd[5:].strip())
                else:
                    result = self.train_mode(cmd)
                    if result is None and display.confirm_save():
                        self.save()
                        break
            
            except KeyboardInterrupt:
                if display.confirm_save():
                    self.save()
                break
            except Exception as e:
                print(f"\n Error: {e}")


class Translation:
    """Helper class for translating between Suffix objects, Strings, and ML Vectors."""
    
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
        """Convert Suffix objects into (suffix_id, category_id) tuples."""
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

    def reconstruct_morphology(self, word: str, decomposition: Tuple) -> Dict[str, Any]:
        """
        Reconstructs the surface forms of the suffixes based on the root and chain.
        Returns a dict ready for display.
        """
        root, pos, chain, final_pos = decomposition
        
        if not chain:
            verb_marker = "-" if pos == "verb" else ""
            return {
                'root_str': f"{root} ({pos})",
                'final_pos': final_pos,
                'has_chain': False,
                'formation_str': f"{root}{verb_marker} (no suffixes)"
            }
        
        # --- Logic Moved from display.py ---
        current_stem = root
        suffix_forms = []
        suffix_names = []
        formation = [root + ("-" if pos == "verb" else "")]
        
        cursor = len(root)
        start_idx = 0
        
        # 1. HANDLE SPECIAL PREFIX CASE (PEKISTIRME)
        if chain and chain[0].name == "pekistirme":
            root_idx = word.find(root)
            if root_idx > 0:
                prefix_str = word[:root_idx]
                suffix_forms.append(prefix_str)
                suffix_names.append(chain[0].name)
                current_stem = prefix_str + root
                formation.append(current_stem)
                cursor = root_idx + len(root)
                start_idx = 1

        # 2. HANDLE STANDARD SUFFIXES (Initial Check)
        if start_idx == 0:
            if not word.startswith(root) and chain:
                first_suffix = chain[0]
                possible_forms = first_suffix.form(root)
                match_found = False
                for offset in range(3): 
                    test_cursor = len(root) - offset
                    if test_cursor <= 0: break
                    rest_of_word = word[test_cursor:]
                    for form in possible_forms:
                        if rest_of_word.startswith(form):
                            cursor = test_cursor 
                            match_found = True
                            break
                    if match_found: break

        # 3. MAIN LOOP
        for i in range(start_idx, len(chain)):
            suffix_obj = chain[i]
            possible_forms = suffix_obj.form(current_stem)
            found_form = None 
            
            # A. Exact Match
            for form in possible_forms:
                if word.startswith(form, cursor):
                    found_form = form
                    break
            
            # B. Narrowing Match (Extended Lookahead for -iyor)
            if found_form is None:
                has_iyor_ahead = False
                for k in range(i + 1, len(chain)):
                    if "iyor" in chain[k].name:
                        has_iyor_ahead = True
                        break
                
                if has_iyor_ahead:
                    for form in possible_forms:
                        if form and form[-1] in ['a', 'e']:
                            shortened = form[:-1]
                            if word.startswith(shortened, cursor):
                                found_form = shortened
                                break

            # C. Soft Match
            if found_form is None:
                for form in possible_forms:
                    if len(form) > 0 and word.startswith(form, cursor - 1):
                        found_form = form
                        cursor -= 1 
                        break
            
            if found_form is None:
                if possible_forms:
                    guessed_len = len(possible_forms[0])
                    suffix_forms.append(possible_forms[0] + "?")
                    suffix_names.append(suffix_obj.name)
                    current_stem += possible_forms[0]
                    cursor += guessed_len
                continue
            
            suffix_forms.append(found_form if found_form else "(ø)")
            suffix_names.append(suffix_obj.name)
            current_stem += found_form
            cursor += len(found_form)
            
            verb_marker = "-" if suffix_obj.makes.name == "Verb" else ""
            formation.append(current_stem + verb_marker)
            
        return {
            'root_str': f"{root} ({pos})",
            'final_pos': final_pos,
            'has_chain': True,
            'suffixes_str': ' + '.join(suffix_forms),
            'names_str': ' + '.join(suffix_names),
            'formation_str': ' → '.join(formation)
        }