import sys
import os
import itertools
import concurrent.futures
import multiprocessing
from typing import List, Optional, Tuple, Dict, Any

try:
    import msvcrt
    def getch():
        return msvcrt.getwch()
except ImportError:
    import tty, termios
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

import app.display as display
from app.data_manager import DataManager
from ml.ml_ranking_model import Ranker, Trainer
import util.decomposer as sfx
import util.word_methods as wrd 
from util.suffix import Suffix

class InteractiveTrainer:
    """
    Handles the interactive training loop.
    Contains logic for formatting logs, validating dictionary rules, and processing sentences.
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        self.model = Ranker(suffix_vocab_size=len(sfx.ALL_SUFFIXES))
        self.trainer = Trainer(model=self.model)
        self.training_count = self.data_manager.load_training_count()

    def save(self):
        self.trainer.save_checkpoint()
        with open(self.data_manager.paths.training_count_path, "w") as f:
            f.write(str(self.training_count))
        print(f"Model saved")
    
    def train_mode(self, word: str) -> Optional[bool]:
        decompositions = sfx.decompose(word)

        if not decompositions:
            print(f"\n  No decompositions found for '{word}'")
            return False
        
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
        
        scored_decomps = []
        for i, decomp in enumerate(decompositions):
            score = scores[i] if scores else 0.0
            scored_decomps.append((score, decomp))
        
        if scores:
            scored_decomps.sort(key=lambda x: x[0])

        view_models = []
        sorted_decomps = []
        
        translator = Translation()
        for score, decomp in scored_decomps:
            vm = translator.reconstruct_morphology(word, decomp)
            vm['score'] = score if scores else None
            view_models.append(vm)
            sorted_decomps.append(decomp)

        display.show_decompositions(word, view_models)
        
        choices = display.get_user_choices(len(sorted_decomps))
        
        if choices is None: return None
        if choices == [-1]: return False
        
        correct_decomps = [sorted_decomps[i] for i in choices]
        original_indices = [decompositions.index(d) for d in correct_decomps]

        log_entries = []
        for decomp in correct_decomps:
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
            
            log_entries.append({
                'word': word,
                'root': root,
                'suffixes': suffix_info,
                'final_pos': final_pos
            })
            
        self.data_manager.log_decompositions(log_entries)

        word_lower = word.lower()
        for decomp in correct_decomps:
            root = decomp[0].lower()
            if root == word_lower:
                continue
            
            if wrd.can_be_noun(root) or wrd.can_be_verb(root):
                if self.data_manager.delete(word_lower):
                    print(f"  Deleted '{word}' (root '{root}' exists)")
                    
                    infinitive_form = wrd.infinitive(word_lower)
                    if infinitive_form and infinitive_form != word_lower:
                        if self.data_manager.delete(infinitive_form):
                            print(f"  Deleted infinitive '{infinitive_form}'")
        
        encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
        
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

    def _format_aligned_sentence(self, words: List[str], parts: List[str]) -> str:
        """Aligns words and their decomposed parts into two padded rows."""
        top_line = ""
        bot_line = ""
        for w, p in zip(words, parts):
            width = max(len(w), len(p))
            top_line += w.ljust(width) + "   "
            bot_line += p.ljust(width) + "   "
        return f"{top_line.strip()}\n    {bot_line.strip()}"

    def sentence_train_mode(self, sentence: str) -> Optional[bool]:
        words = sentence.strip().split()
        if not words:
            return False

        word_data = []
        for word in words:
            decomps = sfx.decompose(word)
            if not decomps:
                print(f"\nNo decompositions found for '{word}'. Skipping sentence.")
                return False
            word_data.append({'word': word, 'decomps': decomps})

        translator = Translation()
        all_chains_to_predict = []
        
        for wd in word_data:
            suffix_chains = [chain for _, _, chain, _ in wd['decomps']]
            encoded_chains = [translator.encode_suffix_chain(chain) for chain in suffix_chains]
            all_chains_to_predict.append(encoded_chains)

        batch_results = self.trainer.batch_predict(all_chains_to_predict)

        for i, wd in enumerate(word_data):
            wd['scores'] = batch_results[i][1]
            wd['vms'] = []
            wd['typing_strings'] = []
            
            for decomp in wd['decomps']:
                vm = translator.reconstruct_morphology(wd['word'], decomp)
                wd['vms'].append(vm)
                
                root = decomp[0]
                if vm['has_chain']:
                    suffixes = vm['suffixes_str'].replace(" + ", " ")
                    wd['typing_strings'].append(f"{root} {suffixes}")
                else:
                    wd['typing_strings'].append(root)

        decomp_indices = [list(range(len(wd['decomps']))) for wd in word_data]
        all_combinations = list(itertools.product(*decomp_indices))
        
        all_sentences = []
        for combo in all_combinations:
            total_score = sum(word_data[w_idx]['scores'][d_idx] for w_idx, d_idx in enumerate(combo))
            text_parts = [word_data[w_idx]['typing_strings'][d_idx] for w_idx, d_idx in enumerate(combo)]
            text_str = " ".join(text_parts)
            all_sentences.append({
                'score': total_score,
                'combo_indices': combo,
                'text': text_str,
                'parts': text_parts
            })

        all_sentences.sort(key=lambda x: x['score'], reverse=True)
        top_k_sentences = all_sentences[:10]

        current_input = ""
        correct_combo = None
        locked_in = False
        display_list = top_k_sentences

        while True:
            if not locked_in:
                matches = [c for c in all_sentences if c['text'].startswith(current_input)]
                display_list = top_k_sentences if current_input == "" else matches[:10]
                
                if len(matches) == 1 and current_input != "":
                    correct_combo = matches[0]['combo_indices']
                    print(f"\nAuto-selected: {matches[0]['text']}")
                    break
                    
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Sentence: {sentence}\n")
            
            if not display_list:
                print("No matching decompositions found for your input.")
            else:
                print("Predictions:" if not locked_in else "Locked Predictions (Select by number):")
                for i, c in enumerate(display_list):
                    aligned_display = self._format_aligned_sentence(words, c['parts'])
                    vms = [word_data[w_idx]['vms'][d_idx] for w_idx, d_idx in enumerate(c['combo_indices'])]
                    
                    display_idx = i if locked_in or current_input == "" else "-"
                    display.show_sentence_prediction(display_idx, c['score'], words, vms, aligned_display)
                    
            if locked_in:
                print(f"\nLocked on: '{current_input}'")
                print(f"Press 0-{len(display_list)-1} to select, or Backspace to edit: ", end="", flush=True)
            else:
                print(f"\nType decomposition (Press Enter to lock options): {current_input}", end="", flush=True)
            
            ch = getch()
            
            if ch in ('\r', '\n'):
                if not locked_in and display_list:
                    locked_in = True
            elif ch in ('\x08', '\x7f'):
                if locked_in:
                    locked_in = False
                else:
                    current_input = current_input[:-1]
            elif ch == '\x03': 
                return None
            elif ch == '\x1b': 
                return False
            elif locked_in and ch.isdigit():
                idx = int(ch)
                if idx < len(display_list):
                    correct_combo = display_list[idx]['combo_indices']
                    break
            elif not locked_in:
                if current_input == "" and ch.isdigit():
                    idx = int(ch)
                    if idx < len(display_list):
                        correct_combo = display_list[idx]['combo_indices']
                        break
                else:
                    current_input += ch

        if not correct_combo:
            return False

        training_data = []
        log_entries = []
        
        for w_idx, correct_d_idx in enumerate(correct_combo):
            wd = word_data[w_idx]
            word = wd['word']
            decomps = wd['decomps']
            typing_str = wd['typing_strings'][correct_d_idx]
            
            suffix_chains = [chain for _, _, chain, _ in decomps]
            encoded_chains = [translator.encode_suffix_chain(chain) for chain in suffix_chains]
            training_data.append(([], encoded_chains, correct_d_idx))
            
            root, pos, chain, final_pos = decomps[correct_d_idx]
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
                    
            log_entries.append({
                'word': word,
                'morphology_string': typing_str,
                'root': root,
                'suffixes': suffix_info,
                'final_pos': final_pos
            })

        self.data_manager.log_sentence_decompositions(log_entries, sentence)
        
        print("\nTraining...")
        loss = self.trainer.train_persistent(training_data)
        print(f"Average word loss: {loss:.4f}")

        self.training_count += len(training_data)
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
            
        return True
    
    def evaluation_mode(self, word: str):
        decompositions = sfx.decompose(word)
        if not decompositions:
            print(f"\n  No decompositions found")
            return
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        
        try:
            encoded_chains = [Translation().encode_suffix_chain(chain) for chain in suffix_chains]
            pred_idx, scores = self.trainer.predict(encoded_chains)
            
            print(f"\n ML Model's top prediction:")
            
            best_decomp = decompositions[pred_idx]
            vm = Translation().reconstruct_morphology(word, best_decomp)
            vm['score'] = scores[pred_idx]
            
            display.show_decompositions(word, [vm])

        except Exception as e:
            print(f"\n Error: {e}")

    def relearn_mode(self):
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
                    print("\n No more words!")
                    break
                
                auto_stats['words_processed'] += 1
                result = self.train_mode(word)
                
                if result is None:
                    auto_stats['words_processed'] -= 1 
                    print("\n Exiting auto mode...")
                    break
                
                if result is False: 
                    auto_stats['words_skipped'] += 1
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n Error: {e}")
                break
        
        display.show_auto_summary(auto_stats)

    def sample_mode(self):
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
        print(f"\n Training Statistics:")
        print(f"  Total examples: {self.trainer.training_count}")
        
        if self.trainer.trainer.training_history:
            recent = self.trainer.trainer.training_history[-20:]
            print(f"  Recent avg loss: {sum(recent)/len(recent):.4f}")
            print(f"  Latest loss: {self.trainer.trainer.training_history[-1]:.4f}")
        
        if self.trainer.trainer.validation_history:
            print(f"  Best validation: {self.trainer.trainer.best_val_loss:.4f}")

    def menu(self):
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
                elif cmd.startswith('sentence '):
                    result = self.sentence_train_mode(cmd[9:].strip())
                    if result is None and display.confirm_save():
                        self.save()
                        break
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
        root, pos, chain, final_pos = decomposition
        if not chain:
            verb_marker = "-" if pos == "verb" else ""
            return {
                'root_str': f"{root} ({pos})",
                'final_pos': final_pos,
                'has_chain': False,
                'formation_str': f"{root}{verb_marker} (no suffixes)"
            }
        
        current_stem = root
        suffix_forms = []
        suffix_names = []
        formation = [root + ("-" if pos == "verb" else "")]
        
        cursor = len(root)
        start_idx = 0
        
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

        for i in range(start_idx, len(chain)):
            suffix_obj = chain[i]
            possible_forms = suffix_obj.form(current_stem)
            found_form = None 
            
            for form in possible_forms:
                if word.startswith(form, cursor):
                    found_form = form
                    break
            
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