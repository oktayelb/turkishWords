import concurrent.futures
import multiprocessing
from typing import List, Optional, Tuple, Dict, Any

import util.decomposer as sfx
import util.word_methods as wrd 
from ml.ml_ranking_model import SentenceDisambiguator, Trainer
from app.data_manager import DataManager
import app.morphology_adapter as morph
from app.sequence_matcher import find_matching_combinations

class WorkflowEngine:
    def __init__(self):
        self.data_manager = DataManager()
        self.model = SentenceDisambiguator(suffix_vocab_size=len(sfx.ALL_SUFFIXES))
        self.trainer = Trainer(model=self.model)
        self.training_count = self.data_manager.load_training_count()
        self.decomp_cache = {}

    def get_decompositions(self, word: str) -> List[Tuple]:
        if word not in self.decomp_cache:
            self.decomp_cache[word] = sfx.decompose(word)
        return self.decomp_cache[word]

    def save(self):
        self.trainer.save_checkpoint()
        with open(self.data_manager.paths.training_count_path, "w") as f:
            f.write(str(self.training_count))
        print("Model saved")

    def prepare_word_training(self, word: str) -> Optional[Dict[str, Any]]:
        decompositions = self.get_decompositions(word)
        if not decompositions:
            return None
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        if len(suffix_chains) == 1:
            return {'single_decomposition': True}

        encoded_chains = [morph.encode_suffix_chain(chain) for chain in suffix_chains]

        scores = None
        if self.training_count > 0:
            try:
                _, scores = self.trainer.predict(encoded_chains)
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
        
        for score, decomp in scored_decomps:
            vm = morph.reconstruct_morphology(word, decomp)
            vm['score'] = score if scores else None
            view_models.append(vm)
            sorted_decomps.append(decomp)

        return {
            'single_decomposition': False,
            'view_models': view_models,
            'sorted_decomps': sorted_decomps,
            'encoded_chains': encoded_chains,
            'has_scores': bool(scores),
            'original_decompositions': decompositions
        }

    def commit_word_training(self, word: str, correct_decomps: List[Tuple], encoded_chains: List[List[Tuple[int, int]]], original_indices: List[int]) -> float:
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
                        'makes': suffix.makes.name,
                    })
                    current += used_form
            log_entries.append({
                'word': word,
                'root': root,
                'suffixes': suffix_info,
                'final_pos': final_pos,
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

        loss = 0.0
        for idx in original_indices:
            confirmed_chain = encoded_chains[idx]
            loss = self.trainer.train_sentence([confirmed_chain])

        self.training_count += 1
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
        return loss

    def prepare_sentence_training(self, sentence: str) -> Optional[List[Dict]]:
        words = sentence.strip().split()
        if not words:
            return None

        word_data = []
        for word in words:
            decomps = self.get_decompositions(word)
            if not decomps:
                print(f"No decompositions found for '{word}'. Skipping sentence.")
                return None
            word_data.append({'word': word, 'decomps': decomps})

        for wd in word_data:
            suffix_chains = [chain for _, _, chain, _ in wd['decomps']]
            wd['encoded_chains'] = [morph.encode_suffix_chain(chain) for chain in suffix_chains]
            wd['vms'] = []
            wd['typing_strings'] = []
            
            for decomp in wd['decomps']:
                vm = morph.reconstruct_morphology(wd['word'], decomp)
                wd['vms'].append(vm)
                root = decomp[0]
                if vm['has_chain']:
                    suffixes = vm['suffixes_str'].replace(" + ", " ")
                    wd['typing_strings'].append(f"{root} {suffixes}")
                else:
                    wd['typing_strings'].append(root)
        return word_data

    def evaluate_sentence_target(self, word_data: List[Dict], target_str: str) -> Tuple[List[Dict], str, int]:
        return find_matching_combinations(word_data, target_str, self.trainer)

    def commit_sentence_training(self, sentence: str, words: List[str], word_data: List[Dict], correct_combo: List[int]) -> float:
        confirmed_chains = []
        log_entries = []
        
        for w_idx, correct_d_idx in enumerate(correct_combo):
            wd = word_data[w_idx]
            word = wd['word']
            decomps = wd['decomps']
            typing_str = wd['typing_strings'][correct_d_idx]
            confirmed_chain = wd['encoded_chains'][correct_d_idx]

            confirmed_chains.append(confirmed_chain)
            
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
                        'makes': suffix.makes.name,
                    })
                    current += used_form
            log_entries.append({
                'word': word,
                'morphology_string': typing_str,
                'root': root,
                'suffixes': suffix_info,
                'final_pos': final_pos,
            })

        self.data_manager.log_sentence_decompositions(log_entries, sentence)
        loss = self.trainer.train_sentence(confirmed_chains)

        self.training_count += len(confirmed_chains)
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
        return loss

    def relearn_all(self) -> Tuple[int, int]:
        entries = self.data_manager.get_valid_decomps()
        
        word_groups = {}
        for entry in entries:
            if entry.get('type') == 'sentence':
                for word_entry in entry.get('words', []):
                    word_groups.setdefault(('__sentence__', id(entry)), []).append(entry)
                    break
            else:
                word_groups.setdefault(entry['word'], []).append(entry)
        
        total_trained = 0
        skipped = 0

        for key, correct_entries in word_groups.items():
            if key[0] == '__sentence__':
                continue  
            word = key
            try:
                decompositions = self.get_decompositions(word)
                if not decompositions:
                    skipped += 1
                    continue
                
                correct_indices = morph.match_decompositions(correct_entries, decompositions)
                if not correct_indices:
                    skipped += 1
                    continue
                
                suffix_chains = [chain for _, _, chain, _ in decompositions]
                encoded_chains = [morph.encode_suffix_chain(chain) for chain in suffix_chains]
                
                for idx in correct_indices:
                    self.trainer.train_sentence([encoded_chains[idx]])
                    total_trained += 1
            except Exception:
                skipped += 1

        sentence_entries = [e for e in entries if e.get('type') == 'sentence']
        for sent_entry in sentence_entries:
            try:
                confirmed_chains = []
                for word_entry in sent_entry.get('words', []):
                    word = word_entry['word']
                    decomps = self.get_decompositions(word)
                    if not decomps:
                        break
                    matched = morph.match_decompositions([word_entry], decomps)
                    if not matched:
                        break
                    suffix_chains = [chain for _, _, chain, _ in decomps]
                    encoded_chains = [morph.encode_suffix_chain(chain) for chain in suffix_chains]
                    confirmed_chains.append(encoded_chains[matched[0]])
                else:
                    if confirmed_chains:
                        self.trainer.train_sentence(confirmed_chains)
                        total_trained += len(confirmed_chains)
            except Exception:
                skipped += 1

        self.training_count += total_trained
        self.save()
        return total_trained, skipped

    def evaluate_word(self, word: str) -> Optional[Dict]:
        decompositions = self.get_decompositions(word)
        if not decompositions:
            return None
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        encoded_chains = [morph.encode_suffix_chain(chain) for chain in suffix_chains]

        try:
            pred_idx, scores = self.trainer.predict(encoded_chains)
            best_decomp = decompositions[pred_idx]
            vm = morph.reconstruct_morphology(word, best_decomp)
            vm['score'] = scores[pred_idx]
            return vm
        except Exception as e:
            print(f"Error evaluating: {e}")
            return None

    def sample_text(self, filename: str) -> bool:
        text = self.data_manager.get_text_tokenized()
        if not text:
            return False
            
        print(f"Input: {len(text)} tokens")
        unique_words = list(set(text))
        print(f"Unique words: {len(unique_words)}")
        
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Spawning {num_cores} worker processes...")
        
        word_results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = executor.map(sfx.decompose, unique_words, chunksize=100)
            for word, decompositions in zip(unique_words, results):
                word_results[word] = decompositions

        print("Preparing AI Batch...")
        cache = {}
        ambiguous_batches = []
        
        for i, word in enumerate(unique_words):
            decomps = word_results[word]
            if not decomps:
                cache[word] = word
            elif len(decomps) == 1:
                cache[word] = self._format_simple_decomp(word, decomps[0])
            else:
                suffix_chains = [chain for _, _, chain, _ in decomps]
                encoded_chains = [morph.encode_suffix_chain(chain) for chain in suffix_chains]
                ambiguous_batches.append((i, encoded_chains))
                cache[word] = ("__WAITING__", decomps)

        if ambiguous_batches:
            print(f"Ranking {len(ambiguous_batches)} ambiguous words...")
            just_chains = [x[1] for x in ambiguous_batches]
            batch_predictions = self.trainer.batch_predict(just_chains)

            for idx, (original_idx, _) in enumerate(ambiguous_batches):
                best_idx_in_decomp = batch_predictions[idx][0]
                word = unique_words[original_idx]
                _, decomps = cache[word]
                if best_idx_in_decomp >= len(decomps):
                    best_idx_in_decomp = 0
                best_decomp = decomps[best_idx_in_decomp]
                cache[word] = self._format_simple_decomp(word, best_decomp)

        print("Reconstructing...")
        final_output = []
        for word in text:
            final_output.append(cache.get(word, word))
            
        output_text = '\n'.join(final_output)
        self.data_manager.write_decomposed_text(output_text)
        print("Done.")
        return True

    def _format_simple_decomp(self, word: str, decomposition: Tuple) -> str:
        root, pos, chain, final_pos = decomposition
        if not chain:
            return root
        suffix_names = [suffix.name for suffix in chain]
        return root + '+' + '+'.join(suffix_names)

    def get_stats(self) -> Dict:
        stats = {
            'total': self.training_count,
            'recent_avg': 0.0,
            'latest': 0.0,
            'best_val': self.trainer.best_val_loss
        }
        if self.trainer.train_history:
            recent = self.trainer.train_history[-20:]
            stats['recent_avg'] = sum(recent)/len(recent)
            stats['latest'] = self.trainer.train_history[-1]
        return stats