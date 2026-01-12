from typing import List, Optional, Tuple, Dict

class TrainerDisplay:
    """Handles all user interaction and display formatting"""
    
    @staticmethod
    def welcome():
        """Show welcome message and available commands"""
        print("\n Commands:")
        print("  - Enter a word to analyze and train")
        print("  - 'auto' - Start auto mode (random words from dictionary)")
        print("  - 'eval <word>' - Evaluate model on a word")
        print("  - 'batch' - Train on all logged decompositions")
        print("  - 'stats' - Show training statistics")
        print("  - 'text' - Analyze a text.")
        print("  - 'save' - Save model")
        print("  - 'quit' - Exit")
    

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
            TrainerDisplay.show_single_decomposition(
                word, decomp, score, display_idx, orig_idx
            )
            index_mapping[display_idx] = orig_idx
        
        return index_mapping
    
    @staticmethod
    def show_single_decomposition(word: str, decomp: Tuple, score: Optional[float],
                                   display_idx: int, orig_idx: int):
        """Format and print a single decomposition"""
        root, pos, chain, final_pos = decomp
        
        print(f"\n[Option {display_idx}]")
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
        start_idx = 0
        
        # --- 1. HANDLE SPECIAL PREFIX CASE (PEKISTIRME) ---
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
            else:
                pass

        # --- 2. HANDLE STANDARD SUFFIXES ---
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

        # --- 3. MAIN LOOP ---
        for i in range(start_idx, len(chain)):
            suffix_obj = chain[i]
            possible_forms = suffix_obj.form(current_stem)
            found_form = None 
            
            # A. Try Standard Exact Match
            for form in possible_forms:
                if word.startswith(form, cursor):
                    found_form = form
                    break
            
            # B. Try Narrowing Match (Extended Lookahead)
            # Check if this suffix (or a sequence of suffixes) is followed by -iyor
            if found_form is None:
                has_iyor_ahead = False
                # Scan future suffixes to see if iyor is coming
                for k in range(i + 1, len(chain)):
                    if "iyor" in chain[k].name:
                        has_iyor_ahead = True
                        break
                
                if has_iyor_ahead:
                     for form in possible_forms:
                         # Allow dropping 'a'/'e' (me -> m, or e -> ø)
                         if form and form[-1] in ['a', 'e']:
                             shortened = form[:-1]
                             if word.startswith(shortened, cursor):
                                 found_form = shortened
                                 break

            # C. Try Soft Match (Retry with 1 char overlap/deletion)
            if found_form is None:
                 for form in possible_forms:
                     if len(form) > 0 and word.startswith(form, cursor - 1):
                         found_form = form
                         cursor -= 1 
                         break
            
            if found_form is None:
                # print(f"[Warning: Could not match suffix '{suffix_obj.name}']")
                if possible_forms:
                     guessed_len = len(possible_forms[0])
                     suffix_forms.append(possible_forms[0] + "?")
                     suffix_names.append(suffix_obj.name)
                     current_stem += possible_forms[0]
                     cursor += guessed_len
                continue
            
            # Add found suffix
            suffix_forms.append(found_form if found_form else "(ø)")
            suffix_names.append(suffix_obj.name)
            
            current_stem += found_form
            cursor += len(found_form)
            
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
    def show_statistics(trainer):
        """Display comprehensive training statistics"""
        print(f"\n Training Statistics:")
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
        print(f" Processed: {stats['words_processed']}")
        print(f" Deleted: {stats['words_deleted']}")
        print(f" Skipped: {stats['words_skipped']}")
        print(f"{'='*70}\n")
    
    @staticmethod
    def show_batch_summary(trained: int, skipped: int, final_loss: float, total: int):
        """Show batch training summary"""
        print(f"\n{'='*70}")
        print(f"BATCH TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f" Trained: {trained} examples")
        print(f" Skipped: {skipped} words")
        print(f" Final loss: {final_loss:.4f}")
        print(f" Total examples: {total}")
        print(f"{'='*70}\n")

    
    def format_decomposition(self, word: str, decomposition: Tuple, simple: bool = False) -> str:
        """Format a decomposition into readable text."""
        root, pos, chain, final_pos = decomposition
        
        if not chain:
            return root
        
        if simple:
            suffix_names = [suffix.name for suffix in chain]
            result = root + '+' + '+'.join(suffix_names)
            return result
        
        suffix_parts = []
        current_word = root
        
        # Bu fonksiyon sadece düz metin formatı için, görselleştirme kadar kritik değil
        # ama basit bir form seçimi yapar.
        for suffix_obj in chain:
            # Check for pekistirme to avoid crashing on dummy form
            if suffix_obj.name == "pekistirme":
                # For plain text format, we might just assume standard 'mas' style or skip detailed form logic
                # Just use the name to keep it safe
                suffix_parts.append(f"{suffix_obj.name}")
                continue

            forms = suffix_obj.form(current_word)
            used_form = forms[0] if forms else suffix_obj.suffix
            
            suffix_str = f"{suffix_obj.name}_{used_form}"
            suffix_parts.append(suffix_str)
            
            current_word += used_form
        
        result = root + '+' + '+'.join(suffix_parts)
        return result