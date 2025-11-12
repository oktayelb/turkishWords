from typing import List, Optional, Tuple, Dict


def format_decomposition_steps(root, initial_pos, suffix_objects):
    """Format word formation progressively."""
    root_display = root + "-" if initial_pos == "verb" else root
    if not suffix_objects:
        return root_display

    steps = [root_display]
    current_word = root
    for suffix_obj in suffix_objects:
        # Generate the actual suffix form for this word, selecting the one that matches
        found_form = None
        for suffix_form in suffix_obj.form(current_word):
            if current_word + suffix_form in steps[-1].replace('-', '') + suffix_form: # Simplified check for display
                found_form = suffix_form
                break
        
        if found_form:
            current_word += found_form
            # Determine if the word should have a dash (verb marker)
            target_pos = "verb" if suffix_obj.makes.name == "Verb" else "noun"
            word_display = current_word + "-" if target_pos == "verb" else current_word
            steps.append(word_display)
        # If not found, it implies an issue or a boundary condition not met in this simple function.
        # We proceed to the next suffix.
        
    return " → ".join(steps)


def single_decomposition(root, pos, chain, final_pos, index):
    """Print a single morphological decomposition."""
    print(f"\n[Analysis #{index}]")
    print(f"Root:      {root} ({pos})")

    if chain:
        # Generate suffix forms for display
        current_word = root
        suffix_forms = []   
        for suffix_obj in chain:
            found_form = None
            for suffix_form in suffix_obj.form(current_word):
                # The logic here is inherently flawed without the full original word,
                # but we'll assume the first generated form is the one used for the next step
                # of the *decomposition* chain for this simple print function.
                # The correct logic is implemented in DecompositionDisplay._build_suffix_info
                found_form = suffix_form
                break
            
            if found_form:
                suffix_forms.append(found_form)
                current_word += found_form
        
        print(f"Suffixes:  {' + '.join(suffix_forms)}")
        print(f"Formation: {format_decomposition_steps(root, pos, chain)}")
    else:
        print(f"Formation: {root + '-' if pos == 'verb' else root} (no suffixes)")
    print(f"Final POS: {final_pos}")
    print("-" * 70)


def compound_decomposition(head_decomp, tail_decomp, index):
    """Print all decomposition combinations for a compound word."""
    print(f"\n[Compound Analysis #{index}]")
    print("=" * 70)
    print(f"{'HEAD COMPONENT':^33} | {'TAIL COMPONENT':^33}")
    print("=" * 70)

    combo_count = 0
    for h_idx, (h_root, h_pos, h_chain, h_final) in enumerate(head_decomp, 1):
        for t_idx, (t_root, t_pos, t_chain, t_final) in enumerate(tail_decomp, 1):
            combo_count += 1
            print(f"\n--- Combination #{combo_count} (Head #{h_idx}, Tail #{t_idx}) ---")
            
            # Generate suffix forms for head
            h_suffix_forms = []
            current_word = h_root
            if h_chain:
                for suffix_obj in h_chain:
                    found_form = None
                    for suffix_form in suffix_obj.form(current_word):
                        found_form = suffix_form
                        break
                    if found_form:
                        h_suffix_forms.append(found_form)
                        current_word += found_form
                h_suffixes = " + ".join(h_suffix_forms)
            else:
                h_suffixes = "(no suffixes)"
            
            # Generate suffix forms for tail
            t_suffix_forms = []
            current_word = t_root
            if t_chain:
                for suffix_obj in t_chain:
                    found_form = None
                    for suffix_form in suffix_obj.form(current_word):
                        found_form = suffix_form
                        break
                    if found_form:
                        t_suffix_forms.append(found_form)
                        current_word += found_form
                t_suffixes = " + ".join(t_suffix_forms)
            else:
                t_suffixes = "(no suffixes)"
            
            h_form = format_decomposition_steps(h_root, h_pos, h_chain)
            t_form = format_decomposition_steps(t_root, t_pos, t_chain)

            print(f"Root:      {h_root:<25} | {t_root:<25}")
            print(f"POS:       {h_pos:<25} | {t_pos:<25}")
            print(f"Suffixes:  {h_suffixes:<25} | {t_suffixes:<25}")
            print(f"Formation: {h_form:<25} | {t_form:<25}")
            print(f"Final POS: {h_final:<25} | {t_final:<25}")
            print("-" * 70)
            print(f"Combined Word → {h_root + t_root}")
    print("=" * 70)
    print(f"Total combinations shown: {combo_count}")
    print("=" * 70)


def welcome():
    """Print welcome message and instructions"""
    print("\n" + "=" * 70)
    print("INTERACTIVE MORPHOLOGICAL ANALYZER TRAINER")
    print("=" * 70)
    print("\nCommands:")
    print("  - Enter a word to analyze and train")
    print("  - 'batch' to train on all valid decompositions from file")
    print("  - 'batch <filepath>' to train from a specific file")
    print("  - 'eval <word>' to evaluate without training")
    print("  - 'save' to save the model")
    print("  - 'stats' to see training statistics")
    print("  - 'quit' to exit")
    print("="*70)


def ml_choices(mode, use_lstm):
    print("\n" + "=" * 60)
    print(f"Configuration:")
    print(f"  - Loss: {mode}")
    print(f"  - Architecture: {'LSTM' if use_lstm else 'Transformer'}")
    print("=" * 60 + "\n")  

class DecompositionDisplay:
    """Handles formatting and display of decompositions"""
    
    @staticmethod
    def format_decomposition(word: str, decomp: Tuple, score: Optional[float] = None, 
                            display_idx: int = 1, original_idx: int = 0) -> str:
        """ Format a single decomposition for display """
        root, pos, chain, final_pos = decomp
        
        lines = []
        lines.append(f"\n[Option {display_idx}] (Original index: {original_idx + 1})")
        
        if score is not None:
            lines.append(f"ML Score: {score:.4f}")
        
        lines.append(f"Root:      {root} ({pos})")
        
        if chain:
            # Pass the full word for accurate suffix form determination
            suffix_forms, suffix_names, formation_steps = DecompositionDisplay._build_suffix_info(
                word, root, pos, chain
            )
            lines.append(f"Suffixes:  {' + '.join(suffix_forms)}")
            lines.append(f"Names:     {' + '.join(suffix_names)}")
            lines.append(f"Formation: {' → '.join(formation_steps)}")
        else:
            lines.append(f"Formation: {root + '-' if pos == 'verb' else root} (no suffixes)")
        
        lines.append(f"Final POS: {final_pos}")
        lines.append("-" * 70)
        
        return print("\n".join(lines))
    
    @staticmethod
    def _build_suffix_info(full_word: str, root: str, pos: str, chain: List) -> Tuple[List[str], List[str], List[str]]:
        """Build suffix forms, names, and formation steps, selecting the correct form."""
        current_stem = root
        suffix_forms = []
        suffix_names = []
        # Initial display step
        formation_steps = [root + ("-" if pos == "verb" else "")]
        
        # Cursor to track where the suffix should begin in the full word
        cursor = len(root)
        
        for suffix_obj in chain:
            found_form = None
            
            # Iterate over possible forms for the current stem
            for suffix_form in suffix_obj.form(current_stem):
                # Check if this form matches the actual string in the full word starting from the cursor
                if full_word.startswith(suffix_form, cursor):
                    found_form = suffix_form
                    break # Found the correct form
            
            if found_form:
                suffix_forms.append(found_form)
                suffix_names.append(suffix_obj.name)
            
                # Update the stem and cursor
                current_stem += found_form
                cursor += len(found_form)
                
                # Update the formation steps display
                target_pos = "verb" if suffix_obj.makes.name == "Verb" else "noun"
                word_display = current_stem + ("-" if target_pos == "verb" else "")
                formation_steps.append(word_display)
            else:
                # This should not happen if the decomposition chain is valid for the full word,
                # but if it does, we stop processing this chain for display.
                print(f"[Warning: Could not match suffix form for '{suffix_obj.name}' after stem '{current_stem}']")
                break
        
        return suffix_forms, suffix_names, formation_steps
    
    @staticmethod
    def display_all(word: str, decompositions: List[Tuple], 
                   scores: Optional[List[float]] = None) -> Dict[int, int]:
        """
        Display all decompositions sorted by score
        Returns mapping from display index to original index
        """

        print(word)

        # Sort by score if available
        if scores:
            indexed_decomps = list(enumerate(zip(decompositions, scores)))
            # Note: Assuming lower score is better, or sorting is handled externally if needed.
            # Sorting by score descending for typically higher-better ML scores:
            indexed_decomps.sort(key=lambda x: x[1][1], reverse=True) 
        else:
            indexed_decomps = [(i, (d, None)) for i, d in enumerate(decompositions)]
        
        # Display each decomposition
        index_mapping = {}
        for display_idx, (original_idx, (decomp, score)) in enumerate(indexed_decomps, 1):
            # The word must be passed here to format_decomposition
            DecompositionDisplay.format_decomposition(
                word, decomp, score, display_idx, original_idx
            )
            index_mapping[display_idx] = original_idx
        
        return index_mapping