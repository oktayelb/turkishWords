def header(title: str):
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)


def subtitle(title: str):
    print("-" * 70)
    print(f"{title}")
    print("-" * 70)


def info_line(label, value):
    print(f"{label:<20} {value}")


def format_decomposition_steps(root, initial_pos, suffix_objects):
    """Format word formation progressively."""
    root_display = root + "-" if initial_pos == "verb" else root
    if not suffix_objects:
        return root_display

    steps = [root_display]
    current_word = root
    for suffix_obj in suffix_objects:
        # Generate the actual suffix form for this word
        suffix_form = suffix_obj.form(current_word)
        current_word += suffix_form
        # Determine if the word should have a dash (verb marker)
        target_pos = "verb" if suffix_obj.makes.name == "Verb" else "noun"
        word_display = current_word + "-" if target_pos == "verb" else current_word
        steps.append(word_display)
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
            suffix_form = suffix_obj.form(current_word)
            suffix_forms.append(suffix_form)
            current_word += suffix_form
        
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
            if h_chain:
                current_word = h_root
                h_suffix_forms = []
                for suffix_obj in h_chain:
                    suffix_form = suffix_obj.form(current_word)
                    h_suffix_forms.append(suffix_form)
                    current_word += suffix_form
                h_suffixes = " + ".join(h_suffix_forms)
            else:
                h_suffixes = "(no suffixes)"
            
            # Generate suffix forms for tail
            if t_chain:
                current_word = t_root
                t_suffix_forms = []
                for suffix_obj in t_chain:
                    suffix_form = suffix_obj.form(current_word)
                    t_suffix_forms.append(suffix_form)
                    current_word += suffix_form
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