
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


def format_decomposition_steps(root, initial_pos, chain, suffix_transitions):
    """Format word formation progressively."""
    root_display = root + "-" if initial_pos == "verb" else root
    if not chain:
        return root_display

    steps = [root_display]
    current_word = root
    for suffix_form, target_pos in suffix_transitions:
        current_word += suffix_form
        word_display = current_word + "-" if target_pos == "verb" else current_word
        steps.append(word_display)
    return " → ".join(steps)


def single_decomposition(root, pos, chain, final_pos, index):
    """Print a single morphological decomposition."""
    print(f"\n[Analysis #{index}]")
    print(f"Root:      {root} ({pos})")

    if chain:
        suffix_forms = [s for s, _ in chain]
        print(f"Suffixes:  {' + '.join(suffix_forms)}")
        print(f"Formation: {format_decomposition_steps(root, pos, suffix_forms, chain)}")
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
            h_suffixes = " + ".join([s for s, _ in h_chain]) if h_chain else "(no suffixes)"
            t_suffixes = " + ".join([s for s, _ in t_chain]) if t_chain else "(no suffixes)"
            h_form = format_decomposition_steps(h_root, h_pos, [s for s, _ in h_chain], h_chain)
            t_form = format_decomposition_steps(t_root, t_pos, [s for s, _ in t_chain], t_chain)

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

