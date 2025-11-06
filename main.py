import word_methods as wrd
import suffixes as sfx
import print as display



def find_suffix_chain(word, start_pos, root, visited=None):
    """Recursively find valid suffix chains after a root."""
    if visited is None:
        visited = set()

    state_key = (len(root), start_pos)
    if state_key in visited:
        return []

    visited = visited | {state_key}
    rest = word[len(root):]

    # Base case: fully matched word
    if not rest:
        return [([], start_pos)]

    # No suffix transitions available
    if start_pos not in sfx.SUFFIX_TRANSITIONS:
        return []

    results = []
    for target_pos, suffix_list in sfx.SUFFIX_TRANSITIONS[start_pos].items():
        for suffix_obj in suffix_list:
            suffix_form = suffix_obj.form(root)
            if rest.startswith(suffix_form):
                next_root = root + suffix_form
                remaining = rest[len(suffix_form):]
                subchains = find_suffix_chain(word, target_pos, next_root, visited) if remaining else [([], target_pos)]
                for chain, final_pos in subchains:
                    results.append(([(suffix_form, target_pos)] + chain, final_pos))
    return results


# TODO  gereksiz ayristirma denemelerini yok etmek. 
def decompose(word):
    """Find all possible legal decompositions of a word."""
    if not word:
        return []
    analyses = []
    for i in range(1, len(word) + 1):
        root = word[:i]
        if not wrd.exists(root):
            continue

        pos = "verb" if wrd.can_be_verb(root) else "noun"

        # Try both POS routes if it's potentially a verb
        chains = (find_suffix_chain(word, "verb", root) +
                  find_suffix_chain(word, "noun", root)) if pos == "verb" \
                  else find_suffix_chain(word, "noun", root)

        for chain, final_pos in chains:
            analyses.append((root, pos, chain, final_pos))
    return analyses


# =====================================================================
# ANALYSIS LOGIC
# =====================================================================

def word_info(word):
    """Print basic info about a word."""
    exists = wrd.exists(word)
    major = wrd.major_harmony(word)
    minor = wrd.minor_harmony(word)
    display.info_line("Dictionary entry:", "Yes" if exists else "No")
    display.info_line("Major harmony:", major.name if major else "None")
    display.info_line("Minor harmony:", minor.name if minor else "None")
    print("-" * 70)


def analyze_word(word):
    """
    Analyze both full word and its possible compound splits.
    If no valid decomposition found, report accordingly.
    """
    display.header("TURKISH MORPHOLOGICAL ANALYZER")
    display.info_line("Analyzing:", word)
    print("-" * 70)
    word_info(word)

    all_compounds = []
    for i in range(2, len(word) - 1):
        head, tail = word[:i], word[i:]
        head_decomp, tail_decomp = decompose(head), decompose(tail)
        if head_decomp and tail_decomp:
            all_compounds.append([head_decomp, tail_decomp])

    # Print results in order of priority
    if all_compounds:
        display.header("COMPOSITE ANALYSIS RESULTS")
        display.info_line("Total possible splits:", len(all_compounds))
        for idx, (head_d, tail_d) in enumerate(all_compounds, 1):
            display.compound_decomposition(head_d, tail_d, idx)
    else:
        full = decompose(word)
        if full:
            display.header("SIMPLE WORD ANALYSIS RESULTS")
            display.info_line("Total valid decompositions:", len(full))
            for idx, (root, pos, chain, final_pos) in enumerate(full, 1):
                display.single_decomposition(root, pos, chain, final_pos, idx)
        else:
            display.header("NO VALID DECOMPOSITIONS FOUND")
            print("The input word does not exist in dictionary or has no valid suffix chain.")


# =====================================================================
# MAIN LOOP
# =====================================================================

def main():
    display.header("Turkish Morphological Analyzer")

    while True:
        word = input("\nSözcük giriniz (Enter word): ").lower().strip()
        if not word:
            print("No word provided. Exiting.")
            break
        analyze_word(word)


if __name__ == "__main__":
    main()
