import util.word_methods as wrd
from util.suffixes import find_suffix_chain  




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




#LOGIC FOR COMPOUND WORDS.
def analyze_word(word):
    """
    Analyze both full word and its possible compound splits.
    If no valid decomposition found, report accordingly.
    """
    all_compounds = []
    for i in range(2, len(word) - 1):
        head, tail = word[:i], word[i:]
        head_decomp, tail_decomp = decompose(head), decompose(tail)
        if head_decomp and tail_decomp:
            all_compounds.append([head_decomp, tail_decomp])

    # Print results in order of priority
    if all_compounds:
        for idx, (head_d, tail_d) in enumerate(all_compounds, 1):
            pass
        ##    display.compound_decomposition(head_d, tail_d, idx)


