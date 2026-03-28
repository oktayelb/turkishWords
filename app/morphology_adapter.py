from typing import List, Dict, Tuple, Any
import util.decomposer as sfx
from ml.ml_ranking_model import SUFFIX_OFFSET, CATEGORY_CLOSED_CLASS
from util.suffix import Suffix
from util.words.closed_class import ClosedClassMarker, ALL_CLOSED_CLASS_WORDS


## translatxions between representations
def match_decompositions(entries: List[Dict], decompositions: List[Tuple]) -> List[int]:
    """Matches logged decomposition entries against dynamically generated decompositions."""
    indices = []
    for entry in entries:
        entry_root     = entry['root']
        entry_suffixes = [s['name'] for s in entry.get('suffixes', [])]
        for idx, (root, _, chain, _) in enumerate(decompositions):
            if root != entry_root:
                continue
            chain_suffixes = [s.name for s in chain] if chain else []
            if chain_suffixes == entry_suffixes and idx not in indices:
                indices.append(idx)
                break
    return indices

def encode_suffix_chain(suffix_chain: List) -> List[Tuple[int, int]]:
    """
    Encodes a suffix chain into (token_id, category_id) pairs for the ML model.

    Handles two kinds of chain elements:
    - Regular Suffix objects    → suffix token IDs with Noun/Verb category
    - ClosedClassMarker objects → closed-class token IDs with CATEGORY_CLOSED_CLASS
    """
    suffix_to_id = {
        suffix.name: idx + SUFFIX_OFFSET
        for idx, suffix in enumerate(sfx.ALL_SUFFIXES)
    }
    # Closed-class token IDs start right after all suffix IDs
    cc_offset = SUFFIX_OFFSET + len(sfx.ALL_SUFFIXES)
    cc_to_id  = {
        id(cc_obj): cc_offset + idx
        for idx, cc_obj in enumerate(ALL_CLOSED_CLASS_WORDS)
    }
    category_to_id = {'Noun': 0, 'Verb': 1}

    if not suffix_chain:
        return []

    encoded = []
    for s in suffix_chain:
        if isinstance(s, ClosedClassMarker):
            token_id = cc_to_id.get(id(s.cc_word), cc_offset)
            encoded.append((token_id, CATEGORY_CLOSED_CLASS))
        else:
            token_id = suffix_to_id.get(s.name, SUFFIX_OFFSET)
            cat_id   = category_to_id.get(s.makes.name, 0)
            encoded.append((token_id, cat_id))
    return encoded

def reconstruct_morphology(word: str, decomposition: Tuple) -> Dict[str, Any]:
    """Reconstructs the step-by-step morphology string from a root and suffix chain."""
    root, pos, chain, final_pos = decomposition

    # Closed-class word: display the category label, no suffix breakdown
    if chain and isinstance(chain[0], ClosedClassMarker):
        cc = chain[0].cc_word
        return {
            'root_str':      f"{root} ({cc.category})",
            'final_pos':     final_pos,
            'has_chain':     False,
            'formation_str': f"{root} [{cc.category}]",
        }

    if not chain:
        verb_marker = "-" if pos == "verb" else ""
        return {
            'root_str':      f"{root} ({pos})",
            'final_pos':     final_pos,
            'has_chain':     False,
            'formation_str': f"{root}{verb_marker} (no suffixes)",
        }
    
    current_stem = root
    suffix_forms = []
    suffix_names = []
    formation    = [root + ("-" if pos == "verb" else "")]
    
    cursor    = len(root)
    start_idx = 0
    
    if chain and chain[0].name == "pekistirme":
        root_idx = word.find(root)
        if root_idx > 0:
            prefix_str = word[:root_idx]
            suffix_forms.append(prefix_str)
            suffix_names.append(chain[0].name)
            current_stem = prefix_str + root
            formation.append(current_stem)
            cursor    = root_idx + len(root)
            start_idx = 1

    if start_idx == 0:
        if not word.startswith(root) and chain:
            first_suffix = chain[0]
            possible_forms = first_suffix.form(root)
            match_found = False
            for offset in range(3):
                test_cursor = len(root) - offset
                if test_cursor <= 0:
                    break
                rest_of_word = word[test_cursor:]
                for form in possible_forms:
                    if rest_of_word.startswith(form):
                        cursor     = test_cursor
                        match_found = True
                        break
                if match_found:
                    break

    for i in range(start_idx, len(chain)):
        suffix_obj     = chain[i]
        possible_forms = suffix_obj.form(current_stem)
        found_form     = None 
        
        for form in possible_forms:
            if word.startswith(form, cursor):
                found_form = form
                break
        
        if found_form is None:
            has_iyor_ahead = any("iyor" in chain[k].name for k in range(i + 1, len(chain)))
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
                suffix_forms.append(possible_forms[0] + "?")
                suffix_names.append(suffix_obj.name)
                current_stem += possible_forms[0]
                cursor       += len(possible_forms[0])
            continue
        
        suffix_forms.append(found_form if found_form else "(ø)")
        suffix_names.append(suffix_obj.name)
        current_stem += found_form
        cursor       += len(found_form)
        
        verb_marker = "-" if suffix_obj.makes.name == "Verb" else ""
        formation.append(current_stem + verb_marker)
        
    return {
        'root_str':      f"{root} ({pos})",
        'final_pos':     final_pos,
        'has_chain':     True,
        'suffixes_str':  ' + '.join(suffix_forms),
        'names_str':     ' + '.join(suffix_names),
        'formation_str': ' → '.join(formation),
    }

def format_detailed_decomp(decomp: Tuple) -> str:
    """
    Formats the decomposition to include both suffix name and specific surface form.
    Example: ev+plural_ler+ablative_de+marking_ki
    """
    root, pos, chain, final_pos = decomp
    if not chain:
        return root
        
    parts = [root]
    current = root
    for suffix in chain:
        forms = suffix.form(current)
        used_form = forms[0] if forms else suffix.suffix
        
        if used_form:
            parts.append(f"{suffix.name}_{used_form}")
        else:
            parts.append(suffix.name)
            
        current += used_form
        
    return "+".join(parts)
