from typing import List, Optional, Tuple, Dict, Any

def welcome():
    """Show welcome message and available commands"""
    print("\n Commands:")
    print("  - Enter a word to analyze and train")
    print("  - 'sentence <text>' - Train on a full sentence")
    print("  - 'auto' - Start auto mode (random words from dictionary)")
    print("  - 'eval <word>' - Evaluate model on a word")
    print("  - 'relearn' - Train on all logged decompositions")
    print("  - 'stats' - Show training statistics")
    print("  - 'sample' - Analyze a text.")
    print("  - 'save' - Save model")
    print("  - 'quit' - Exit")

def show_decompositions(word: str, view_models: List[Dict[str, Any]]) -> None:
    """
    Display all decompositions. 
    Expects 'view_models' to be already sorted and formatted.
    """
    print(f"\n{word}")
    
    for i, vm in enumerate(view_models, 1):
        _show_single_option(i, vm)

def _show_single_option(display_idx: int, vm: Dict[str, Any]):
    """Format and print a single pre-calculated option"""
    print(f"\n[Option {display_idx}]")
    
    if vm.get('score') is not None:
        print(f"ML Score: {vm['score']:.4f}")
    
    print(f"Root:      {vm['root_str']}")
    
    if vm.get('has_chain'):
        print(f"Suffixes:  {vm['suffixes_str']}")
        print(f"Names:     {vm['names_str']}")
        print(f"Formation: {vm['formation_str']}")
    else:
        print(f"Formation: {vm['root_str']} (no suffixes)")
    
    print(f"Final POS: {vm['final_pos']}")
    print("-" * 70)

def show_sentence_prediction(display_idx: int, score: float, words: List[str], view_models: List[Dict[str, Any]], aligned_str: str) -> None:
    """Format and print a sentence prediction with detailed word breakdowns."""
    print(f"\n[Option {display_idx}] Score: {score:.4f}")
    print(f"    {aligned_str}\n")
    
    for w, vm in zip(words, view_models):
        print(f"  {w}:")
        print(f"    Root:      {vm['root_str']}")
        if vm.get('has_chain'):
            print(f"    Suffixes:  {vm['suffixes_str']}")
            print(f"    Names:     {vm['names_str']}")
            print(f"    Formation: {vm['formation_str']}")
        else:
            print(f"    Formation: {vm['root_str']} (no suffixes)")
        print(f"    Final POS: {vm['final_pos']}")
    print("-" * 70)

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
        
        parsed = _parse_numbers(choice, num_options)
        if parsed is not None:
            return [c - 1 for c in parsed]

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
    
    return list(dict.fromkeys(choices))

def confirm_save() -> bool:
    return input("Save model before quitting? (y/n): ").strip().lower() == 'y'

def show_auto_summary(stats: Dict):
    print(f"\n{'='*70}")
    print(f"AUTO MODE SUMMARY")
    print(f"{'='*70}")
    print(f" Processed: {stats['words_processed']}")
    print(f" Deleted: {stats['words_deleted']}")
    print(f" Skipped: {stats['words_skipped']}")
    print(f"{'='*70}\n")

def show_batch_summary(trained: int, skipped: int, final_loss: float, total: int):
    print(f"\n{'='*70}")
    print(f"BATCH TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f" Trained: {trained} examples")
    print(f" Skipped: {skipped} words")
    print(f" Final loss: {final_loss:.4f}")
    print(f" Total examples: {total}")
    print(f"{'='*70}\n")

def format_decomposition(word: str, decomposition: Tuple, simple: bool = False) -> str:
    """Format a decomposition into readable text (Simplified for display only)."""
    root, pos, chain, final_pos = decomposition
    if not chain:
        return root
    
    suffix_names = [suffix.name for suffix in chain]
    result = root + '+' + '+'.join(suffix_names)
    return result