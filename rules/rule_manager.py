"""
Rule Manager - Utility for managing, testing, and debugging suffix rules
"""

from suffix_rules import (
    RULE_ENGINE, 
    IncompatibilityRule, 
    NoRepetitionRule,
    ForbiddenSequenceRule,
    MaxOccurrencesRule,
    RequiredAfterRule,
    CustomRule,
    add_rule
)
import suffixes as sfx
from decomposition import decompose


def list_all_rules():
    """Display all currently registered rules"""
    print("\n" + "=" * 70)
    print("REGISTERED SUFFIX RULES")
    print("=" * 70)
    
    if not RULE_ENGINE.rules:
        print("No rules registered.")
        return
    
    for i, rule in enumerate(RULE_ENGINE.rules, 1):
        print(f"\n[{i}] {rule.rule_type.value.upper()}")
        print(f"    {rule.description}")


def list_all_suffixes():
    """Display all available suffixes grouped by type"""
    print("\n" + "=" * 70)
    print("ALL AVAILABLE SUFFIXES")
    print("=" * 70)
    
    print("\n--- VERB TO VERB ---")
    v2v = [s for s in sfx.ALL_SUFFIXES if s.comes_to == sfx.WType.Verb and s.makes == sfx.WType.Verb]
    for s in v2v:
        print(f"  {s.name:30} ('{s.suffix}')")
    
    print("\n--- NOUN TO NOUN ---")
    n2n = [s for s in sfx.ALL_SUFFIXES if s.comes_to == sfx.WType.Noun and s.makes == sfx.WType.Noun]
    for s in n2n:
        print(f"  {s.name:30} ('{s.suffix}')")
    
    print("\n--- NOUN TO VERB ---")
    n2v = [s for s in sfx.ALL_SUFFIXES if s.comes_to == sfx.WType.Noun and s.makes == sfx.WType.Verb]
    for s in n2v:
        print(f"  {s.name:30} ('{s.suffix}')")
    
    print("\n--- VERB TO NOUN ---")
    v2n = [s for s in sfx.ALL_SUFFIXES if s.comes_to == sfx.WType.Verb and s.makes == sfx.WType.Noun]
    for s in v2n:
        print(f"  {s.name:30} ('{s.suffix}')")


def test_rule_on_word(word: str, verbose: bool = True):
    """
    Test how rules affect decomposition of a word.
    Shows which potential decompositions were filtered out by rules.
    """
    print("\n" + "=" * 70)
    print(f"RULE TESTING FOR: {word}")
    print("=" * 70)
    
    decompositions = decompose(word)
    
    if not decompositions:
        print("\nNo valid decompositions found (all filtered by rules or no valid paths).")
        return
    
    print(f"\nFound {len(decompositions)} valid decomposition(s) after rule filtering:")
    
    for i, (root, pos, chain, final_pos) in enumerate(decompositions, 1):
        print(f"\n[{i}] Root: {root} ({pos})")
        if chain:
            suffix_names = [s.name for s in chain]
            print(f"    Suffixes: {' → '.join(suffix_names)}")
            
            # Test each suffix addition for rule violations
            if verbose:
                for j, suffix in enumerate(chain):
                    prev_chain = chain[:j]
                    is_valid, violated = RULE_ENGINE.validate_suffix_addition(prev_chain, suffix)
                    if not is_valid:
                        print(f"    ⚠️  Step {j+1} would violate: {violated}")
        else:
            print("    No suffixes")


def test_suffix_sequence(suffix_names: list):
    """
    Test if a specific sequence of suffix names would be valid according to rules.
    """
    print("\n" + "=" * 70)
    print(f"TESTING SUFFIX SEQUENCE")
    print("=" * 70)
    print(f"Sequence: {' → '.join(suffix_names)}")
    
    # Build actual suffix objects from names
    name_to_suffix = {s.name: s for s in sfx.ALL_SUFFIXES}
    
    suffix_objs = []
    for name in suffix_names:
        if name not in name_to_suffix:
            print(f"\n❌ Error: Suffix '{name}' not found!")
            return
        suffix_objs.append(name_to_suffix[name])
    
    print("\nValidating sequence step by step:")
    all_valid = True
    
    for i, suffix in enumerate(suffix_objs):
        prev_chain = suffix_objs[:i]
        is_valid, violated = RULE_ENGINE.validate_suffix_addition(prev_chain, suffix)
        
        status = "✓" if is_valid else "✗"
        print(f"\n{status} Step {i+1}: Adding '{suffix.name}'")
        
        if not is_valid:
            all_valid = False
            print(f"   Violated rules:")
            for rule in violated:
                print(f"   - {rule}")
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✓ Sequence is VALID according to all rules")
    else:
        print("✗ Sequence is INVALID - violates one or more rules")
    print("=" * 70)


def suggest_rules_from_feedback(word: str, correct_idx: int):
    """
    Analyze a word where user indicated correct decomposition.
    Suggest potential rules that might filter incorrect alternatives.
    """
    decompositions = decompose(word)
    
    if not decompositions or correct_idx >= len(decompositions):
        print("Invalid word or decomposition index.")
        return
    
    print("\n" + "=" * 70)
    print(f"RULE SUGGESTIONS FOR: {word}")
    print("=" * 70)
    
    correct_root, correct_pos, correct_chain, correct_final = decompositions[correct_idx]
    correct_names = [s.name for s in correct_chain]
    
    print(f"\nCorrect decomposition:")
    print(f"  Root: {correct_root} ({correct_pos})")
    print(f"  Suffixes: {' → '.join(correct_names) if correct_names else '(none)'}")
    
    print(f"\n{len(decompositions) - 1} alternative decomposition(s) to filter:")
    
    suggestions = []
    
    for i, (root, pos, chain, final) in enumerate(decompositions):
        if i == correct_idx:
            continue
        
        names = [s.name for s in chain]
        print(f"\n  Alternative {i+1}:")
        print(f"    Root: {root} ({pos})")
        print(f"    Suffixes: {' → '.join(names) if names else '(none)'}")
        
        # Analyze differences
        if len(chain) != len(correct_chain):
            suggestions.append(f"  - Different chain lengths: correct has {len(correct_chain)}, this has {len(chain)}")
        
        # Check for repeated suffixes
        name_counts = {}
        for name in names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        for name, count in name_counts.items():
            if count > 1:
                suggestions.append(f"  - Consider NoRepetitionRule for '{name}' (appears {count} times)")
        
        # Check for incompatible sequences
        for j in range(len(names) - 1):
            suggestions.append(f"  - Consider IncompatibilityRule: '{names[j]}' cannot be followed by '{names[j+1]}'")
    
    if suggestions:
        print("\n" + "-" * 70)
        print("SUGGESTED RULES:")
        unique_suggestions = list(set(suggestions))
        for suggestion in unique_suggestions[:10]:  # Limit to top 10
            print(suggestion)


def interactive_rule_builder():
    """Interactive CLI for building and adding new rules"""
    print("\n" + "=" * 70)
    print("INTERACTIVE RULE BUILDER")
    print("=" * 70)
    print("\nRule Types:")
    print("  1. Incompatibility Rule (A cannot be followed by B)")
    print("  2. No Repetition Rule (suffix cannot repeat)")
    print("  3. Forbidden Sequence (specific sequence is invalid)")
    print("  4. Max Occurrences (suffix can occur at most N times)")
    print("  5. Back to menu")
    
    choice = input("\nSelect rule type (1-5): ").strip()
    
    if choice == "1":
        first = input("Enter first suffix name: ").strip()
        forbidden = input("Enter forbidden next suffix names (comma-separated): ").strip()
        forbidden_list = [s.strip() for s in forbidden.split(",")]
        desc = input("Enter description (optional): ").strip()
        
        rule = IncompatibilityRule(first, forbidden_list, desc if desc else None)
        add_rule(rule)
        print(f"✓ Rule added: {rule.description}")
    
    elif choice == "2":
        suffixes = input("Enter suffix names that cannot repeat (comma-separated): ").strip()
        suffix_list = [s.strip() for s in suffixes.split(",")]
        desc = input("Enter description (optional): ").strip()
        
        rule = NoRepetitionRule(suffix_list, desc if desc else None)
        add_rule(rule)
        print(f"✓ Rule added: {rule.description}")
    
    elif choice == "3":
        sequence = input("Enter forbidden sequence (comma-separated): ").strip()
        seq_list = [s.strip() for s in sequence.split(",")]
        desc = input("Enter description (optional): ").strip()
        
        rule = ForbiddenSequenceRule(seq_list, desc if desc else None)
        add_rule(rule)
        print(f"✓ Rule added: {rule.description}")
    
    elif choice == "4":
        suffix = input("Enter suffix name: ").strip()
        max_count = int(input("Enter maximum occurrences: ").strip())
        desc = input("Enter description (optional): ").strip()
        
        rule = MaxOccurrencesRule(suffix, max_count, desc if desc else None)
        add_rule(rule)
        print(f"✓ Rule added: {rule.description}")


def main_menu():
    """Main interactive menu for rule management"""
    while True:
        print("\n" + "=" * 70)
        print("SUFFIX RULE MANAGER")
        print("=" * 70)
        print("\nOptions:")
        print("  1. List all rules")
        print("  2. List all suffixes")
        print("  3. Test rules on a word")
        print("  4. Test specific suffix sequence")
        print("  5. Add new rule")
        print("  6. Suggest rules from user feedback")
        print("  7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            list_all_rules()
        elif choice == "2":
            list_all_suffixes()
        elif choice == "3":
            word = input("Enter word to test: ").strip().lower()
            verbose = input("Verbose output? (y/n): ").strip().lower() == 'y'
            test_rule_on_word(word, verbose)
        elif choice == "4":
            sequence = input("Enter suffix names (comma-separated): ").strip()
            seq_list = [s.strip() for s in sequence.split(",")]
            test_suffix_sequence(seq_list)
        elif choice == "5":
            interactive_rule_builder()
        elif choice == "6":
            word = input("Enter word: ").strip().lower()
            idx = int(input("Enter correct decomposition index (0-based): ").strip())
            suggest_rules_from_feedback(word, idx)
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main_menu()
