import os

# Define Turkish alphabet order for sorting
# Note: 'ı' comes before 'i' in Turkish
TR_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
TR_MAP = {char: index for index, char in enumerate(TR_ALPHABET)}

def turkish_lower(text):
    """
    Custom lowercase function for Turkish.
    Standard .lower() maps 'I' to 'i' (wrong for Turkish).
    This maps 'I' to 'ı' and 'İ' to 'i'.
    """
    # 1. Handle capital dotted I -> i
    text = text.replace('İ', 'i')
    # 2. Handle capital dotless I -> ı
    text = text.replace('I', 'ı')
    # 3. Standard lower for the rest
    return text.lower()

def turkish_sort_key(word):
    """
    Generates a list of weights for characters in a word 
    based on the Turkish alphabet order.
    """
    lower_word = turkish_lower(word)
    key = []
    for char in lower_word:
        # If char is in our alphabet, get its index. 
        # If not (e.g., numbers, punctuation), use its unicode value + 100 to push it to end (or keep relative order)
        key.append(TR_MAP.get(char, ord(char) + 100))
    return key

def read_file_to_set(filepath):
    """
    Reads a file and returns a set of unique lines.
    Handles Turkish characters using utf-8.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, and ignore empty lines
            content = {line.strip() for line in f if line.strip()}
        return content
    except Exception as e:
        print(f"An error occurred reading {filepath}: {e}")
        return None

def write_set_to_file(filepath, word_collection):
    """
    Writes the collection back to the file, sorted by Turkish rules.
    """
    try:
        # Convert to list and sort using the custom Turkish key
        sorted_words = sorted(list(word_collection), key=turkish_sort_key)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for word in sorted_words:
                f.write(word + '\n')
        
        print(f"Success! '{filepath}' has been updated.")
        print(f"Total line count: {len(sorted_words)}")
    except Exception as e:
        print(f"An error occurred writing to {filepath}: {e}")

def main():
    print("--- Turkish Word List Manager ---")
    
    # 1. Get the Main File
    main_file = input("Enter the main file name (e.g., words.txt): ").strip()
    
    # Load the main file data
    main_words = read_file_to_set(main_file)
    if main_words is None:
        return 

    print(f"\nLoaded {len(main_words)} words from {main_file}.")
    print("Choose operation:")
    print("1: ADD words (Union - standard addition)")
    print("2: SUBTRACT words (Affix deletion - removes if word contains sub-word)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice not in ['1', '2']:
        print("Invalid choice. Exiting.")
        return

    # 2. Get the Second File (Operand)
    second_file = input("Enter the second file name: ").strip()
    second_words = read_file_to_set(second_file)
    
    if second_words is None:
        return 

    # 3. Perform Operations
    final_words = set()

    if choice == '1':
        # ADD / UNION
        initial_count = len(main_words)
        main_words.update(second_words)
        final_words = main_words
        
        added_count = len(main_words) - initial_count
        print(f"Processing... Added {added_count} new unique words.")
        
    elif choice == '2':
        # SUBTRACT / AFFIX FILTER
        print("Processing subtraction (this may take a moment for large files)...")
        initial_count = len(main_words)
        
        # We need to keep words from main_words ONLY IF they do not contain
        # any word from second_words as a substring.
        # Note: We iterate over a copy or simply construct a new set.
        
        words_to_keep = set()
        
        # Sort removal words by length (descending) can sometimes optimize matching,
        # but simple iteration is robust.
        removal_list = list(second_words)
        
        for word in main_words:
            should_remove = False
            # Check if this word contains ANY of the removal keywords
            # Using turkish_lower for comparison to ensure case-insensitive matching if desired,
            # but usually affix removal implies exact text match or case-insensitive match.
            # Assuming standard case-insensitive match for robustness:
            
            w_lower = turkish_lower(word)
            
            for removal_word in removal_list:
                r_lower = turkish_lower(removal_word)
                if r_lower in w_lower:
                    should_remove = True
                    break # Stop checking other removal words, it's already gone
            
            if not should_remove:
                words_to_keep.add(word)
                
        final_words = words_to_keep
        removed_count = initial_count - len(final_words)
        print(f"Processing... Removed {removed_count} words containing banned substrings.")

    # 4. Save changes
    write_set_to_file(main_file, final_words)

if __name__ == "__main__":
    main()