def process_verb_file(filename):
    # The Turkish alphabet in the correct order
    # Note: We include both 'i' and 'ı'
    turkish_alphabet = "abcçdefgğhıijklmnoöprsştuüvyz"
    
    # Create a lookup dictionary: {'a': 0, 'b': 1, 'c': 2, ...}
    # This assigns a numerical rank to each letter.
    letter_rank = {letter: index for index, letter in enumerate(turkish_alphabet)}

    def turkish_sort_key(word):
        """
        Converts a word into a list of numbers representing its sort order.
        Example: 'aç' becomes [0, 3] whereas 'ad' becomes [0, 4].
        This ensures 'aç' comes before 'ad'.
        """
        word = word.lower() # Ensure we sort case-insensitively
        return [letter_rank.get(char, 999) for char in word]

    try:
        # 1. Read the file
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 2. Clean and Deduplicate
        # strip() removes whitespace; conditional ensures no empty lines
        unique_words = {line.strip() for line in lines if line.strip()}

        # 3. Sort using the custom Turkish key
        sorted_words = sorted(list(unique_words), key=turkish_sort_key)

        # 4. Write back
        with open(filename, 'w', encoding='utf-8') as f:
            for word in sorted_words:
                f.write(f"{word}\n")
        
        print(f"Success! Processed {len(sorted_words)} unique words sorted by Turkish rules.")
        return sorted_words

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []

# --- Usage ---
process_verb_file('fiiller.txt')