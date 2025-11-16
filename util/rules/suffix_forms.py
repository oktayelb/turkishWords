def custom_form_for_special_suffix(word, suffix_obj):
    # Your custom logic here
    result = suffix_obj.suffix
    
    # Can still use the helper methods if needed
    #result = Suffix._apply_major_harmony(word, result, suffix_obj.major_harmony)
    
    # Add your special rules
    if word.endswith('k'):
        result = 'special_' + result
    
    return [result, result + 'variant']

