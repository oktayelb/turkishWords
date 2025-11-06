import importlib

SUFFIX_MODULES = {
    "noun": {
        "noun_to_noun": "suffixes.name2name",
        "noun_to_verb": "suffixes.name2verb",
    },
    "verb": {
        "verb_to_noun": "suffixes.verb2name",
        "verb_to_verb": "suffixes.verb2verb",
    },
}

def load_suffixes():
    """Load all suffixes and store transition info."""
    suffixes = {}
    for pos, modules in SUFFIX_MODULES.items():
        suffixes[pos] = {}
        for transition, module_name in modules.items():
            mod = importlib.import_module(module_name)
            target_pos = transition.split("_to_")[1]
            suffixes[pos][target_pos] = {
                "front": mod.front,
                "back": mod.back
            }
    return suffixes

SUFFIX_DATA = load_suffixes()  # load once globally
