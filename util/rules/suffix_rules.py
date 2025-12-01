# suffix_rules.py
"""
Rule-based filtering system for Turkish morphological decompositions.
Rules are checked during decomposition to discard invalid suffix chains.
"""

from typing import List, Callable
from enum import Enum


class RuleType(Enum):
    """Types of morphological rules"""
    INCOMPATIBILITY = "incompatibility"       # Suffix A cannot be followed by suffix B
    NO_REPETITION = "no_repetition"           # Suffix cannot occur twice
    REQUIRED_AFTER = "required_after"         # After A, one of [B, C, D] must follow (if anything follows)
    ONLY_AFTER = "only_after"                 # Suffix A can ONLY occur immediately after [B, C, D]
    ALLOWED_SUCCESSORS = "allowed_successors" # After A, ONLY [B, C, D] can come. All else forbidden.
    FORBIDDEN_SEQUENCE = "forbidden_sequence" # Specific sequence [A ... B ... C] is forbidden (anywhere in chain)
    MAX_OCCURRENCES = "max_occurrences"       # Suffix can occur at most N times
    ONLY_AT_BEGINNING = "only_at_beginning"   # Suffix can only be the FIRST suffix (attached to root)
    ONLY_AT_END = "only_at_end"               # Suffix must be the LAST suffix (nothing can follow it)


class SuffixRule:
    """Base class for suffix rules"""
    def __init__(self, rule_type: RuleType, description: str):
        self.rule_type = rule_type
        self.description = description
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        """
        Check if adding current_suffix to suffix_chain violates this rule.
        
        Args:
            suffix_chain: List of Suffix objects already in the chain
            current_suffix: Suffix object being considered to add
        
        Returns:
            True if rule is satisfied (valid), False if violated (invalid)
        """
        raise NotImplementedError


class IncompatibilityRule(SuffixRule):
    """Rule: Suffix A cannot be followed by suffix B"""
    def __init__(self, first_suffix_name: str, forbidden_next_names: List[str], description: str = None):
        self.first_suffix = first_suffix_name
        self.forbidden_next = set(forbidden_next_names)
        desc = description or f"After '{first_suffix_name}', cannot use: {', '.join(forbidden_next_names)}"
        super().__init__(RuleType.INCOMPATIBILITY, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        if not suffix_chain:
            return True
        
        last_suffix = suffix_chain[-1]
        if last_suffix.name == self.first_suffix:
            if current_suffix.name in self.forbidden_next:
                return False
        return True


class NoRepetitionRule(SuffixRule):
    """Rule: Certain suffixes cannot occur more than once in a chain"""
    def __init__(self, suffix_names: List[str], description: str = None):
        self.suffix_names = set(suffix_names)
        desc = description or f"Cannot repeat: {', '.join(suffix_names)}"
        super().__init__(RuleType.NO_REPETITION, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        if current_suffix.name not in self.suffix_names:
            return True
        
        # Check if this suffix already exists in the chain
        for suffix in suffix_chain:
            if suffix.name == current_suffix.name:
                return False
        return True


class OnlyAfterRule(SuffixRule):
    """
    Rule: Suffix A can ONLY occur if it is immediately preceded by one of [B, C, D].
    If the chain is empty (attaching to root) or the predecessor is not in the list, it is invalid.
    """
    def __init__(self, target_suffix: str, allowed_predecessors: List[str], description: str = None):
        self.target_suffix = target_suffix
        self.allowed_predecessors = set(allowed_predecessors)
        desc = description or f"'{target_suffix}' can only be used immediately after: {', '.join(allowed_predecessors)}"
        super().__init__(RuleType.ONLY_AFTER, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        if current_suffix.name != self.target_suffix:
            return True
        
        if not suffix_chain:
            return False
        
        last_suffix = suffix_chain[-1]
        return last_suffix.name in self.allowed_predecessors


class AllowedSuccessorsRule(SuffixRule):
    """
    Rule: After 'trigger_suffix', ONLY suffixes in 'allowed_next' can be used.
    Any suffix NOT in 'allowed_next' will be rejected immediately after 'trigger_suffix'.
    """
    def __init__(self, trigger_suffix: str, allowed_next: List[str], description: str = None):
        self.trigger_suffix = trigger_suffix
        self.allowed_next = set(allowed_next)
        desc = description or f"After '{trigger_suffix}', only the following are allowed: {', '.join(allowed_next)}"
        super().__init__(RuleType.ALLOWED_SUCCESSORS, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        if not suffix_chain:
            return True
        
        last_suffix = suffix_chain[-1]
        if last_suffix.name == self.trigger_suffix:
            return current_suffix.name in self.allowed_next
            
        return True


class OnlyAtBeginningRule(SuffixRule):
    """
    Rule: A specific suffix can ONLY be used at the very beginning of the chain
    (i.e., attached directly to the root).
    """
    def __init__(self, suffix_name: str, description: str = None):
        self.suffix_name = suffix_name
        desc = description or f"'{suffix_name}' can only be used at the beginning of the suffix chain"
        super().__init__(RuleType.ONLY_AT_BEGINNING, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        if current_suffix.name == self.suffix_name:
            # It is only valid if the chain is empty (meaning we are at the root)
            return len(suffix_chain) == 0
        return True


class OnlyAtEndRule(SuffixRule):
    """
    Rule: A specific suffix must be the LAST suffix.
    This implies that NO other suffix can be added after it.
    """
    def __init__(self, suffix_name: str, description: str = None):
        self.suffix_name = suffix_name
        desc = description or f"'{suffix_name}' must be the last suffix (cannot be followed by others)"
        super().__init__(RuleType.ONLY_AT_END, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        # If the chain is empty, we are fine (we are adding the first suffix)
        if not suffix_chain:
            return True
            
        # If the PREVIOUS suffix (the end of the current chain) is the restricted one,
        # then adding *any* new suffix is invalid.
        last_suffix = suffix_chain[-1]
        if last_suffix.name == self.suffix_name:
            return False
            
        return True


class ForbiddenSequenceRule(SuffixRule):
    """
    Rule: A specific sequence of suffixes is forbidden.
    IMPORTANT: This checks for the sequence appearing anywhere in the chain relative to each other,
    not necessarily adjacent. 
    E.g. If forbidden is [A, B], then A -> C -> D -> B is INVALID.
    """
    def __init__(self, forbidden_sequence: List[str], description: str = None):
        self.forbidden_sequence = forbidden_sequence
        desc = description or f"Forbidden sequence order: {' ... '.join(forbidden_sequence)}"
        super().__init__(RuleType.FORBIDDEN_SEQUENCE, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        if len(self.forbidden_sequence) == 0:
            return True
        
        full_sequence = [s.name for s in suffix_chain] + [current_suffix.name]
        
        forbidden_idx = 0
        target_len = len(self.forbidden_sequence)
        
        for suffix_name in full_sequence:
            if suffix_name == self.forbidden_sequence[forbidden_idx]:
                forbidden_idx += 1
                if forbidden_idx == target_len:
                    return False
                    
        return True


class MaxOccurrencesRule(SuffixRule):
    """Rule: A suffix can occur at most N times in a chain"""
    def __init__(self, suffix_name: str, max_count: int, description: str = None):
        self.suffix_name = suffix_name
        self.max_count = max_count
        desc = description or f"'{suffix_name}' can occur at most {max_count} time(s)"
        super().__init__(RuleType.MAX_OCCURRENCES, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        if current_suffix.name != self.suffix_name:
            return True
        
        count = sum(1 for s in suffix_chain if s.name == self.suffix_name)
        return count < self.max_count


class RequiredAfterRule(SuffixRule):
    """Rule: After suffix A, one of the suffixes in [B, C, D] must follow (if anything follows)"""
    def __init__(self, trigger_suffix: str, required_next: List[str], description: str = None):
        self.trigger_suffix = trigger_suffix
        self.required_next = set(required_next)
        desc = description or f"After '{trigger_suffix}', must use one of: {', '.join(required_next)}"
        super().__init__(RuleType.REQUIRED_AFTER, desc)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        if not suffix_chain:
            return True
        
        last_suffix = suffix_chain[-1]
        if last_suffix.name == self.trigger_suffix:
            return current_suffix.name in self.required_next
        return True


class CustomRule(SuffixRule):
    """Rule: Custom validation function"""
    def __init__(self, validation_func: Callable, description: str):
        self.validation_func = validation_func
        super().__init__(RuleType.FORBIDDEN_SEQUENCE, description)
    
    def validate(self, suffix_chain: List, current_suffix) -> bool:
        return self.validation_func(suffix_chain, current_suffix)


class SuffixRuleEngine:
    """
    Central rule engine that validates suffix chains against all registered rules.
    """
    def __init__(self):
        self.rules: List[SuffixRule] = []
        self._initialize_default_rules()
    
    def add_rule(self, rule: SuffixRule):
        """Add a new rule to the engine"""
        self.rules.append(rule)
    
    def validate_suffix_addition(self, suffix_chain: List, new_suffix) -> tuple[bool, List[str]]:
        """
        Check if adding new_suffix to suffix_chain is valid according to all rules.
        
        Returns:
            (is_valid, violated_rules): True if valid, and list of violated rule descriptions
        """
        violated = []
        for rule in self.rules:
            if not rule.validate(suffix_chain, new_suffix):
                violated.append(rule.description)
        
        return len(violated) == 0, violated
    
    def validate_complete_chain(self, suffix_chain: List) -> tuple[bool, List[str]]:
        """
        Validate an entire suffix chain by checking each addition.
        
        Returns:
            (is_valid, violated_rules): True if valid, and list of violated rule descriptions
        """
        for i in range(len(suffix_chain)):
            is_valid, violated = self.validate_suffix_addition(suffix_chain[:i], suffix_chain[i])
            if not is_valid:
                return False, violated
        return True, []
    
    def _initialize_default_rules(self):
        """Initialize with common Turkish morphological rules"""
        
        # Rule: Plural suffix cannot repeat
        self.add_rule(NoRepetitionRule(
            ['plural_ler'],
            "Plural suffix cannot occur twice"
        ))
        
        # Forbidden sequences (Non-adjacent check)
        self.add_rule(NoRepetitionRule(["posessive_3sg"]))
        self.add_rule(ForbiddenSequenceRule(["aplicative_le","active_ir"]))
        self.add_rule(ForbiddenSequenceRule(["aplicative_le","factative_ir"]))
        self.add_rule(ForbiddenSequenceRule(["negative_me","reflexive_ik"]))
        self.add_rule(ForbiddenSequenceRule(["negative_me","reflexive_is"]))
        self.add_rule(ForbiddenSequenceRule(["negative_me","active_it"]))
        self.add_rule(ForbiddenSequenceRule(["negative_me","passive_il"]))
        self.add_rule(ForbiddenSequenceRule(["negative_me","active_ir"]))
        self.add_rule(ForbiddenSequenceRule(["negative_me","reflexive_in"]))
        self.add_rule(ForbiddenSequenceRule(["negative_me","perfectative_ik"]))
        self.add_rule(ForbiddenSequenceRule(["negative_me","toolative_ek"]))
        #self.add_rule(ForbiddenSequenceRule(["infinitive_me","dimunitive_ek_archaic"]))
        #self.add_rule(ForbiddenSequenceRule(["accusative","dimunitive_ek_archaic"]))
        #self.add_rule(ForbiddenSequenceRule(["composessive_li","dimunitive_ek_archaic"]))
        self.add_rule(ForbiddenSequenceRule(["approximative_si","pluralizer_archaic_iz"]))
        self.add_rule(ForbiddenSequenceRule(["accusative","posessive_3sg"]))
        self.add_rule(ForbiddenSequenceRule(["confactuous_le","counting_er"]))
        self.add_rule(OnlyAtBeginningRule("onomatopea_de"))
        self.add_rule(OnlyAtEndRule("approximative_si"))
        self.add_rule(OnlyAfterRule("conjugation_3sg", ["nounifier_ecek","factative_ir","pasttense_noundi","pastfactative_miş"]))
        self.add_rule(OnlyAfterRule("conjugation_2sg", ["nounifier_ecek","factative_ir","pasttense_noundi","pastfactative_miş"]))
        self.add_rule(OnlyAfterRule("conjugation_1sg", ["nounifier_ecek","factative_ir","pasttense_noundi","pastfactative_miş"]))
        self.add_rule(OnlyAfterRule("conjugation_3pl", ["nounifier_ecek","factative_ir","pasttense_noundi","pastfactative_miş"]))
        self.add_rule(OnlyAfterRule("conjugation_2pl", ["nounifier_ecek","factative_ir","pasttense_noundi","pastfactative_miş"]))
        self.add_rule(OnlyAfterRule("conjugation_1pl", ["nounifier_ecek","factative_ir","pasttense_noundi","pastfactative_miş"]))
        

# Global rule engine instance
RULE_ENGINE = SuffixRuleEngine()


# Convenience functions
def validate_suffix_addition(suffix_chain: List, new_suffix) -> bool:
    """Check if adding new_suffix to chain is valid"""
    is_valid, _ = RULE_ENGINE.validate_suffix_addition(suffix_chain, new_suffix)
    return is_valid


def validate_complete_chain(suffix_chain: List) -> bool:
    """Check if entire chain is valid"""
    is_valid, _ = RULE_ENGINE.validate_complete_chain(suffix_chain)
    return is_valid


def get_violated_rules(suffix_chain: List, new_suffix) -> List[str]:
    """Get list of rule descriptions that would be violated"""
    _, violated = RULE_ENGINE.validate_suffix_addition(suffix_chain, new_suffix)
    return violated


# Helper function to add custom rules from outside
def add_rule(rule: SuffixRule):
    """Add a custom rule to the global rule engine"""
    RULE_ENGINE.add_rule(rule)