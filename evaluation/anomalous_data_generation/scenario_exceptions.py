class NoPossibleTradeException(Exception):
    """Raised when no possible trade can be made in a scenario."""
    pass

class IncorrectMatchingException(Exception):
    """Raised when generated trades do not match required sums."""
    pass