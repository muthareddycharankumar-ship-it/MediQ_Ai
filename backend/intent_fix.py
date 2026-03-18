# Test fuzzy matching
from thefuzz import fuzz

test = "hlep me"
phrases = ["help me", "help", "can you help"]
for p in phrases:
    print(f"{p}: {fuzz.partial_ratio(p, test)}")
