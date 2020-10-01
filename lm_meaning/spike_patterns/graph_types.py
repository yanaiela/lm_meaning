from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class PatternNode:
    lm_pattern: str
    spike_pattern: str
    lemma: str
    extended_lemma: str
    tense: str
    example: str = None

    wiki_occurence: int = None

    def __str__(self):
        return self.lm_pattern


class EdgeType(Enum):
    syntactic = 1
    lexical = 2
    both = 3
