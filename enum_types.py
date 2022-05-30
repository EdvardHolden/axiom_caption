from enum import Enum


class AxiomOrder(Enum):
    # Set axiom order type
    ORIGINAL = "original"
    LEXICOGRAPHIC = "lexicographic"
    LENGTH = "length"
    FREQUENCY = "frequency"
    RANDOM = "random"
    RANDOM_GLOBAL = "random_global"

    def __str__(self):
        return self.value


class Context(Enum):
    AXIOMS = "axioms"
    WORDS = "words"

    def __str__(self):
        return self.value


class EncoderInput(Enum):
    SEQUENCE = "sequence"
    FLAT = "flat"

    def __str__(self):
        return self.value


class AttentionMechanism(Enum):
    BAHDANAU = "bahdanau"
    FLAT = "flat"
    NONE = "none"
    LOUNG_DOT = "loung_dot"
    LOUNG_CONCAT = "loung_concat"

    def __str__(self):
        return self.value


class GenerationMode(Enum):
    # Enum type for the different generating modes
    CLEAN = "clean"
    IDEAL = "ideal"
    POSITIVE_AXIOMS = "positive_axioms"
    CAPTION = "caption"
    CAPTION_SINE = "caption_sine"
    SINE = "sine"

    def __str__(self):
        return self.value
