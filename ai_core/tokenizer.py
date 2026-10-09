"""
Educational Tokenizer Module
============================
Explains how computers convert human text (letters/words) into numbers (tokens)
that neural networks and LLMs can actually process.

Covers:
1. Character Tokenization: 1 token = 1 character (simple, huge sequence length).
2. Word Tokenization: 1 token = 1 word (large vocabulary, fails on typos/new words).
3. Subword (BPE - Byte-Pair Encoding) Tokenization: The standard modern approach
   used by GPT-4, Gemini, Claude, and LLaMA. Balances vocabulary size and sequence length.
"""

import re
from typing import List, Dict, Any, Tuple


class CharTokenizer:
    """Character-level tokenizer. Splits text into individual characters."""
    def __init__(self):
        # Printable ASCII + common symbols
        self.vocab: Dict[str, int] = {chr(i): i for i in range(32, 127)}
        self.vocab['\n'] = 127
        self.vocab['<UNK>'] = 0
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        return list(text)

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(ch, self.vocab['<UNK>']) for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        return "".join([self.inv_vocab.get(tid, '?') for tid in token_ids])


class WordTokenizer:
    """Word-level tokenizer. Splits by whitespace and punctuation."""
    def __init__(self, vocabulary: List[str] = None):
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        if vocabulary:
            for word in vocabulary:
                w = word.lower().strip()
                if w and w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        # Split words while keeping punctuation separate
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        return [t.lower() for t in tokens]

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        return " ".join([self.inv_vocab.get(tid, "<UNK>") for tid in token_ids])


class BPETokenizer:
    """
    Subword Tokenizer simulation (Byte-Pair Encoding concept).
    Modern LLMs (GPT-4, Gemini, LLaMA) use BPE / WordPiece.
    Instead of whole words or single characters, it breaks words into frequent subword chunks.
    Example: 'unhappiness' -> ['un', 'happi', 'ness']
             'transformer' -> ['trans', 'former']
    """
    def __init__(self):
        # A curated educational vocabulary of common subwords, roots, prefixes, suffixes, and words
        common_subwords = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>", " ", "\n",
            # Letters and digits
            *"abcdefghijklmnopqrstuvwxyz0123456789.,!?:;\"'()[]{}/*+-=@#$%^&_~",
            # Common subwords & morphemes
            "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of",
            "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
            "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea", "ra", "ce", "li", "ch", "ll",
            "be", "ma", "si", "om", "ur", "ca", "el", "ta", "la", "wa", "wh", "wi", "un", "dis", "pre",
            "ing", "tion", "sion", "ment", "ness", "able", "ible", "ous", "ful", "less", "ly", "est",
            "trans", "form", "former", "learn", "deep", "neural", "net", "work", "network", "intel",
            "lig", "ence", "intelligence", "arti", "ficial", "artificial", "model", "token", "vector",
            "embed", "ding", "embedding", "chat", "bot", "chatbot", "atten", "attention", "query",
            "key", "value", "matrix", "layer", "weight", "bias", "loss", "grad", "ient", "gradient",
            "optim", "izer", "optimizer", "prompt", "gener", "ate", "generation", "sample", "temp",
            "erature", "temperature", "prob", "ability", "probability", "comput", "er", "computer",
            "data", "train", "test", "val", "human", "code", "python", "reason", "math", "logic"
        ]

        self.vocab: Dict[str, int] = {}
        for idx, token in enumerate(dict.fromkeys(common_subwords)):
            self.vocab[token] = idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # Sort subwords by length descending for greedy longest-match subword segmentation
        self.sorted_tokens = sorted([k for k in self.vocab.keys() if len(k) > 1 and not k.startswith("<")],
                                    key=len, reverse=True)

    def _segment_word(self, word: str) -> List[str]:
        """Segments a single word into subwords using greedy longest match."""
        if not word:
            return []
        
        segments = []
        i = 0
        w_lower = word.lower()
        n = len(w_lower)

        while i < n:
            matched = False
            for sub in self.sorted_tokens:
                if w_lower.startswith(sub, i):
                    segments.append(word[i:i + len(sub)])
                    i += len(sub)
                    matched = True
                    break
            if not matched:
                # Fallback to single character
                segments.append(word[i])
                i += 1
        return segments

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into subword tokens preserving spaces (indicated by 'Ġ' or ' ').
        """
        raw_words = re.findall(r"\w+|[^\w\s]|\s+", text)
        result_tokens = []

        for piece in raw_words:
            if piece.isspace():
                # Represent space as explicit token marker
                result_tokens.append(" ")
            elif re.match(r"[^\w\s]", piece):
                result_tokens.append(piece)
            else:
                subwords = self._segment_word(piece)
                result_tokens.extend(subwords)

        return result_tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(t.lower(), self.vocab.get("<UNK>", 1)) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        parts = [self.inv_vocab.get(tid, "<UNK>") for tid in token_ids]
        return "".join(parts)


def analyze_text_tokens(text: str) -> Dict[str, Any]:
    """
    Returns an in-depth comparative educational analysis of the given text
    across Char, Word, and BPE tokenizers.
    """
    char_tok = CharTokenizer()
    word_tok = WordTokenizer()
    bpe_tok = BPETokenizer()

    char_tokens = char_tok.tokenize(text)
    char_ids = char_tok.encode(text)

    word_tokens = word_tok.tokenize(text)
    word_ids = word_tok.encode(text)

    bpe_tokens = bpe_tok.tokenize(text)
    bpe_ids = bpe_tok.encode(text)

    char_len = len(text)
    bpe_count = max(len(bpe_tokens), 1)
    compression_ratio = round(char_len / bpe_count, 2)

    # Color palette for token visualization
    colors = [
        "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
        "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1"
    ]

    bpe_chips = []
    for i, (tok, tid) in enumerate(zip(bpe_tokens, bpe_ids)):
        bpe_chips.append({
            "token": tok,
            "id": tid,
            "color": colors[i % len(colors)],
            "is_space": tok == " "
        })

    return {
        "text": text,
        "char_count": char_len,
        "bpe": {
            "tokens": bpe_tokens,
            "token_ids": bpe_ids,
            "token_count": len(bpe_tokens),
            "chips": bpe_chips,
            "compression_ratio": compression_ratio,
            "explanation": "Byte-Pair Encoding splits words into frequent subwords. Notice how common words remain whole, while complex or new words are broken into meaningful parts."
        },
        "word": {
            "tokens": word_tokens,
            "token_ids": word_ids,
            "token_count": len(word_tokens),
            "explanation": "Word tokenization treats every word as 1 token. It struggles with out-of-vocabulary words and requires a dictionary with millions of words."
        },
        "char": {
            "tokens": char_tokens,
            "token_ids": char_ids,
            "token_count": len(char_tokens),
            "explanation": "Character tokenization treats each letter as 1 token. Small vocabulary (~100 tokens), but sequence lengths explode, making attention computationally expensive ($O(N^2)$)."
        }
    }
