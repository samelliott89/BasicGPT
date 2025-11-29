"""
Tokenizer using tiktoken for GPT models.

This module provides a simple tokenizer interface that uses tiktoken,
which is the same tokenization library used by OpenAI's GPT models.
"""

import tiktoken


class Tokenizer:
    """
    A tokenizer class that wraps tiktoken for encoding and decoding text.

    Attributes:
        encoding: The tiktoken encoding object used for tokenization
        vocab_size: The size of the vocabulary (number of possible tokens)
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the tokenizer with a specific encoding.

        Args:
            encoding_name: The name of the encoding to use.
                          - "cl100k_base": Used by GPT-4 and GPT-3.5-turbo
                          - "p50k_base": Used by GPT-3 models
                          - "r50k_base": Used by older GPT-3 models
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """
        Convert text into a list of token IDs.

        Args:
            text: The input text string to tokenize

        Returns:
            A list of integers representing the tokens
        """
        return self.encoding.encode(text, disallowed_special=())

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert a list of token IDs back into text.

        Args:
            token_ids: A list of token IDs to decode

        Returns:
            The decoded text string
        """
        return self.encoding.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """
        Count how many tokens are in a given text.

        Args:
            text: The input text string

        Returns:
            The number of tokens in the text
        """
        return len(self.encode(text))


if __name__ == "__main__":
    tokenizer = Tokenizer()
    sample = "Hello, world! This is a test."

    print(f"Original: {sample}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    tokens = tokenizer.encode(sample)
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    print(f"✓ Round-trip successful: {decoded == sample}")
