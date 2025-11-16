"""
Tokenizer using tiktoken for GPT models.

This module provides a simple tokenizer interface that uses tiktoken,
which is the same tokenization library used by OpenAI's GPT models.
"""

import tiktoken


class Tokenizer:
    """
    A tokenizer class that wraps tiktoken for encoding and decoding text.
    
    Tokenization is the process of converting text into numbers (tokens)
    that a language model can understand. Each token typically represents
    a word or part of a word.
    
    Attributes:
        encoding: The tiktoken encoding object used for tokenization
        vocab_size: The size of the vocabulary (number of possible tokens)
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the tokenizer with a specific encoding.
        
        Args:
            encoding_name: The name of the encoding to use.
                          Common options:
                          - "cl100k_base": Used by GPT-4 and GPT-3.5-turbo
                          - "p50k_base": Used by GPT-3 models
                          - "r50k_base": Used by older GPT-3 models
        """
        # Get the encoding from tiktoken
        # This encoding knows how to convert text to tokens and vice versa
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # The vocabulary size is the maximum token ID + 1
        # This tells us how many different tokens the model can use
        self.vocab_size = self.encoding.n_vocab
    
    def encode(self, text: str) -> list[int]:
        """
        Convert text into a list of token IDs (numbers).
        
        This is what we do before feeding text to a language model.
        The model works with numbers, not text directly.
        
        Args:
            text: The input text string to tokenize
            
        Returns:
            A list of integers representing the tokens
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.encode("Hello, world!")
            [9906, 11, 1917, 0]
        """
        # tiktoken's encode method converts text to a list of token IDs
        return self.encoding.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Convert a list of token IDs back into text.
        
        This is what we do after getting output from a language model.
        The model produces numbers, and we need to convert them back to text.
        
        Args:
            token_ids: A list of token IDs (integers) to decode
            
        Returns:
            The decoded text string
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.decode([9906, 11, 1917, 0])
            "Hello, world!"
        """
        # tiktoken's decode method converts token IDs back to text
        return self.encoding.decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """
        Count how many tokens are in a given text.
        
        This is useful for:
        - Checking if text fits within model limits
        - Estimating costs (many APIs charge per token)
        - Understanding how the model "sees" your text
        
        Args:
            text: The input text string
            
        Returns:
            The number of tokens in the text
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.count_tokens("Hello, world!")
            4
        """
        # Encode the text and return the length of the resulting list
        return len(self.encode(text))


# Example usage and testing
if __name__ == "__main__":
    # Create a tokenizer instance
    # Using "cl100k_base" which is the encoding for GPT-4 and GPT-3.5-turbo
    tokenizer = Tokenizer(encoding_name="cl100k_base")
    
    # Example text to tokenize
    sample_text = "Hello, world! This is a test of the tokenizer."
    
    print(f"Original text: {sample_text}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    
    # Encode the text (convert to token IDs)
    token_ids = tokenizer.encode(sample_text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {tokenizer.count_tokens(sample_text)}")
    print()
    
    # Decode back to text (convert token IDs back to text)
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")
    print()
    
    # Verify that encoding and decoding are reversible
    if decoded_text == sample_text:
        print("✓ Encoding and decoding work correctly!")
    else:
        print("✗ Warning: Encoding/decoding may have issues")

