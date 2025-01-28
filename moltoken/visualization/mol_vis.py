import matplotlib.pyplot as plt
import numpy as np
from typing import List


class MoleculeVisualizer:
    def plot_token_distribution(self, tokens: List[int]) -> None:
        """Plot distribution of tokens."""
        plt.figure(figsize=(10, 6))
        plt.hist(tokens, bins=50)
        plt.title('Token Distribution')
        plt.xlabel('Token Value')
        plt.ylabel('Frequency')
        plt.show()
        
    def visualize_attention(self, molecule, attention_weights) -> None:
        """Visualize attention patterns on molecular structure."""
        raise NotImplementedError