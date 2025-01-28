import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np

class MolecularEncoder(nn.Module):
    """Encoder network for 3D molecular structures."""
    
    def __init__(
        self, 
        input_dim: int = 3,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ])
        
        # Add intermediate layers
        for _ in range(num_layers - 1):
            self.layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            
        # Final projection to latent space
        self.layers.append(nn.Linear(hidden_dim, latent_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, num_atoms, 3)
        
        Returns:
            Encoded representation of shape (batch_size, num_atoms, latent_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x

class MolecularDecoder(nn.Module):
    """Decoder network for 3D molecular structures."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 3,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ])
        
        # Add intermediate layers
        for _ in range(num_layers - 1):
            self.layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            
        # Final projection to coordinate space
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Input tensor of shape (batch_size, num_atoms, latent_dim)
        
        Returns:
            Reconstructed coordinates of shape (batch_size, num_atoms, 3)
        """
        for layer in self.layers:
            x = layer(x)
        return x

class VectorQuantizer(nn.Module):
    """Vector Quantizer module for VQ-VAE."""
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Vector quantization forward pass.
        
        Args:
            inputs: Input tensor of shape (batch_size, num_atoms, embedding_dim)
            
        Returns:
            Tuple of:
                - Quantized tensor of shape (batch_size, num_atoms, embedding_dim)
                - Commitment loss
        """
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(inputs.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, commitment_loss + q_latent_loss

class MolecularVQVAE(nn.Module):
    """Complete VQ-VAE model for 3D molecular structures."""
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        commitment_cost: float = 0.25,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.encoder = MolecularEncoder(
            input_dim=3,
            hidden_dim=hidden_dim,
            latent_dim=embedding_dim,
            num_layers=num_layers
        )
        
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )
        
        self.decoder = MolecularDecoder(
            latent_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=num_layers
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            x: Input tensor of shape (batch_size, num_atoms, 3)
            
        Returns:
            Dictionary containing:
                - 'reconstruction': Reconstructed coordinates
                - 'quantized': Quantized representation
                - 'loss': VQ-VAE loss
        """
        z = self.encoder(x)
        quantized, vq_loss = self.vector_quantizer(z)
        reconstruction = self.decoder(quantized)
        
        reconstruction_loss = F.mse_loss(reconstruction, x)
        loss = reconstruction_loss + vq_loss
        
        return {
            'reconstruction': reconstruction,
            'quantized': quantized,
            'loss': loss
        }
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to discrete tokens."""
        z = self.encoder(x)
        quantized, _ = self.vector_quantizer(z)
        return quantized
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode tokens to 3D coordinates."""
        return self.decoder(z)

# Training utilities
def train_vqvae(
    model: MolecularVQVAE,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
) -> List[float]:
    """
    Train the VQ-VAE model.
    
    Args:
        model: MolecularVQVAE model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        List of training losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            loss = output['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Record average epoch loss
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return losses