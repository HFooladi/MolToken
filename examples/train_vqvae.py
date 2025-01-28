import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from moltoken.models.vqvae import MolecularVQVAE, train_vqvae

def prepare_molecule_data(smiles_list, conformers_per_mol=1):
    """Prepare 3D molecular data from SMILES strings."""
    all_conformers = []
    
    for smiles in smiles_list:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Generate conformers
        AllChem.EmbedMultipleConfs(mol, numConfs=conformers_per_mol)
        
        # Get coordinates for each conformer
        for conf_id in range(conformers_per_mol):
            conf = mol.GetConformer(conf_id)
            coords = conf.GetPositions()
            all_conformers.append(coords)
    
    # Convert to torch tensor
    return torch.tensor(np.array(all_conformers), dtype=torch.float32)

def main():
    # Example SMILES strings
    smiles_list = [
        'CCO',  # ethanol
        'CC(=O)O',  # acetic acid
        'c1ccccc1',  # benzene
        'CCN'  # ethylamine
    ]
    
    # Prepare data
    data = prepare_molecule_data(smiles_list, conformers_per_mol=5)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )
    
    # Initialize model
    model = MolecularVQVAE(
        num_embeddings=512,
        embedding_dim=64,
        hidden_dim=256,
        commitment_cost=0.25,
        num_layers=3
    )
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = train_vqvae(
        model=model,
        train_loader=dataloader,
        num_epochs=100,
        learning_rate=1e-3,
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), 'molecular_vqvae.pt')
    
    # Example of encoding and decoding
    with torch.no_grad():
        model.eval()
        
        # Get a sample molecule
        sample = data[0:1].to(device)
        
        # Encode to discrete representation
        encoded = model.encode(sample)
        
        # Decode back to 3D coordinates
        decoded = model.decode(encoded)
        
        # Calculate reconstruction error
        reconstruction_error = torch.mean((sample - decoded) ** 2)
        print(f'Reconstruction error: {reconstruction_error.item():.4f}')

if __name__ == '__main__':
    main()