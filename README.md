# MolToken: 3D Molecular Structure Tokenization

[![PyPI version](https://badge.fury.io/py/moltoken.svg)](https://badge.fury.io/py/moltoken)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

MolToken is a Python library for tokenizing 3D molecular structures, enabling better representation learning and analysis of molecular geometries.

## Features

- 🧬 Tokenization of 3D molecular structures
- 🔄 Rotation and translation invariant representations
- ⚡ Fast and efficient processing
- 📊 Built-in visualization tools
- 🔌 Easy integration with deep learning frameworks
- 📁 Support for common molecular file formats (PDB, MOL, XYZ)

## Installation

### Using pip

```bash
pip install moltoken
```

### From source

```bash
# Clone the repository
git clone https://github.com/HFooladi/MolToken.git
cd MolToken

# Create and activate conda environment
conda env create -f environment.yml
conda activate tokenmol

# Install in development mode
pip install -e .
```

## Quick Start

Here's a simple example of how to use MolToken:

```python
from moltoken.tokenizer import MoleculeTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem

# Initialize tokenizer
tokenizer = MoleculeTokenizer(
    max_atoms=50,
    spatial_resolution=0.1,
    consider_bonds=True
)

# Create a simple molecule (ethanol)
mol = Chem.MolFromSmiles('CCO')
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)

# Get molecular information
conf = mol.GetConformer()
coordinates = conf.GetPositions()
atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 
         bond.GetBondTypeAsDouble()) for bond in mol.GetBonds()]

# Tokenize the molecule
tokens = tokenizer.encode_molecule(
    coordinates=coordinates,
    atom_types=atom_types,
    bonds=bonds
)

print(f"Generated tokens: {tokens}")
```

## Documentation

For detailed documentation, visit our [documentation page]().

The documentation includes:
- Detailed API reference
- Tutorials and examples
- Advanced usage guides
- Contribution guidelines

## Development

We welcome contributions! Here's how to set up the development environment:

```bash
# Clone the repository
git clone https://github.com/hfooladi/MolToken.git
cd MolToken

# Create development environment
conda env create -f environment.yml
conda activate tokenmol

# Install development dependencies
pip install -e ".[test]"

# Run tests
pytest tests/
```

### Code Style

We use:
- Black for code formatting
- Ruff for linting
- MyPy for type checking

To format your code:
```bash
black moltoken/
ruff check moltoken/
mypy moltoken/
```

## Citation

If you use MolToken in your research, please cite:

```bibtex
@software{moltoken2024,
  author = {Fooladi, Hosein},
  title = {MolToken: 3D Molecular Structure Tokenization},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/hfooladi/MolToken}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Thanks to all contributors who have helped shape MolToken
- Special thanks to the RDKit and PyTorch communities

## Contact

Hosein Fooladi - fooladi.hosein@gmail.com

Project Link: [https://github.com/hfooladi/MolToken](https://github.com/hfooladi/MolToken)

## Project Status

MolToken is under active development. Check our [project board](https://github.com/hfooladi/MolToken/projects) for planned features and current progress.
