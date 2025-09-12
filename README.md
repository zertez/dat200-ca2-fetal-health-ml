# Supervised Machine Learning Classification Pipeline for Medical Data

A machine learning pipeline for binary classification of fetal health data using linear classifiers. This project implements and compares Perceptron, Adaline, and Logistic Regression algorithms on medical diagnostic data.

## Authors

- **Jannicke Ådalen**
- **Marcus Dalaker Figenschou** 
- **Rikke Sellevold Vegstein**

*Group project for DAT200 Applied Machine Learning, Spring 2025 (CA2)*

## Project Overview

This project implements a supervised machine learning workflow for fetal health classification. We compare three linear classifiers on cardiotocographic data to predict fetal health outcomes (Normal vs Pathological).

## Dataset

- **Source**: `assets/fetal_health.csv`
- **Target**: Binary classification (0 = Normal, 1 = Pathological)
- **Features**: Cardiotocographic measurements (baseline fetal heart rate, accelerations, etc.)

## Implementation

### Models
- **Perceptron**: Binary linear classifier with threshold activation
- **Adaline**: Adaptive linear neuron with continuous activation  
- **Logistic Regression**: Probabilistic linear classifier with sigmoid activation

### Key Features
- Data loading and validation with missing value handling
- Stratified train/test split to maintain class balance
- Feature standardization using Z-score normalization
- Performance evaluation across different dataset sizes and epochs
- Visualization of results with heatmaps and decision boundaries

## Requirements

- Python >= 3.13
- pandas >= 2.3.2
- numpy >= 2.3.3
- matplotlib >= 3.10.6
- seaborn >= 0.13.2
- mlxtend >= 0.23.4

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ca2_supervised_medical_ml_pipeline
```

2. Install dependencies:
```bash
uv sync
```

## Usage

Run the complete pipeline:
```bash
python CA2.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook CA2.ipynb
```

## Project Structure

```
ca2_supervised_medical_ml_pipeline/
├── CA2.py                      # Main implementation
├── CA2.ipynb                   # Jupyter notebook version
├── README.md                   # This file
├── pyproject.toml             # Dependencies
├── assets/
│   ├── fetal_health.csv       # Dataset
│   └── example_output.png     # Sample output
└── .gitignore                 # Git ignore rules
```

## Methodology

1. **Data Exploration**: Statistical analysis and visualization of features and target distribution
2. **Preprocessing**: Stratified splitting and feature standardization
3. **Model Training**: Training across multiple dataset sizes (50-700 samples) and epochs (2-97)
4. **Evaluation**: Performance comparison using accuracy metrics and visualization

## Results

- Logistic Regression shows most stable performance across configurations
- Model performance generally improves with larger training sets
- Feature standardization significantly improves convergence
- Limited linear separability suggests potential for more complex models

## License

This project is created for educational purposes as part of DAT200 coursework.