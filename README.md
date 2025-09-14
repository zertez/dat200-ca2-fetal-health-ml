# DAT200 CA2: Supervised Machine Learning Classification Pipeline for Medical Data

**Group 37**
- Jannicke Ådalen
- Marcus Dalaker Figenschou
- Rikke Sellevold Vegstein

## Project Overview

This project implements a supervised machine learning workflow for fetal health classification using linear classifiers. We compare three linear classification algorithms (Perceptron, Adaline, and Logistic Regression) on cardiotocographic data to predict fetal health outcomes in a binary classification task (Normal vs Pathological).

**Final Result:** Academic assignment focusing on algorithm comparison and implementation

## Technical Approach

- **Models**: Perceptron, Adaline (Adaptive Linear Neuron), Logistic Regression
- **Data Processing**: Stratified train/test split, Z-score normalization, missing value handling
- **Evaluation**: Performance comparison across different dataset sizes (50-700 samples) and epochs (2-97)
- **Visualization**: Decision boundaries, performance heatmaps, and convergence analysis

## Key Features

- Binary linear classifier with threshold activation (Perceptron)
- Adaptive linear neuron with continuous activation (Adaline)
- Probabilistic linear classifier with sigmoid activation (Logistic Regression)
- Feature standardization for improved convergence
- Comprehensive performance evaluation and visualization

## Results

- Logistic Regression demonstrated the most stable performance across configurations
- Model performance generally improved with larger training sets
- Feature standardization significantly enhanced convergence rates
- Limited linear separability suggested potential benefits from more complex models

## Files Structure

- `CA2.py` - Main implementation script
- `CA2.ipynb` - Jupyter notebook version with detailed analysis
- `assets/fetal_health.csv` - Cardiotocographic dataset
- `assets/example_output.png` - Sample visualization output
- `pyproject.toml` - Project dependencies

## Requirements

- Python >= 3.13
- pandas >= 2.3.2
- numpy >= 2.3.3
- matplotlib >= 3.10.6
- seaborn >= 0.13.2
- mlxtend >= 0.23.4

## Usage

1. Install dependencies: `uv sync` or `pip install -e .`
2. Run the main script: `python CA2.py`
3. Or use the Jupyter notebook: `jupyter notebook CA2.ipynb`

## Dataset

- **Source**: Cardiotocographic measurements from `assets/fetal_health.csv`
- **Target**: Binary classification (0 = Normal, 1 = Pathological)
- **Features**: Baseline fetal heart rate, accelerations, and other medical indicators
- **Preprocessing**: Stratified splitting and feature standardization applied