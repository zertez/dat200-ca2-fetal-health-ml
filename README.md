# Supervised Machine Learning Classification Pipeline for Medical Data

A machine learning pipeline for binary classification of fetal health data using linear classifiers. This project implements and compares the performance of Perceptron, Adaline, and Logistic Regression algorithms on medical diagnostic data.

## Authors

- **Jannicke Ådalen**
- **Marcus Dalaker Figenschou** 
- **Rikke Sellevold Vegstein**

*Group project for DAT200 Applied Machine Learning, Spring 2025*

## Project Overview

This project demonstrates a complete supervised machine learning workflow applied to fetal health classification. The pipeline includes data exploration, preprocessing, model training, evaluation, and visualization of results. The goal is to classify fetal health outcomes based on cardiotocographic features.

## Dataset

The project uses the Fetal Health Classification dataset containing:
- **Target variable**: Fetal health status (binary: 0 = Normal, 1 = Pathological)
- **Features**: Cardiotocographic measurements including baseline fetal heart rate, accelerations, and other clinical indicators
- **Size**: Varies based on preprocessing steps
- **Source**: `assets/fetal_health.csv`

## Features

### Data Processing
- Automated data loading with validation
- Missing value detection and handling
- Stratified train/test splitting to maintain class balance
- Feature standardization using Z-score normalization

### Model Implementation
- **Perceptron**: Binary linear classifier with threshold activation
- **Adaline**: Adaptive linear neuron with continuous activation
- **Logistic Regression**: Probabilistic linear classifier with sigmoid activation

### Analysis and Visualization
- Distribution analysis of features and target classes
- Learning curve analysis across different dataset sizes
- Performance comparison across varying training epochs
- Decision boundary visualization for all feature pairs
- Comprehensive performance heatmaps

### Experimental Design
- Dataset size analysis: 50 to 700 samples in increments of 50
- Epoch analysis: 2 to 97 epochs in increments of 5
- Stratified sampling to prevent class imbalance
- Proper train/test isolation with no data leakage

## Requirements

- Python >= 3.13
- pandas >= 2.3.2
- numpy >= 2.3.3
- matplotlib >= 3.10.6
- seaborn >= 0.13.2
- mlxtend >= 0.23.4
- jupyter >= 1.1.1 (optional, for notebook interface)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ca2_supervised_medical_ml_pipeline
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or install dependencies manually:
```bash
pip install pandas numpy matplotlib seaborn mlxtend jupyter
```

## Usage

### Running the Complete Pipeline

Execute the main script:
```bash
python CA2.py
```

Or run the Jupyter notebook:
```bash
jupyter notebook CA2.ipynb
```

### Key Functions

#### Data Loading
```python
df = load_data()
```
Loads the dataset, validates data integrity, and handles missing values.

#### Feature Scaling
```python
X_train_scaled = Xscale(X_train, X_train)
X_test_scaled = Xscale(X_test, X_train)
```
Standardizes features using training set statistics to prevent data leakage.

#### Model Training
The pipeline automatically trains and evaluates multiple classifiers across different configurations.

## Project Structure

```
ca2_supervised_medical_ml_pipeline/
├── CA2.py                      # Main implementation script
├── CA2.ipynb                   # Jupyter notebook version
├── README.md                   # This file
├── pyproject.toml             # Project configuration and dependencies
├── assets/
│   ├── fetal_health.csv       # Dataset
│   └── example_output.png     # Sample visualization output
├── .gitignore                 # Git ignore rules
├── .python-version            # Python version specification
└── .venv/                     # Virtual environment (if created)
```

## Methodology

### 1. Data Exploration
- Statistical summary of features
- Distribution visualization using histograms and count plots
- Class balance analysis
- Linear separability assessment through scatter plots

### 2. Data Preprocessing
- Stratified train/test split (75/25) maintaining class proportions
- Feature standardization using training set mean and standard deviation
- Data shuffling to prevent order-dependent learning

### 3. Model Training and Evaluation
- Training across multiple dataset sizes to analyze learning curves
- Epoch variation analysis to study convergence behavior
- Performance comparison using accuracy metrics
- Hyperparameter exploration across dataset sizes and training epochs

### 4. Results Visualization
- Heatmaps showing accuracy across dataset sizes and epochs
- Decision boundary plots for all feature pair combinations
- Performance comparison charts between different algorithms

## Results and Insights

### Key Findings
- **Dataset Size Impact**: Model performance generally improves with larger training sets, with diminishing returns after ~500 samples
- **Convergence Behavior**: Logistic Regression shows most stable convergence, while Perceptron exhibits high variance
- **Linear Separability**: The dataset shows limited linear separability, suggesting the need for more complex models for optimal performance
- **Feature Scaling Importance**: Standardization significantly improves convergence speed and stability

### Model Comparison
- **Logistic Regression**: Most stable and consistent performance across different configurations
- **Adaline**: Good convergence properties with gradual improvement
- **Perceptron**: Fast training but unstable performance, especially with smaller datasets

## Limitations and Future Work

### Current Limitations
- Limited to linear classifiers only
- Binary classification focus
- Dataset-specific implementation
- No cross-validation implemented

### Potential Improvements
- Implementation of non-linear classifiers (SVM, Random Forest, Neural Networks)
- Multi-class classification extension
- Cross-validation for more robust evaluation
- Automated hyperparameter tuning
- Feature selection and dimensionality reduction
- Advanced regularization techniques

## Academic Context

This project was developed for DAT200 Applied Machine Learning, Spring 2025 (CA2 - Computer Assignment 2). It demonstrates understanding of:
- Fundamental machine learning concepts
- Proper experimental design and validation
- Data preprocessing best practices
- Model comparison methodologies
- Statistical analysis and visualization

## Contributing

This is an academic project. For educational purposes, please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Submit a pull request with clear description of modifications

## License

This project is created for educational purposes. Please refer to your institution's academic integrity policies when using this code.

## Contact

This project was completed as a group assignment. For questions or discussions about this implementation, please refer to the course materials or contact through appropriate academic channels.

## Acknowledgments

Thanks to all group members for their contributions to this project and to the DAT200 course staff for guidance and support.