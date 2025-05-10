# ELITE-GARP: An Explainable Genetic Ensemble for Immunotherapy Response Prediction in Pancreatic Cancer

## About the Project

ELITE-GARP (ELISE Inspired Technique using Genetic Algorithm for Response Prediction) is a novel computational framework that combines evolutionary feature selection with ensemble learning to predict patient response to immunotherapy in pancreatic cancer. Leveraging a real-world pancreatic cancer dataset, this framework integrates genetic algorithm-based feature selection with a stacked ensemble of classical machine learning models and neural network components to achieve high predictive performance with robust explainability.

## Dataset Description
The PDAC MSK 2024 dataset, available on cBioPortal, contains comprehensive clinical and molecular data from patients diagnosed with pancreatic ductal adenocarcinoma (PDAC) treated at Memorial Sloan Kettering Cancer Center[1]. This rich dataset includes:

### Patient Information

- **Demographics**: Race, Sex, and Ethnicity data enabling demographic subgroup analysis
- **Clinical Outcomes**: Overall Survival Status and Overall Survival (Months) for survival analysis
- **Disease Classification**: Cancer Study, Cancer Type, Cancer Type Detailed, and Primary Tumor Site for precise tumor origin classification

### Molecular Characteristics

- **Genomic Profiles**: Mutation Count, Fraction Genome Altered, and TMB (nonsynonymous) providing insights into tumor mutational landscapes
- **MSI Indicators**: Microsatellite instability data through MSI Type and MSI Score, crucial for immunotherapy response prediction
- **Technical Metadata**: Sample Coverage and Tumor Purity information to evaluate sequencing data quality

### Additional Features

- **Metastasis Information**: Metastatic Site documentation showing anatomical locations of tumor spread
- **Standardized Coding**: Oncotree Code for structured classification and cross-study data integration
- **Sample Details**: Sample Type, Sample ID, and Number of Samples Per Patient for assessing sample diversity

The dataset provides a solid foundation for developing predictive models in pancreatic cancer, particularly for immunotherapy response prediction, by offering the combination of detailed clinical annotations with comprehensive genomic profiling data.

## Key Features

1. **Genetic Algorithm Feature Selection**
   * Uses evolutionary search to identify optimal feature subsets
   * Combines with Principal Component Analysis (PCA) for dimensionality reduction
   * Achieves 95.17% explained variance with selected components

2. **Heterogeneous Ensemble Architecture**
   * Integrates classical ML algorithms (Random Forest, XGBoost, LightGBM, Logistic Regression)
   * Incorporates deep learning branches (DNN and AutoInt with self-attention)
   * Uses stacked ensemble approach with meta-learner for final predictions

3. **Dual-Layer Explainability**
   * PCA back-mapping connects abstract components to original biomarkers
   * SHAP (SHapley Additive exPlanations) values at both ensemble and model levels
   * Identifies key biomarkers like tumor mutational burden (TMB) and microsatellite instability (MSI)

4. **High Performance Metrics**
   * Achieves 95.95% accuracy and 0.9601 AUC
   * Balanced precision and recall across classes
   * Outperforms baseline models consistently

## How to Run the Project

### Prerequisites
* Python 3.8 or higher
* Required libraries:
```bash
pip install numpy pandas scikit-learn tensorflow xgboost lightgbm shap matplotlib seaborn
```

### Steps to Run
1. **Data Preparation**
   * Load and preprocess the pancreatic cancer dataset
   * Create the target variable based on survival thresholds

2. **Feature Selection**
   * Run PCA to extract principal components
   * Apply genetic algorithm to select optimal components

3. **Model Training**
   * Train base learners (RF, XGB, LGB, LR)
   * Train neural network branches (DNN, AutoInt)
   * Train meta-learner on stacked predictions

4. **Evaluation and Interpretation**
   * Generate performance metrics (accuracy, F1-score, AUC)
   * Create SHAP visualizations for model interpretability
   * Back-map PCA components to original features

## Project Structure

```
├── data/                      # Dataset files and preprocessing scripts
├── models/                    # Implementation of all model components
│   ├── genetic_algorithm.py   # GA implementation for feature selection
│   ├── base_learners.py       # Classical ML models
│   ├── neural_branches.py     # DNN and AutoInt implementations
│   └── ensemble.py            # Stacking ensemble implementation
├── explainability/            # Explainability modules
│   ├── pca_backmapping.py     # PCA component interpretation
│   └── shap_analysis.py       # SHAP value computation and visualization
├── utils/                     # Utility functions
├── main.py                    # Main execution script
└── README.md                  # Project documentation
```

## Results

### Performance Comparison
| Metric | Basic ELISE | ELITE-GARP |
|--------|-------------|------------|
| PCA Components | 15 | 20 |
| Explained Variance (PCA) | 84.03% | 95.17% |
| Accuracy | 0.89 | 0.9595 |
| F1-Score | 0.8974 | 0.9577 |
| AUC | - | 0.9601 |
| Macro Avg F1-Score | 0.89 | 0.96 |
| Weighted Avg F1-Score | 0.89 | 0.96 |

### Key Biomarkers Identified
1. Tumor Mutational Burden (TMB)
2. Microsatellite Instability (MSI) Score
3. Ethnicity factors
4. Oncotree Code IPMN

## Future Work

* Expand and standardize datasets for pancreatic cancer
* Enhance model transparency and interpretability
* Refine multi-omics data integration approaches
* Explore data augmentation techniques for imbalanced datasets
* Investigate end-to-end differentiable approaches

## Contributors

Pranshu Jain, Sooryakiran B, Mohsina Bilal, RVK Sravya  
Department of Computer Science, National Institute of Technology Calicut, India

## Citation

If you use this work in your research, please cite:
```
Jain, P., B, S., Bilal, M., & Sravya, R. (2025). ELITE-GARP: An Explainable Genetic Ensemble for Immunotherapy Response Prediction in Pancreatic Cancer. 
```