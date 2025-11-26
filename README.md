# Credit Card Fraud Detection using Random Forest


## Project Overview
This project demonstrates the use of a **Random Forest classifier** to detect fraudulent credit card transactions. The dataset contains transactions made by European cardholders in 2023, with over 550,000 anonymized records. The primary goal is to develop a model capable of identifying potentially fraudulent transactions.

## Dataset
- **Source**: [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data)  
- **Features**:
  - **id**: Unique identifier for each transaction  
  - **V1-V28**: Anonymized features representing transaction attributes  
  - **Amount**: Transaction amount  
  - **Class**: Target variable (0 = non-fraud, 1 = fraud)  

> **Note on Class Distribution**: This dataset is overly simplified, with a nearly equal proportion of fraud and non-fraud transactions. In real-world scenarios, fraudulent transactions are extremely rare compared to legitimate ones, making actual fraud detection significantly more challenging.

## Class Distribution
```python
0    0.500248
1    0.499752
Name: proportion, dtype: float64
```

## Model Performance
### Cross-Validation F1 Score
```text
Cross-Validation F1 scores: [0.9847, 0.9864, 0.9847, 0.9843, 0.9839]
Average F1 Score: 0.9848
```


### Understanding the Classification Report

| Metric       | Meaning |
|-------------|---------|
| **Precision** | Measures how many of the transactions predicted as fraud (or non-fraud) are actually correct. High precision means fewer false positives. Formula: `Precision = TP / (TP + FP)` |
| **Recall**    | Measures how many of the actual fraudulent (or non-fraudulent) transactions were correctly identified. High recall means fewer false negatives. Formula: `Recall = TP / (TP + FN)` |
| **F1-score**  | The harmonic mean of precision and recall. Provides a balance between precision and recall. Formula: `F1 = 2 * (Precision * Recall) / (Precision + Recall)` |
| **Support**   | The number of actual instances of each class in the dataset. It helps contextualize precision, recall, and F1-score. |
| **Accuracy**  | Overall proportion of correctly classified transactions. Formula: `Accuracy = (TP + TN) / Total` |
| **Macro Avg** | Average of precision, recall, and F1-score across all classes without weighting by support. Treats all classes equally. |
| **Weighted Avg** | Average of precision, recall, and F1-score across all classes weighted by the number of instances in each class. Accounts for class imbalance. |


### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.97      | 1.00   | 0.99     | 56,750  |
| 1     | 1.00      | 0.97   | 0.99     | 56,976  |
| **Accuracy** | -       | -      | 0.99     | 113,726 |
| **Macro Avg** | 0.99    | 0.99   | 0.99     | 113,726 |
| **Weighted Avg** | 0.99  | 0.99   | 0.99     | 113,726 |


## Conclusion

In this project, a **Random Forest classifier** was used to detect fraudulent credit card transactions on a 2023 European cardholder dataset. The model achieved near-perfect performance:

- **Cross-Validation F1 Score:** ~0.985  
- **Accuracy:** 0.99  
- **ROC–AUC:** 1.00  

While the results indicate excellent model performance, it is important to note that this dataset is **overly simplified**, with nearly equal proportions of fraud and non-fraud transactions. In real-world scenarios:

- Fraudulent transactions are extremely rare, making detection more challenging.  
- High F1-score or AUC may be difficult to achieve without careful handling of **class imbalance**.
- 
# Handling Class Imbalance in Fraud Detection

## Class Weights
Assign higher weights to the minority class (fraud) during model training to penalize misclassifying fraud more heavily.  
This helps the model focus on rare events without oversampling the already large dataset.

## Threshold Adjustment
Instead of the default 0.5 probability cutoff, adjust the classification threshold to optimize **F1-score** or **Recall**.  
This ensures the model better captures fraudulent transactions while controlling false positives.

## SMOTE (Synthetic Minority Over-sampling Technique)
Generate synthetic samples for the minority class to improve model learning on rare events.  
**Caution:** Can be memory- and compute-intensive for very large datasets, and may introduce artificial patterns if applied indiscriminately.

## Isolation Forest / Anomaly Detection
Use anomaly detection algorithms to identify outliers as potential frauds without oversampling.  
**Con:** May require careful tuning and can be less effective if anomalies are not clearly separable.

