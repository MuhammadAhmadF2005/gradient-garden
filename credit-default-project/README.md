## Credit Default Prediction

### Problem
Predict whether a credit card client will default next month using demographic and financial features.

### Dataset
UCI Credit Card Default Dataset (30,000 samples, imbalanced binary classification).

### Models Used
- Dummy Classifier (baseline)
- Logistic Regression
- Logistic Regression + SMOTE
- Hyperparameter-tuned Logistic Regression

### Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC
(Accuracy not used due to class imbalance)

### Final Model Performance (Test Set)
- ROC-AUC: ~X.XX
- Recall (Default): ~X.XX
- Precision (Default): ~X.XX

### Key Takeaways
- Accuracy is misleading for imbalanced data
- Recall is critical for risk-sensitive domains
- SMOTE improves minority class detection
- Threshold and regularization control risk tradeoffs
