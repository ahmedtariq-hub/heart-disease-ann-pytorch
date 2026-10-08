# heart-disease-ann-pytorch
# Heart Disease Prediction with an ANN (PyTorch)

A feed-forward neural network built in PyTorch to predict heart disease from clinical measurements.

## Dataset
UCI Heart Disease (Cleveland), 303 patients, 13 features (age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, ECG results, exercise-induced angina, etc.). The original target (0 to 4) is binarized: 0 = no disease, 1 to 4 = disease.

## Approach
- Stratified train/test split (80/20), with a further validation split from the training data
- Missing values in `ca` and `thal` filled with the training-set mode
- One-hot encoding of categorical features, standard scaling (fit on train only)
- Model: 2 hidden layers (32 and 16 units), ReLU, dropout 0.3, single logit output
- Loss: `BCEWithLogitsLoss`, optimizer: Adam (lr 1e-3, weight decay 1e-4)
- Early stopping on validation loss (patience 15), best weights restored

## Results (test set, 61 patients)
| Metric | Value |
|---|---|
| Accuracy | 0.869 |
| ROC-AUC | 0.951 |
| Recall (disease) | 0.89 |
| Precision (disease) | 0.83 |

Confusion matrix: 28 true negatives, 5 false positives, 3 false negatives, 25 true positives.

The test set is small, so these numbers are approximate.

## Files
- `heart_disease_ann.ipynb`: full notebook (data prep, training, evaluation, inference on new data)
- `heart_ann.pt`: trained model weights
- `scaler.pkl`: fitted StandardScaler
- `columns.json`: feature columns used at training time

## Run
```bash
pip install torch pandas scikit-learn ucimlrepo joblib
```
Open the notebook in Colab or Jupyter and run the cells in order.

## Disclaimer
This is a learning project, not a medical tool. Do not use it for diagnosis.
