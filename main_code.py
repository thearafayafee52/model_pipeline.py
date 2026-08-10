import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef

def run_clinical_pipeline(data_path):
    # 1. Load the Sakar et al. dataset
    df = pd.read_csv(data_path)
    
    # Assuming 'class' is the target variable and 'subject_id' is the unique patient ID
    X = df.drop(columns=['class', 'subject_id'])
    y = df['class']
    groups = df['subject_id'] 

    # 2. Strict Subject-Wise Data Partitioning (Data Leakage Prevention)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print("Subject-wise split complete. No patient data crosses the train/test boundary.")

    # 3. Information Gain Biomarker Selection (Top 20 Features)
    ig_scores = mutual_info_classif(X_train, y_train, random_state=42)
    top_20_indices = np.argsort(ig_scores)[-20:]
    
    X_train_top20 = X_train.iloc[:, top_20_indices]
    X_test_top20 = X_test.iloc[:, top_20_indices]
    
    print(f"Dimensionality heavily reduced. Selected Top 20 Acoustic Biomarkers.")

    # 4. Extreme Gradient Boosting (XGBoost) Champion Engine
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        n_estimators=200,
        objective='binary:logistic',
        random_state=42
    )
    
    # Train the model
    xgb_model.fit(X_train_top20, y_train)

    # Calculate basic evaluation metrics on independent holdout
    y_pred = xgb_model.predict(X_test_top20)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"Model Training Complete. Holdout MCC Score: {mcc:.3f}")

    # 5. Game-Theoretic XAI (SHAP Attribution)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_top20)
    
    return xgb_model, explainer, shap_values, X_test_top20

if __name__ == "__main__":
    # Replace 'parkinsons_acoustic_data.csv' with your actual dataset file name
    # run_clinical_pipeline('parkinsons_acoustic_data.csv')
    pass
