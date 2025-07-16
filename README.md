# 🧠 Breast Cancer Prediction using Machine Learning

This project uses supervised machine learning algorithms to predict whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on various cell features from the Breast Cancer Wisconsin dataset.

---

## 📂 Dataset

- **Source**: [Kaggle - Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- The dataset contains **569 samples** with **30 numerical features** describing characteristics of cell nuclei present in breast cancer biopsies.
- Target variable:  
  - `M` → Malignant  
  - `B` → Benign

---

## 🛠️ Technologies Used

- Python 🐍
- pandas, numpy
- matplotlib, seaborn (for visualization)
- scikit-learn (for ML models)
- pickle (for model saving)

---

## 📈 Project Workflow

```python
# 1. Load dataset
df = pd.read_csv('data.csv')

# 2. Clean & preprocess
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
df.rename(columns={'diagnosis': 'target'}, inplace=True)
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

# 3. Split & scale
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train models
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

---
## 📊 Visualizations
- Class distribution plot
- Feature correlation heatmap
- Confusion matrix

---

## 📌 How to Run This Notebook
1. Clone this repository:
- ```
  git clone https://github.com/CodeCrafter-101/Breast_Cancer_Prediction.git
  cd Breast_Cancer_Prediction
  ```

2. Install required libraries:
- ```
  pip install -r requirements.txt
  ```
  
3. Launch the notebook:












