# ❤️ Heart Attack Risk Triage System (CardioGuard AI)

This repository contains a **Medical Triage Application** powered by a Support Vector Machine (SVM). The system is designed to help clinical staff prioritize patients based on their risk of heart disease using 13 key physiological and diagnostic indicators.

## 🚀 Features

* **High-Precision Engine:** Optimized SVM with RBF kernel achieving **99% accuracy**.
* **Clinical Triage:** Focused on **Recall (97%)** to minimize dangerous False Negatives (missed cases).
* **Interactive Dashboard:** Streamlit-based UI with tabs, tooltips, and real-time risk assessment.
* **Confidence Metrics:** Visualizes "Hyperplane Distance" to show how certain the model is about a specific diagnosis.

---

## 📊 Model Performance

After extensive hyperparameter tuning using `GridSearchCV`, the model achieved the following results on the test set:

| Metric | Score |
| --- | --- |
| **Accuracy** | 99% |
| **Precision (High Risk)** | 100% |
| **Recall (High Risk)** | 97% |
| **F1-Score** | 99% |

### Confusion Matrix

```text
[[102   0]  <- Correctly identified 102 Healthy
 [  3 100]] <- Caught 100 Sick, missed 3

```

---

## 🧪 Dataset Description

The model utilizes 13 clinical features for its predictions:

1. **Age**: Patient's age.
2. **Sex**: 1 = Male, 0 = Female.
3. **CP**: Chest pain type (0-3).
4. **Trestbps**: Resting blood pressure.
5. **Chol**: Serum cholesterol.
6. **FBS**: Fasting blood sugar > 120 mg/dl.
7. **RestECG**: Resting electrocardiographic results.
8. **Thalach**: Maximum heart rate achieved.
9. **Exang**: Exercise-induced angina.
10. **Oldpeak**: ST depression induced by exercise.
11. **Slope**: Slope of the peak exercise ST segment.
12. **CA**: Number of major vessels (0-3) colored by fluoroscopy.
13. **Thal**: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect).

---

## 🛠️ Installation & Usage

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Clone and Install Dependencies

```bash
git clone https://github.com/yourusername/heart-risk-triage.git
cd heart-risk-triage
pip install -r requirements.txt

```

### 3. File Structure

```text
.
├── app.py             # Main Streamlit application
├── notebook/
│   └── cardioAI.ipynb   #notebook file 
├── pkl_files/
│   ├── grid_pramas.pkl   # Trained SVM Model
│   └── scaler.pkl        # StandardScaler object
├── requirements.txt
└── README.md

```

### 4. Running the App

```bash
streamlit run app.py

```

---

## 🔬 Methodology

* **Preprocessing:** Data was standardized using `StandardScaler` to ensure the SVM's distance-based calculations were not skewed by different units (e.g., Age vs. Cholesterol).
* **Hyperparameters:**
* **Kernel:** Radial Basis Function (RBF) to handle non-linear clinical patterns.
* **C:** 100 (Strong penalty for misclassification).
* **Gamma:** 0.1 (Focused influence for high precision).


* **Optimization:** Optimized for **F1-score** to balance the need for identifying sick patients without over-burdening hospitals with false alarms.

---

## ⚖️ Disclaimer

*This tool is intended for educational and triage prioritization assistance only. It is not a replacement for professional medical diagnosis, advice, or treatment.*

---
