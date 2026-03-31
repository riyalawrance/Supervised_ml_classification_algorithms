# 🩺 Diabetes Prediction using Supervised ML Algorithms

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> Comparing four supervised classification algorithms to predict diabetes onset using the Pima Indians Diabetes Dataset. Best accuracy achieved: **75.97% with Naive Bayes**.

---

## 🎯 Project Overview

This project explores and compares four classic supervised machine learning classification algorithms applied to a real-world medical dataset. The goal is to predict whether a patient has diabetes based on diagnostic measurements, while understanding the strengths and trade-offs of each algorithm.

**Algorithms Compared:**
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

---

## 📊 Dataset

**Pima Indians Diabetes Dataset** — a well-known benchmark dataset in medical ML research.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |
| **Outcome** | **Target: 0 = No Diabetes, 1 = Diabetes** |

---

## 📈 Results Summary

Each algorithm was tested across three train-test splits (80/20, 70/30, 60/40):

| Algorithm | Test 0.2 | Test 0.3 | Test 0.4 | Best Accuracy |
|---|---|---|---|---|
| Decision Tree | 66.88% | 69.70% | 70.78% | 70.78% |
| SVM (Hard Margin) | 29.22% | 35.06% | 57.47% | 57.47% |
| KNN (k=5) | 69.48% | 73.59% | 69.81% | 73.59% |
| **Naive Bayes** | **75.97%** | 72.72% | 73.37% | **75.97% ✅** |

**🏆 Winner: Naive Bayes** with 75.97% accuracy at 80/20 split.

> **Note:** SVM's poor performance is attributed to the absence of feature scaling — a known requirement for SVM to converge effectively.

---

## 🔍 Key Observations

- **Naive Bayes** performed best overall despite its independence assumption, likely due to the small dataset size
- **SVM** underperformed significantly without feature normalization — a great learning point about preprocessing
- **KNN** showed competitive performance, peaking at 73.59% with a 70/30 split
- **Decision Tree** accuracy improved as test size increased, suggesting mild overfitting with smaller test sets

---

## 🛠️ Tech Stack

- Python 3.x
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/riyalawrance/Supervised_ml_classification_algorithms.git
cd Supervised_ml_classification_algorithms

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Open any notebook
jupyter notebook ques1_DT.ipynb
```

---

## 📁 Repository Structure

```
📦 Supervised_ml_classification_algorithms
 ┣ 📓 ques1_DT.ipynb       # Decision Tree implementation
 ┣ 📓 ques1_KNN.ipynb      # K-Nearest Neighbors implementation
 ┣ 📓 ques1_NBC.ipynb      # Naive Bayes implementation
 ┣ 📓 ques1_SVM.ipynb      # Support Vector Machine implementation
 ┣ 📄 Diabetes.csv         # Dataset
 ┗ 📄 README.md            # This file
```

---

## 📝 Evaluation Metrics Used

Each notebook includes:
- ✅ Accuracy Score
- ✅ Confusion Matrix (with heatmap)
- ✅ Classification Report (Precision, Recall, F1-Score)

---

## 🌱 Future Improvements

- [ ] Add feature scaling/normalization (especially for SVM and KNN)
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Add cross-validation for more robust evaluation
- [ ] Try ensemble methods (Random Forest, XGBoost)
- [ ] Deploy best model as a web app using Streamlit

---

## 👩‍💻 Author

**Riya Lawrance**  
CS Engineering Student | ML & Data Science Enthusiast  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/riya-lawrance-66993b2a8/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/riyalawrance)

---

⭐ If you found this useful, consider giving the repo a star!
