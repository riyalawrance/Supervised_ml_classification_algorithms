# Diabetes Prediction using Machine Learning
This repository contains machine learning implementations for predicting diabetes using the Pima Indians Diabetes dataset. The project compares four classification approaches: Decision Trees, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Naive Bayes.
## 📁 Files in Repository

- Diabetes.csv - The Pima Indians Diabetes dataset
- DT.ipynb - Decision Tree classifier implementation
- SVM.ipynb - SVM classifier implementation
- KNN.ipynb - K-Nearest Neighbors classifier implementation
- NB.ipynb - Naive Bayes classifier implementation
- README.md - This file

## 📊 Dataset
The dataset contains medical diagnostic measurements for Pima Indian women, with features including:

- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age

Target Variable: Binary outcome (0 = No Diabetes, 1 = Diabetes)
## 🔍 Implementations
### 1. Decision Tree Classifier (ques1_DT.ipynb)
#### Implements a Decision Tree classifier with three different train-test split ratios:

- Test Size 0.2 (80% train, 20% test)
    - Accuracy: 66.88%
- Test Size 0.3 (70% train, 30% test)
    - Accuracy: 69.70%
- Test Size 0.4 (60% train, 40% test)
    - Accuracy: 70.78%

 #### Key Features:

- Non-parametric algorithm
- Handles both numerical and categorical data
- Prone to overfitting without pruning

### 2. Hard Margin SVM Classifier (ques1_SVM.ipynb)
#### Implements a linear SVM with hard margin (high C value = 10,000) with three different train-test split ratios:

-Test Size 0.2 (80% train, 20% test)
  -Accuracy: 29.22%
-Test Size 0.3 (70% train, 30% test)
  -Accuracy: 35.06%
-Test Size 0.4 (60% train, 40% test)
  -Accuracy: 57.47%

#### Key Features:
- Linear kernel SVM
- Hard margin approach (C=10,000, max_iter=100,000)
- Convergence warnings indicating need for feature scaling

### 3. K-Nearest Neighbors Classifier (ques1_KNN.ipynb)
#### Implements KNN algorithm with k=5 neighbors across three different train-test split ratios:

- Test Size 0.2 (80% train, 20% test)
  - Accuracy: 69.48%
- Test Size 0.3 (70% train, 30% test)
  - Accuracy: 73.59%
- Test Size 0.4 (60% train, 40% test)
  - Accuracy: 69.81%

#### Key Features:
- Instance-based learning
- K=5 neighbors configuration
- No explicit training phase
- Sensitive to feature scaling

### 4. Naive Bayes Classifier (ques1_NB.ipynb)
#### Implements Gaussian Naive Bayes classifier with three different train-test split ratios:

- Test Size 0.2 (80% train, 20% test)
  - Accuracy: To be added
- Test Size 0.3 (70% train, 30% test)
  - Accuracy: To be added
- Test Size 0.4 (60% train, 40% test)
  - Accuracy: To be added

#### Key Features:
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast training and prediction
- Works well with small datasets

## 📈 Performance Comparison
AlgorithmTest Size 0.2Test Size 0.3Test Size 0.4Best AccuracyDecision Tree66.88%69.70%70.78%70.78%SVM (Hard Margin)29.22%35.06%57.47%57.47%KNN (k=5)69.48%73.59%69.81%73.59%Naive BayesTBATBATBATBA
Current Winner: K-Nearest Neighbors with 73.59% accuracy at test size 0.3
🛠️ Requirements
python- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
💻 Installation
bashpip install pandas numpy scikit-learn matplotlib
🚀 Usage
python# Example for any classifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # or KNeighborsClassifier, GaussianNB, SVC

# Load data
data = pd.read_csv('Diabetes.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train and predict
clf = DecisionTreeClassifier()  # or other classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
📝 Key Observations
Strengths and Weaknesses:
Decision Tree:

✅ Easy to interpret and visualize
✅ Handles non-linear relationships
❌ Prone to overfitting
❌ Sensitive to small data variations

SVM:

✅ Effective in high-dimensional spaces
✅ Memory efficient
❌ Poor performance without proper scaling
❌ Requires careful hyperparameter tuning

KNN:

✅ Simple and intuitive
✅ No training phase
✅ Best performance in this comparison
❌ Computationally expensive for large datasets
❌ Sensitive to irrelevant features

Naive Bayes:

✅ Fast and efficient
✅ Works well with small datasets
✅ Performs well with independent features
❌ Assumes feature independence (often unrealistic)

🎯 Future Improvements

Feature Engineering:

Feature scaling (StandardScaler/MinMaxScaler)
Feature selection
Handle missing values and outliers


Model Optimization:

Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
Cross-validation for robust performance estimation
Try ensemble methods (Random Forest, Gradient Boosting)


SVM Improvements:

Implement feature scaling
Try different kernels (RBF, polynomial)
Adjust C and gamma parameters


Advanced Techniques:

Ensemble methods (Voting, Bagging, Boosting)
Neural Networks
Feature importance analysis



📊 Evaluation Metrics
Each implementation includes:

Accuracy Score: Overall correctness
Confusion Matrix: True/False Positives/Negatives
Classification Report: Precision, Recall, F1-Score for each class
Visual Confusion Matrix: Heatmap representation

👥 Contributing
Feel free to fork this repository and submit pull requests for improvements.
📄 License
This project is open source and available for educational purposes.
