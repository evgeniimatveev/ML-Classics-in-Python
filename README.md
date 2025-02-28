# 📊 ML Classics in Python (Google Colab)
**Collection of classic machine learning algorithms implemented in Python using Google Colab**

![ML](https://img.shields.io/badge/Machine_Learning-Python-blue) ![Status](https://img.shields.io/badge/Status-Active-green)

## 📌 Description
This repository provides implementations of **classic machine learning algorithms** in **Python**, structured into five parts. It serves as a comprehensive resource for learning data preprocessing, regression, classification, clustering, and association rule learning.

### Why this project?
- 📌 **Hands-on Learning**: Practical implementation of ML algorithms in Python.
- 📌 **Well-Structured Approach**: Organized into key ML categories for easy navigation.
- 📌 **Beginner & Advanced Friendly**: Covers fundamental and advanced ML techniques.
- 📌 **Educational Purpose**: Inspired by the SuperDataScience ML A-Z course.

## 📂 Project Structure  
```bash
ML-Classics-in-Python/
├── Part 1 - Data Preprocessing/     # Data preprocessing
├── Part 2 - Regression/             # Regression models
├── Part 3 - Classification/         # Classification models
├── Part 4 - Clustering/             # Clustering algorithms
├── Part 5 - Association Rule Learning/ # Association rule learning
├── data/                            # Datasets
├── README.md                        # Documentation
```

## 📚 Content  
### 🔹 **Part 1: Data Preprocessing**  
- 📌 Importing and cleaning data  
- 📌 Handling missing values  
- 📌 Encoding categorical data  
- 📌 Feature scaling  

### 🔹 **Part 2: Regression**  
- ✔ Simple Linear Regression  
- ✔ Multiple Linear Regression  
- ✔ Polynomial Regression  
- ✔ Support Vector Regression (SVR)  
- ✔ Decision Tree Regression  
- ✔ Random Forest Regression  

### 🔹 **Part 3: Classification**  
- ✔ Logistic Regression  
- ✔ K-Nearest Neighbors (KNN)  
- ✔ Support Vector Machine (SVM)  
- ✔ Decision Tree Classification  
- ✔ Random Forest Classification  

### 🔹 **Part 4: Clustering**  
- ✔ K-Means  
- ✔ Hierarchical Clustering  

### 🔹 **Part 5: Association Rule Learning**  
- ✔ Apriori  
- ✔ Eclat  

---

## 🚀 How to Use?  
### 🔧 Installation  
Ensure you have the required libraries installed before running the scripts:  

```python
!pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
```

### ▶ Running the Scripts  
1. Clone the repository:  
   ```bash
   git clone https://github.com/username/ML-Classics-in-Python.git
   cd ML-Classics-in-Python
   ```

2. Run the scripts in Google Colab:  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Open the required notebook:  
   ```python
   %cd '/content/drive/My Drive/Colab Notebooks/Part 2 - Regression'
   !jupyter notebook simple_linear_regression.ipynb
   ```

## 📌 Authors & Acknowledgments  
🔹 **Developed by:** **Evgenii Matveev**  
🔹 **Source:** **SuperDataScience Machine Learning A-Z (Python)**  
🔹 **For educational purposes only**  

🔹 **Special thanks** to the original authors of the SuperDataScience course – **Hadelin de Ponteves** and **Kirill Eremenko** for their contributions to ML education! 🙌

---

## 📜 License  
This project is distributed under the **MIT License**. Feel free to use the code! 🚀
