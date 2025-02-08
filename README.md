# 🎬 Netflix Movie or Series  

📌 **Netflix Movie or Series** is a machine learning classification model that predicts whether a given content on Netflix is a **movie or a TV show** based on various features.  
Additionally, the project incorporates **unsupervised learning methods** such as **K-Means clustering** and **PCA for dimensionality reduction** to enhance the analysis.  

## 🚀 Features  
✅ **Data Preprocessing** (Handling missing values, encoding categorical variables, feature scaling)  
✅ **Machine Learning Classification Models** (RandomForest, Logistic Regression, Gradient Boosting)  
✅ **Hyperparameter Tuning** (GridSearchCV for optimal model performance)  
✅ **Cross-Validation & Model Evaluation** (Accuracy, Precision, Recall, F1-Score)  
✅ **Unsupervised Learning** (K-Means clustering & PCA visualization)  

## 📂 Dataset  
The dataset used in this project is `netflix_titles.csv`, which contains metadata about Netflix content such as title, director, cast, country, release year, rating, and more.  

## 💻 Installation  

Clone the repository:  
```bash
git clone https://github.com/laragunes/netflix-movie-or-series.git  
cd netflix-movie-or-series
pip install -r requirements.txt
```

Usage
```bash
python mlproject.py
```


📊 Model Performance:
The model is evaluated using cross-validation and metrics such as accuracy, precision, recall, and F1-score.
Hyperparameter tuning is applied using GridSearchCV to optimize model performance.

📉 Unsupervised Learning:
In addition to classification, the project applies unsupervised learning techniques:

K-Means Clustering to group similar content
PCA (Principal Component Analysis) for dimensionality reduction and visualization

Technologies Used:
Python
Pandas & NumPy (Data handling)
Scikit-Learn (Machine learning models)
Matplotlib & Seaborn (Data visualization)
GridSearchCV (Hyperparameter tuning)







