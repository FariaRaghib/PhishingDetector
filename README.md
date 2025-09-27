# 🛡️ Phishing Website Detection using Machine Learning

This project aims to detect **phishing websites** by applying **machine learning algorithms** on structured features extracted from URLs.  
It includes a **Streamlit web app** for live testing and visualization.  

---

.
├── README.md

├── requirements.txt

├── ML_models.py               # model training + helper load/save functions

├── feature_extraction.py      # feature functions and create_vector(soup)

├── app.py                     # Streamlit interactive app

├── verified_online.csv        # raw URLs used for scraping

├── structured_data_phishing.csv

├── structured_data_legitimate.csv

├── mini_dataset/              # saved HTML pages from scraping

│   ├── 0.html

│   ├── 1.html

|-- Mindmap


---

## ✨ Features
✅ **Web Scraping** – Extracts raw website data and URLs  
✅ **Feature Engineering** – Creates structured features such as:
- URL length, number of dots (`.`), presence of `@`
- Use of HTTPS / SSL certificate  
- Domain age and WHOIS info  
- External redirection links, JavaScript checks  
✅ **Machine Learning Models** – Multiple algorithms trained & compared:
- Random Forest
- Decision Tree
- AdaBoost
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Neural Network (MLP Classifier)
- K-Nearest Neighbours (KNN)  
✅ **Streamlit App** – User-friendly interface:
- Upload URL / dataset
- See phishing vs legitimate count
- Test prediction live
- Probability output for each class  
✅ **Visualization** – Dataset preview, model accuracy, feature importance  

---

## ⚙️ How It Works
1. **Data Collection**  
   - Web scraping was performed to gather phishing & legitimate URLs.  
   - Structured CSV datasets (`structured_data_phishing.csv`, `structured_data_legitimate.csv`) were created.  

2. **Feature Extraction**  
   - Features like URL length, SSL certificate, presence of suspicious symbols (`@`, `-`, `//`) were extracted.  
   - WHOIS lookups were used for domain-based features (e.g., domain age, expiry date).  

3. **Model Training**  
   - Data was split into train (80%) and test (20%).  
   - Models were trained on structured features.  
   - Performance metrics: Accuracy, Precision, Recall.  

4. **Prediction Flow in Streamlit App**  
   - User enters a URL.  
   - URL is transformed into feature vector.  
   - Selected ML model predicts whether it’s **Phishing (1)** or **Legitimate (0)**.  
   - Prediction probability is displayed.  

---

## 📊 Results
- All ML models achieved **high accuracy** on structured datasets.  
- **Random Forest, Decision Tree, and SVM** performed best.  
- Neural Network showed robust classification with generalization.  

---

## 🖥️ Streamlit App (Screenshots)
<img width="1279" height="597" alt="image" src="https://github.com/user-attachments/assets/b10df2ac-3746-4398-a45b-0b9ea1592196" />

<img width="1323" height="587" alt="image" src="https://github.com/user-attachments/assets/1d21db58-191f-4b47-9110-73b6a8c3baf0" />


---

## 🚀 Installation & Usage
### Clone the Repo
```bash

git clone https://github.com/your-username/Phishing_Detection.git
cd Phishing_Detection


2️⃣ Install Dependencies

pip install -r requirements.txt


3️⃣ Run Streamlit App

streamlit run app.py
