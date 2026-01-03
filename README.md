# CorePulse – Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange)]()
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.3.2-blueviolet)](https://tailwindcss.com/)

---

## 🚀 Project Overview

**CorePulse** is an end-to-end Machine Learning web application that predicts **heart disease risk** using patient and clinical data. The system combines a trained ML model with a modern, responsive frontend, making it easy for users to input data and view risk predictions in real-time.

This project demonstrates full-stack ML deployment with a focus on **accuracy, usability, and professional UI design**.

---

## 📊 Dataset

* Trained on the **Heart Failure Prediction Dataset** sourced from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
* Contains **918 patient records** with clinical and demographic features
* Preprocessing includes:

  * Cleaning missing values
  * Encoding categorical features
  * Scaling numerical values

---

## �� Frontend (UI/UX)

* Built as a **single-page application** with **HTML, Tailwind CSS, and Vanilla JavaScript**
* Dark **neo-glass / cyber-inspired design** for a professional medical-tech aesthetic
* Responsive layout optimized for **desktop and large screens**
* Real-time form handling with **High Risk / Low Risk** visualization
* Smooth UI transitions and interactive feedback states

---

## 🧠 Machine Learning & Backend

* **ML Model:** Logistic Regression
* **Backend:** Flask (Python)
* Feature encoding and scaling using persisted scaler and column mapping
* REST-based prediction endpoint (`/predict`)
* Structured JSON response handling for seamless frontend integration

---

## ⚙️ Features

* Interactive patient data input form
* Scaled and validated ML predictions
* Clear visual risk classification indicators
* Publicly deployed and accessible via Hugging Face Spaces

---

## 🔗 Live Demo

[CorePulse on Hugging Face Spaces](https://huggingface.co/spaces/MoizMalik/CorePulse)

## 💻 GitHub Repository

[https://github.com/moizmalik13588/CorePulse](https://github.com/moizmalik13588/CorePulse)

---

## 📝 Key Learnings

* End-to-end **frontend–backend integration** for ML applications
* Working with **real-world medical datasets**
* Deploying **Flask-based ML models** on Hugging Face Spaces
* Structuring maintainable, production-ready ML pipelines

---

## 🎯 Future Improvements

* Display **prediction probability scores**
* Implement **model comparison** for improved accuracy
* Enhanced **medical data visualization**

---

## 🔧 Installation & Setup (Optional for Local)

```bash
# Clone the repo
git clone https://github.com/moizmalik13588/CorePulse.git
cd CorePulse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## ⚠️ Disclaimer

This project is **for educational purposes only**. It should **not** be used as a medical diagnosis tool. Always consult a licensed healthcare professional for medical advice.

---

## 💥 Technologies

* **Languages:** Python, HTML, JavaScript
* **Frameworks:** Flask, Tailwind CSS
* **ML Model:** Logistic Regression
* **Deployment:** Hugging Face Spaces

---

## 📌 Hashtags / Tags

`#MachineLearning #DataScience #Kaggle #Python #Flask #TailwindCSS #JavaScript #AI #HealthTech #HeartDisease #FullStack #MLDeployment`
