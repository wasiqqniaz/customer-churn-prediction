# Customer Churn Analysis with RFM Segmentation

This project performs **end-to-end churn analysis** on the Olist Brazilian E-commerce dataset using **RFM segmentation**, **exploratory data analysis**, **feature engineering**, and **machine learning models** for churn prediction.

---

## Dataset Relationship Diagram

![Olist Dataset Relationship Diagram](assets/olist_schema.png)

---

## Project Overview

The goal of this project is to analyze **customer behavior**, identify **churn patterns** using **RFM (Recency, Frequency, Monetary) segmentation**, and build **machine learning models** to predict customers at risk of leaving.

### This repository includes:
- Comprehensive **EDA** across orders, customers, sellers, products, and payments  
- **RFM scoring** to segment customers into behavioral groups  
- **Feature engineering** from transactional and behavioral data  
- **Machine learning modeling** for churn prediction  
- Visual insights to support **business retention strategies**

---

## Features

### 1. Exploratory Data Analysis (EDA)
- Order lifecycle analysis  
- Customer purchase behavior trends  
- Seller performance evaluation  
- Payment and product-level analytics  
- Delivery delays, review scores, and customer experience indicators  

### 2. RFM Segmentation
- **Recency** – Time since last purchase  
- **Frequency** – Number of orders  
- **Monetary** – Total customer spending  
- RFM scoring and customer segmentation (Champions, At-Risk, Lost)

### 3. Churn Definition
- Churn is defined based on **customer inactivity** within a configurable business timeframe.

### 4. Machine Learning Modeling
- Data preprocessing  
- Train-test split  
- Logistic Regression  
- Random Forest  
- XGBoost (or similar models)  
- Model evaluation using **Accuracy, F1-score, and ROC-AUC**

### 5. Visualizations
- Customer lifetime value distribution  
- RFM segment analysis  
- Churn vs non-churn comparison  
- Model performance metrics

---

## Repository Structure

├── data/                     # Raw and cleaned datasets

├── notebooks/                # EDA, RFM, and modeling notebooks

├── src/                      # Python modules for processing

├── assets/                   # Images and diagrams (place your PNG here)

├── models/                   # Saved ML models

├── README.md                 # Project documentation

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-Learn
- Jupyter/Colab
- XGBoost

---

## Results Summary

- RFM segmentation clearly identifies customer value tiers
- High-churn segments strongly correlate with poor delivery performance
- Machine learning models show strong predictive performance (varies per run)


## Future Enhancements

- Deploy the model as an API
- Build an interactive dashboard (Streamlit / Power BI)
- Add time-series forecasting for customer lifetime value

---

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt

Run the main notebook:

notebooks/01_olist_churn_analysis.ipynb

---

