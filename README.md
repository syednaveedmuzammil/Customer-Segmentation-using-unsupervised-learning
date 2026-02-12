# Customer Segmentation App (Streamlit)

An interactive **Streamlit web application** that performs **RFM-based customer segmentation** using **K-Means clustering**.  
This project helps businesses understand customer behavior and identify valuable customer segments using **unsupervised machine learning**.

---

## Features

- Upload your own e-commerce CSV dataset
- Automatic data cleaning & preprocessing
- RFM (Recency, Frequency, Monetary) analysis
- K-Means clustering with adjustable number of clusters
- PCA visualization for dimensionality reduction
- Interactive cluster visualizations
- Business interpretation of customer segments
- Download segmented customer data

---

## Machine Learning Techniques Used

- StandardScaler (Feature Scaling)
- K-Means Clustering
- PCA (Principal Component Analysis)
- Silhouette Score for cluster evaluation

---

## Expected Dataset Columns

The app works best with the **Online Retail Dataset** (UCI) or similar data having:

- `CustomerID`
- `InvoiceNo`
- `InvoiceDate`
- `Quantity`
- `UnitPrice`

---

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**

---

##  How to Run Locally

### 1️ Clone the repository
```bash
git clone https://github.com/<your-username>/customer-segmentation-streamlit.git
cd customer-segmentation-streamlit
