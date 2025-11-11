
This project implements an **unsupervised anomaly detection** solution for financial transactions using a Deep Learning model: the **Autoencoder**. The goal is to flag fraudulent transactions by measuring how poorly the model can reconstruct them.

---

## 💡 Core Methodology: The Autoencoder Principle

The Autoencoder is a neural network trained exclusively on **normal (non-fraudulent) transactions**. It learns to compress the data and reconstruct it perfectly.

When an **anomalous** (fraudulent) transaction is presented, the model fails to reconstruct the pattern it has never encountered, leading to a high **Reconstruction Error**, which serves as our **Fraud Risk Score**.



### **Anomaly Scoring: The Fraud Risk Score**

The core of the detection relies on the **Mean Squared Error (MSE)** between the input $X$ and the reconstructed output $\hat{X}$.

The **Fraud Risk Score** ($\mathcal{L}$) is calculated as follows:

$$\text{Fraud Risk Score } (\mathcal{L}) = \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (X_i - \hat{X}_i)^2$$

A transaction is flagged as fraud if its **Fraud Risk Score** exceeds a pre-determined **Threshold** (derived from the MSE distribution of normal training data).

### **Advanced Context: VAEs and KL Divergence**

For robustness and better-defined anomaly boundaries, the **Variational Autoencoder (VAE)** is often used. The VAE loss function includes a regularization term based on the **Kullback-Leibler (KL) Divergence**, which ensures the latent space follows a standard Gaussian distribution $p(z)$:

$$\mathcal{L}_{KL} = D_{KL}(q(z|X) \, || \, p(z))$$

This ensures the model learns a smooth and continuous manifold for normal data, improving the differentiation of anomalies.

---

## 💻 Streamlit Application: `app_simple.py`

An interactive web application is provided using **Streamlit** to facilitate model deployment and analysis in a user-friendly manner.

### **Features**

* **Real-time Analysis:** Performs prediction, scaling, and scoring instantly upon file upload.
* **Artifact Loading:** Securely loads the saved Keras model and `StandardScaler` (artifacts from the notebook).
* **User Control:** An interactive slider allows the user to adjust the **Risk Threshold** in the sidebar, instantly seeing the impact on the number of detected frauds.
* **Comprehensive Evaluation:** If the uploaded data includes a `TARGET` column (ground truth), the application displays the **Classification Report** and **Confusion Matrix** (as calculated in your notebook's final steps).

## 📂 Project Structure

| File/Folder | Description | Role in the Project |
| :--- | :--- | :--- |
| `Fraud_Detection (6).ipynb` | The source Jupyter Notebook, detailing data exploration, preprocessing, model definition, training, and threshold optimization. | **Training & Documentation.** |
| `app_simple.py` | The standalone Streamlit application script for inference. | **Deployment Interface.** |
| `autoencoder_fraud.h5` | **REQUIRED:** The saved Keras Autoencoder model weights and architecture. | **Prediction Engine.** |
| `scaler_fraud.pkl` | **REQUIRED:** The serialized `StandardScaler` object fitted on the training data. | **Data Preprocessing/Normalization.** |

---

## ⚙️ Setup and Execution

### 1. Prerequisites

Ensure you have downloaded and placed both `autoencoder_fraud.h5` and `scaler_fraud.pkl` into the root directory of this project.

### 2. Installation

Install all necessary dependencies using pip:

```bash
pip install pandas numpy tensorflow scikit-learn streamlit matplotlib seaborn

