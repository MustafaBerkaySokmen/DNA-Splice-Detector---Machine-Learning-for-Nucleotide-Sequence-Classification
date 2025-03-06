# DNA Splice Detector - Machine Learning for Nucleotide Sequence Classification

## Overview
The **DNA Splice Detector** is a **Python-based machine learning project** designed to classify **nucleotide sequences** and detect potential splice sites. Using a **Naive Bayes classifier**, this model **estimates class probabilities** and computes **log-likelihood scores** to make predictions on DNA sequence data.

## Features
- **Supervised Learning Model**: Implements **Naive Bayes** classification.
- **Nucleotide Probability Estimation**: Computes conditional probabilities for each base (`A, C, G, T`).
- **Log-Likelihood Score Calculation**: Uses probabilistic inference for classification.
- **Performance Evaluation**: Generates a **confusion matrix** to measure accuracy.
- **Handles Large Datasets**: Efficiently processes thousands of nucleotide sequences.

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/DNA-Splice-Detector.git
cd DNA-Splice-Detector
```

### **2. Set Up a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate    # On Windows
```

### **3. Install Dependencies**
```bash
pip install numpy pandas
```

## Dataset
The dataset consists of:
- `hw01_data_points.csv` → **Nucleotide sequences** (Each row represents a DNA sequence).
- `hw01_class_labels.csv` → **Class labels** (Labels indicating splice site classification).

📌 **Should you upload the datasets?**
✅ **YES**, if:
- The dataset is **publicly available or open-source**.
- You have **permission to distribute it**.

🚫 **NO**, if:
- The dataset is **private, proprietary, or contains sensitive information**.
- You do not own the rights to share it.

If needed, you can include a **download link** instead of uploading the dataset.

## How It Works
1. **Loads the Dataset**
   - Reads **DNA sequences (`hw01_data_points.csv`)**.
   - Reads **class labels (`hw01_class_labels.csv`)**.

2. **Splits Data into Training & Testing Sets**
   - First **50,000 samples** for training.
   - Remaining **43,925 samples** for testing.

3. **Computes Class Prior Probabilities**
   - Estimates the probability distribution of each class.

4. **Estimates Nucleotide Probabilities**
   - Calculates the likelihood of `A, C, G, T` at each position **for each class**.

5. **Predicts Class Labels Using Log-Likelihood Scores**
   - Uses Naive Bayes classification principles.
   - Outputs **a confusion matrix** to evaluate accuracy.

## Running the Model
```bash
python dna_splice_detector.py
```

## Example Output
```
Loading dataset...
Training Naive Bayes Classifier...
Evaluating on test data...
Confusion Matrix:
[[5000  300]
 [ 250 5450]]
```
📌 **Rows** = Actual classes
📌 **Columns** = Predicted classes

## Future Enhancements
- **Deep Learning Model** (Use CNNs or RNNs for improved sequence classification).
- **Feature Engineering** (Incorporate k-mer embeddings for better predictions).
- **Multiple Classifications** (Extend beyond binary classification).

## License
This project is licensed under the **MIT License**.

## Contributions
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-new-feature`).
3. Commit and push your changes.
4. Open a pull request.

## Contact
For any questions or support, please open an issue on GitHub.

