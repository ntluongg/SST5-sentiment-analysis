# Sentiment Analysis Project

## Overview

This project focuses on **sentiment analysis**, which involves determining the emotional tone behind a body of text. Sentiment analysis has applications in various fields, including:

- Customer feedback analysis
- Social media monitoring
- Product reviews

### Problem Description

This is a multi-class classification task. Given a review sentence extracted from movie reviews, the model predicts the sentiment of the text. Sentiment labels include:

- Very Negative
- Negative
- Neutral
- Positive
- Very Positive

**Example:**
- **Text:** "The gorgeously elaborate continuation of 'The Lord of the Rings' trilogy is so huge that a column of words cannot adequately describe co-writer/director Peter Jackson's expanded vision of J.R.R. Tolkien's Middle-earth."
- **Label:** Very Positive

---

## Dataset

We used the **Stanford Sentiment Treebank (SST-5)**, which includes:

- **Labels:** Very Negative, Negative, Neutral, Positive, Very Positive
- **Size:** 11,855 single sentences extracted from movie reviews

---

## Methodology

### Data Preprocessing
- Stop words removal
- Lemmatization
- Cleaning patterns

### Models Evaluated
1. **FastText**
2. **XLNet**
3. **BERT (Base and Large)** 
4. **RoBERTa**
5. **ELECTRA** 
6. **ELECTRA + LCL Loss**

---

## Division of Tasks

| Task           | Assigned To |
|----------------|-------------|
| FastText       | An          |
| XLNet          | Duy         |
| ELECTRA        | Luong       |
| BERT (Large)   | Duong       |
| RoBERTa        | Cuong       |

---

## Results

The accuracy of the models on the SST-5 dataset is summarized below:

| Model                      | Accuracy (%) |
|----------------------------|--------------|
| FastText                   | 40.54        |
| XLNet                      | 56.4         |
| BERT (Base Uncased)        | 53.2         |
| RoBERTa (Large)            | 56.2         |
| ELECTRA                    | 57.0         |
| ELECTRA + LCL Loss         | 57.9         |
| BERT (Large Uncased FFT CE)| 55.0         |

---

## Conclusion

This project demonstrates the potential of transformer-based models like ELECTRA and RoBERTa in achieving higher accuracy for sentiment analysis compared to simpler methods like FastText. Future work can explore additional fine-tuning and the incorporation of ensemble methods to improve performance further.

---

## Authors

- **An**
- **Duy**
- **Luong**
- **Duong**
- **Cuong**
