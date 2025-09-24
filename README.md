# Sentiment Analysis on IMDb Movie Reviews 🎬

This project implements sentiment classification on the IMDb Movie Review dataset, predicting whether a review is positive or negative. The focus is on evaluating and comparing multiple traditional machine learning models using enhanced TF-IDF features.

## Dataset 

- **Source:** IMDb Movie Reviews Dataset  
- **Structure:** Two columns
  - `review` – the text of the movie review
  - `sentiment` – label: positive or negative  

The dataset was preprocessed by cleaning HTML tags, special characters, and extra spaces. Texts were also lowercased for consistency.

## Data Preprocessing 

**Text Cleaning:**
- Remove HTML tags and special characters.
- Convert all text to lowercase.
- Remove extra spaces.

**Label Encoding:**
- positive → 1
- negative → 0

**Data Split:**
- Train: 64%
- Validation: 16%
- Test: 20%

## Feature Engineering 

**TF-IDF Vectorization with enhancements:**
- Unigrams, bigrams, trigrams (ngram_range=(1,3))
- Vocabulary limited to top 30,000 tokens
- Sublinear TF weighting (log(1+tf))

This captures richer context and phrase-level information.

## Models Implemented 

**Logistic Regression**
- Maximum iterations: 2000
- Random state: 42

**Linear Support Vector Machine (SVM)**
- Maximum iterations: 5000
- Random state: 42

Both models were trained on the enhanced TF-IDF features.  
**TF-IDF (Term Frequency–Inverse Document Frequency)** is a numerical representation of text that reflects how important a word is to a document in a collection or corpus.

## Evaluation Metrics 

Models were compared using multiple metrics on validation and test sets:

| Metric    | Definition                                         |
|-----------|---------------------------------------------------|
| Accuracy  | Overall correct predictions                        |
| Precision | Correct positive predictions among predicted positives |
| Recall    | Correct positive predictions among actual positives |
| F1-score  | Harmonic mean of Precision and Recall             |

## Validation Results

**Logistic Regression**
- Accuracy: 0.9055
- Precision: 0.8983
- Recall: 0.9145
- F1-score: 0.9063

**SVM**
- Accuracy: 0.9021
- Precision: 0.8973
- Recall: 0.9083
- F1-score: 0.9027

**Observations:** Both models perform similarly on the validation set.

**Confusion Matrices were plotted to visualize predictions:**
- Logistic Regression (Green)  
  ![Logistic Regression Validation Confusion Matrix and metrics](images/lr_cm_metrics.png)  
- SVM (Blue)  
  ![SVM Validation Confusion Matrix and metrics](images/svm_cm_metrics.png)  

## Training vs Validation Performance

Accuracy and F1-score comparison shows that both models generalize well, with no significant overfitting:

**Accuracy Plot:**  
![Training vs Validation Accuracy](images/train_vs_val_accuracy_plot.png)  

**F1-score Plot:**  
![Training vs Validation F1-score](images/train_vs_val_f1score_plot.png)  

Training and validation bars include exact scores for clarity.

## Test Set Performance

| Model               | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9069   | 0.9022    | 0.9128 | 0.9074  |
| SVM                | 0.9101   | 0.9102    | 0.9100 | 0.9101  |

**Confusion Matrices on Test Set:**
- Logistic Regression (Green) and SVM (Blue) 
  ![Test Confusion Matrix](images/test_cm.png)  
- Logistic Regression vs SVM metrics
  ![Test Metrics Comparison](images/lr_svm_test_result_metrics.png)

Both models maintain high performance on unseen data, confirming stability.

## Comparison between Models

- Logistic Regression  vs SVM  
  ![Metrics](images/lr_vs_svm.png)

## Insights

- Enhanced TF-IDF features (ngrams + sublinear TF) are effective for sentiment classification.
- Logistic Regression slightly outperforms SVM, but both are strong baselines.
- Accuracy, precision, recall, and F1-scores are consistently high (>89%), showing that traditional ML models still work well for text classification tasks.

## Conclusion

This project demonstrates a robust pipeline for text-based sentiment classification:

- Text preprocessing
- Feature extraction using enhanced TF-IDF
- Training and evaluation of Logistic Regression and SVM
- Comprehensive metric comparison with visualization

The results confirm that SVM and Logistic Regression can serve as strong baselines before moving to deep learning approaches.


## Author
Srijani Das
