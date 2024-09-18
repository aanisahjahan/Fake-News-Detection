# Report on Fake News Detection Using SVM and TF-IDF
## Overview
This project aims to classify news articles based on their stance relative to a given headline. The classification task involves determining whether the article "agrees," "disagrees," "discusses," or is "unrelated" to the headline. Given the limited size of the dataset, this project uses a Support Vector Machine (SVM) with Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to extract features and perform classification.
## Table of Contents
1. Dataset
   
2. Methodology
  - Preprocessing the Data
  - Handling Missing or Non-String Data
  - Text Vectorisation Using TF-IDF
  - Model Training and Evaluation

3. Challenges and Solutions
   
4. Results
   
5. Conclusion
   
6. How to Run
    
7. Future Work
## Dataset
The dataset used in this project consists of a small set of news articles. Each article includes:

- Headline: A brief description of the news article.
- Stance: The stance of the article body relative to the headline (e.g., "unrelated," "discuss").
- Article Body: The main content of the news article.
## Methodology
### Preprocessing the Data
Preprocessing is a crucial step in Natural Language Processing (NLP) to ensure the data is clean and consistent. The following preprocessing steps were applied to the text data:

1. Text Cleaning: Removed non-word characters and extra spaces using regular expressions to ensure only relevant text is retained.
   
3. Lowercasing: Converted all text to lowercase to maintain uniformity and reduce the feature space.
   
5. Tokenization: Split the text into individual words (tokens) to facilitate further processing.
   
7. Stopword Removal: Removed common English stopwords using the NLTK library to focus on more informative words.
### Handling Missing or Non-String Data
It is crucial to handle any missing or non-string values in the "Stance" column. This was done by replacing any non-string or missing entries with the default value "unrelated."
### Text Vectorisation Using TF-IDF
To convert the preprocessed text data into numerical features, we used TF-IDF vectorization. TF-IDF helps to represent text data by evaluating the importance of words within the context of the document and the entire corpus. This approach balances the frequency of terms in a document against their frequency across all documents, highlighting more informative words.
### Model Training and Evaluation
Given the dataset's size constraints, we used the entire dataset for training and manual evaluation. The model of choice was a Support Vector Machine (SVM) due to its effectiveness in high-dimensional spaces and its ability to handle sparse data such as TF-IDF vectors. We performed the following steps:

Training the Model: The SVM was trained using the TF-IDF features extracted from the combined text of headlines and article bodies.

Manual Evaluation: Since traditional cross-validation was not feasible with such a small dataset, we evaluated the model on the same data it was trained on.
## Challenges and Solutions
### Challenges:
1. Small Dataset Size: The extremely limited dataset size prevented the use of standard evaluation methods like train-test split and cross-validation. Splitting the data would result in having only one class in the training set, which is problematic for SVM training.
   
2. Overfitting: With only two samples, the model was likely to memorize the training data, resulting in perfect accuracy on the training set but a lack of generalizability.
   
3. Evaluation Limitations: Using the entire dataset for training and testing does not provide an accurate measure of the model’s performance on unseen data.
### Solution:
We opted to use the entire dataset for training and performed a manual evaluation. Although this is not ideal, it allowed us to at least test the process and observe how the model behaves with this limited data.
## Results
The SVM model, when trained and evaluated on the same small dataset, achieved perfect accuracy. However, this result is expected and not indicative of the model’s ability to generalize to new data. The perfect accuracy suggests that the model has likely overfitted to the small dataset, memorizing the training samples rather than learning generalizable patterns.
## Conclusion
This project demonstrated the process of applying machine learning to fake news detection using SVM and TF-IDF. While the model achieved perfect accuracy on the training data, this outcome was primarily due to the limited size of the dataset. The project highlighted the importance of having a sufficiently large and diverse dataset for training and evaluating machine learning models, particularly for tasks like fake news detection that require a nuanced understanding of text.
## How to Run
1. Install Dependencies: Ensure you have all necessary Python packages installed, including pandas, nltk, scikit-learn, and re.

2. Load the Dataset: Place the dataset file fakenews.xlsx in the same directory as the script or modify the path in the code.

3. Run the Script: Execute the script to preprocess the data, train the SVM model, and perform manual evaluation.

4. Review the Results: The script will print out the evaluation results based on the training data.
## Future Work
Given the limitations faced in this project, several areas for future work are identified:

1. Dataset Expansion: Gathering a larger, more diverse dataset would enable a more robust evaluation and better model generalization.

2. Advanced Models: Exploring advanced text representation techniques like word embeddings (e.g., Word2Vec, GloVe) or transformers (e.g., BERT) could provide improved performance in stance detection.

3. Proper Evaluation: With a larger dataset, implementing proper train-test splits or cross-validation would provide a more reliable assessment of model performance.

4. Ethical Considerations: Investigating the ethical implications of fake news detection models, including bias mitigation and interpretability, to ensure responsible AI development.
