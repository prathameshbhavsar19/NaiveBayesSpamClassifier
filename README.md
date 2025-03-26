# PySpark Naive Bayes Spam Classifier

This project implements a custom Naive Bayes text classification model using PySpark RDDs. It classifies SMS messages into `spam` or `ham` without using any machine learning libraries.

## Features
- Built using MapReduce logic on PySpark
- No external ML libraries used
- Handles CSV with quotes, commas, and messy text
- Implements Laplace smoothing and tokenization

## Dataset
- Source: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format:  
  ```
  label,message
  ham,I'm on my way
  spam,WINNER!! You have won $1000...
  ```

## Workflow
1. Load and preprocess data using Spark DataFrames
2. Tokenize and clean messages
3. Compute class priors and word likelihoods manually
4. Predict labels for test messages using log-sum of probabilities

## Final Model Performance
```
Accuracy: 98.43%  
P(ham) = -0.1465 (Count: 3874)  
P(spam) = -1.9934 (Count: 611)
```

## Requirements
- Python 3.x
- PySpark
- Spark cluster or Databricks environment

## Run It
```python
spark.read.csv("spam.csv")...
# then run Naive Bayes logic from script
```

## License
MIT License
