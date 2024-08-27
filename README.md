# Apartment rental- offers in Germany


# Running the scripts:
Before running the scripts, we need to create a virtual environment and install the packages:
- `python3 -m venv env`
- On Linux : `source env/bin/activate` - On windows : `env\scripts\activate.bat`
- `pip3 install -r requirements.txt`
- `jupyter notebook` To run the experiments.ipynb file.

# Scripts
- experiments.ipynb: 
This file has all the steps of data preparation and feature engineering, with explanations.
It also contains the experiments with the machine learning algorithms are found in the jupyter notebook (experiments.ipynb)
- ml_utils.py : it contains some helping functions related to machine learning algorithms.
- data_utils.py : it contains some helping functions related to data preparation.

# Steps:
- Data discovery and preparation.
- Textual data features using embeddings.
- Training machine learning models.
- Saving Models.

# Data discovery and preparation
In this step I looked at the correlation between the features, and the relationship between each feature and the price.

# Feature Engineering
Data preparation includes splitting data into train and test sets, and normalize data.

# Textual data features using embeddings
In order to find the features of text, we use a pretrained sentence embedding model 
that maps the text into a vector of a fixed size containing floating-point values.
The sentence embedding model is pretrained, meaning that it was trained previously 
on a huge corpus of sentences.
We used paraphrase-multilingual-MiniLM-L12-v2 as an embedding model for two reasons:
- It's a multilingual model, meaning that it was trained on multiple languages including German.
- The embedding size (384) isn't very large, that makes the machine learning training easier.

To add the textual data features into the structural data features, I calculated the embedding vector
for every text in description and facilities columns separately, then appended these embeddings 
into the structural data features.

# Using Chromadb to store the embeddings
Calculating the sentence embeddings is time-consuming, so I calculated them once and 
stored them in Chromadb vector database.

I created two collections, one called 'description' to store the vectors of 'description' columns, 
and another one for 'facilities'.
For every embedding vector, I used 'scoutId' column as an Id.

# Training models:
I experimented with 2 different regression models (LinearRegression - DecisionTree).
I chose these models for the following reasons:
- They are regression models and appropriate to our task that is about predicting a continuous value.
- They are easy to train.
- Training time isn't long.
- 
### Results:

I did two main experiments:
- Training the machine learning models using only the structural data (numbers).
- Training the models using the structural data (numbers) and textual data (description and facilities).

|                  | Linear Regression (no embeddings) | Linear Regression (with embeddings) | Decision Tree (no embeddings) | Decision Tree (with embeddings) 
|------------------|-----------------------------------|-------------------------------------|-------------------------------|---------------------------------|
| R2 score (train) | 0.79                              | 0.83                                | 0.99                          | 0.99                            |
| R2 score (test)  | 0.75                              | 0.83                                | 0.45                          | 0.69                            |

### Conclusions:
We observe that r2 scores for the models when including text embeddings are higher than the models without embeddings.
This is clear because the textual feature are providing rich information that contribute to the prediction of total rent values.


# Packages:
- For data preparation and visualization I used Pandas and Plotly.
- For machine learning algorithms I used Scikit-learn package.



