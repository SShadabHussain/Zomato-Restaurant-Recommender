# Zomato Bangalore Restaurants

## Business Case

Bangalore, the third most populous city in India, boasts a diverse array of restaurants representing global cuisines, making it a culinary haven for food enthusiasts. Over recent years, the city has witnessed a notable surge in the number of dining establishments, presenting a challenge in selecting an exceptional dining venue.

The objective of this endeavor is to discern the key attributes that define a remarkable restaurant and to develop a restaurant recommender system. This system aims to simplify the process of choosing the perfect dining destination, offering a more streamlined and informed decision-making experience for patrons.

> **Dataset:** [Zomato Bangalore Restaurants Dataset](https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants)

## Code Overview

```python
# Importing Libraries
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn import model_selection, preprocessing, decomposition, metrics, pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
import pyLDAvis.gensim
import gensim
from gensim import corpora
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px

# Loading Data
zomato = pd.read_csv('/Data/zomato.csv', delimiter=',')
ratings = pd.read_csv('/Data/ratings.csv', index_col=0)
locations = pd.read_csv('/Data/locations.csv')

# Exploratory Data Analysis and Cleaning
# (Details of data preprocessing and exploratory data analysis have been provided in the code)

# Sentiment Analysis
# (Details of sentiment analysis on Zomato reviews have been provided in the code)

# Topic Modelling using Gensim
# (Details of topic modelling have been provided in the code)

# Restaurant Recommender System
# (Details of the recommender system have been provided in the code)

# Visualization - Location Mapping
# (Details of location mapping visualization have been provided in the code)
```

## Results

- **Sentiment Analysis:**
  - Distribution plots of sentiment scores (Negative, Neutral, Positive, Compound) for Zomato reviews.
  - Word clouds for positively and negatively tuned reviews.

- **Topic Modelling:**
  - Implemented LDA (Latent Dirichlet Allocation) for topic modelling on Zomato reviews.

- **Restaurant Recommender System:**
  - Developed a restaurant recommender system using TF-IDF and cosine similarity.
  - Utilized the recommender system to suggest top restaurants similar to a given restaurant.

- **Location Mapping:**
  - Visualized restaurant distribution across different locations in Bangalore using Plotly Express.

## Conclusion

The project aims to provide valuable insights into the restaurant landscape in Bangalore, encompassing sentiment analysis, topic modelling, and a robust recommender system. These tools can assist both patrons and businesses in making informed decisions and enhancing the overall dining experience.

For a detailed walkthrough and code implementation, please refer to the provided Jupyter Notebook.
