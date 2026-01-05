# Climate Change Modeling

This project analyzes climate-related data from NASA and other sources to perform exploratory data analysis (EDA), sentiment analysis, topic modeling, and prediction of engagement metrics (likes/comments) for climate-related text data. The goal is to gain insights into climate discussions and make simple predictions using machine learning techniques.

---

## Project Structure

climate_change_modeling/
│
├── data/
│ ├── raw/ # Raw datasets
│ └── processed/ # Cleaned and preprocessed datasets
│
├── notebooks/ # Jupyter notebooks
│ ├── 01_eda.ipynb
│ ├── 02_text_preprocessing.ipynb
│ ├── 03_sentiment_model.ipynb
│ └── 04_topic_modeling.ipynb
│
├── src/ # Source code modules
│ ├── preprocessing.py
│ ├── sentiment_model.py
│ ├── topic_model.py
│ └── evaluation.py
│
├── src/models/ # Saved ML models (joblib)
├── requirements.txt
└── README.md

markdown
Copy code

---

## Features

1. **Exploratory Data Analysis (EDA)**
   - Visualize trends and patterns in climate data.
   - Understand relationships between climate variables (temperature, precipitation, CO2 levels, etc.).

2. **Text Preprocessing**
   - Clean text by lowercasing, removing punctuation, URLs, and numbers.
   - Tokenize and remove stopwords (if needed).

3. **Sentiment Analysis**
   - Predict sentiment (`positive`, `neutral`, `negative`) for climate-related text.
   - Uses a trained TF-IDF vectorizer and RandomForest classifier.
   - Visualize sentiment distribution.

4. **Topic Modeling**
   - Identify main discussion topics using LDA or NMF models.
   - Assign topic IDs to text entries for further analysis.

5. **Prediction Models**
   - Predict engagement metrics such as `likesCount` or `commentsCount`.
   - Uses features like text length, sentiment, and topic.
   - Employs RandomForestRegressor for prediction.

6. **Visualizations**
   - Sentiment distribution charts
   - Topic distribution charts
   - Model evaluation plots (confusion matrices, RMSE, etc.)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/KarthiKeyanZz/climate_change_modeling
cd climate_change_modeling
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install required packages:

bash
Copy code
pip install -r requirements.txt
(Optional) Download NLTK resources:

python
Copy code
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Usage
Launch Jupyter Notebook:

bash
Copy code
jupyter lab
# or
jupyter notebook
Open notebooks in the notebooks/ folder:

01_eda.ipynb: Exploratory Data Analysis

02_text_preprocessing.ipynb: Text cleaning and tokenization

03_sentiment_model.ipynb: Sentiment modeling and predictions

04_topic_modeling.ipynb: Topic modeling analysis

Run the cells sequentially to reproduce the analysis and predictions.

Predictions
Sentiment Prediction: Classify text as positive, neutral, or negative.

Engagement Prediction: Predict likes/comments for text entries using RandomForestRegressor.

Scenario Modeling: (Future improvement) Use topic and sentiment to analyze potential engagement trends.

License
This project is for educational and research purposes.

Acknowledgements
NASA climate datasets

NLTK, TextBlob, Scikit-learn, Prophet for ML and NLP

Inspiration from climate change research projects and EDA workflows