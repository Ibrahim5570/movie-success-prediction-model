# 🎬 Movie Success & Budget Predictor
# Python • scikit-learn • TMDB Dataset

This project is an end-to-end machine learning system that predicts movie success,
audience approval, and suggested budget using structured metadata and text features
such as genres, cast, director, and production companies.

It includes:
- Data preprocessing and feature engineering
- Multiple trained ML models
- Two interactive CLI prediction tools
- Saved models for reuse and deployment


# 📖 Project Overview

Using the TMDB 5000 Movies & Credits dataset, this project builds machine learning
models to answer:

1. Will this movie be financially successful?
   (Revenue > 1.5 × Budget)

2. Will audiences like it?
   (Vote average ≥ 6)

3. What budget range makes sense for this movie concept?
   (Regression-based estimate)

The system combines numerical features, categorical encodings, and a text
“soup” representation into a single feature space, then trains Random Forest
models for prediction.


# 🚀 Features

## Data Processing
- Merges movies.csv and credits.csv
- Parses JSON fields:
  - genres
  - keywords
  - cast (top 3 actors)
  - crew (director extraction)
- Drops post-release leakage fields when required
- Handles missing values and invalid budgets/revenues

## Feature Engineering
- Release date features:
  - Year
  - Month
  - Weekend release flag
- Text “soup” combining:
  - Genres
  - Keywords
  - Production companies
  - Cast
  - Director
- Encodes:
  - Language (LabelEncoder)
  - Text (CountVectorizer, 5,000 features)
  - Numerics (StandardScaler)


# 🤖 Models Trained

## Movie Success Classifier
Type: RandomForestClassifier  
Target:
  success = revenue > 1.5 × budget

## Audience Approval Classifier
Type: RandomForestClassifier  
Target:
  audience_liked = vote_average ≥ 6

## Budget Recommendation Model
Type: RandomForestRegressor  
Target:
  budget (USD)

Each model is trained on a shared feature space and saved for reuse.


# 📊 Input Features

## Numerical Features
- budget
- popularity
- runtime
- original_language (encoded)
- release_year
- release_month
- is_weekend_release
- vote_average
- vote_count

## Text Features (Soup)
- Genres
- Keywords
- Production companies
- Top cast
- Director

Vectorized using CountVectorizer (Bag of Words).


# 🧪 Interactive CLI Tools

## Full Prediction CLI
Predicts:
- Financial success
- Audience approval
- Probabilities for each outcome

Script:
  predict_interactive.py

Example output:
  🎬 Dune Part Three — ✅ Success, 👍 Liked by audience
  🔥 Financial Success Probability: 82.4%
  ⭐ Audience Approval Probability: 76.1%


## Budget + Success CLI
Predicts:
- Success / flop
- Probability
- Suggested budget range

Script:
  budget_predictor.py

Example output:
  🎬 Galactic Wars Prediction: ✅ Success
  🔥 Probability: 78.2%
  💰 Suggested Budget: $145,000,000


# 🗂️ Repository Structure

data/
├── movies.csv
└── credits.csv

model/
├── movie_success_model.pkl
├── audience_model.pkl
├── success_model.pkl
├── budget_model.pkl
├── vectorizer.pkl
├── scaler.pkl
└── language_encoder.pkl

train_models.py
predict_interactive.py
budget_predictor.py
README.md


# ⚙️ Tech Stack

- Python 3.10
- pandas / numpy
- scikit-learn
- joblib
- Random Forest (Classifier & Regressor)


# ⚠️ Important Notes

- Some models use post-release features (votes, popularity)
  and are intended for analysis, not pre-release forecasting.
- Budget predictions are advisory only.
- Unknown languages default to English encoding.


# 👤 Author

Muhammad Ibrahim Abdullah
GitHub: https://github.com/Ibrahim5570


# 📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.
