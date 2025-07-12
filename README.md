# 🎬 Movie Success Predictor (Python + scikit-learn)

This project implements a console-based machine learning model that predicts a movie’s box office success using metadata such as budget, cast, director, genres, and more. It demonstrates a complete end-to-end ML pipeline with data cleaning, feature extraction, model training, and an interactive CLI-based prediction tool.

---

## 📖 Project Overview

This Python-based machine learning project is built to predict whether a movie will be **successful** (i.e., revenue > 1.5 × budget) based on structured and semi-structured metadata. It is trained on the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and supports:

* JSON parsing of complex fields like `cast`, `crew`, `genres`
* Feature engineering from release dates and language
* A `soup`-based vectorization method for combining categorical inputs
* A console tool for step-by-step predictions using custom inputs

---

## 🚀 Features

* Merges and processes two raw datasets (`movies.csv` + `credits.csv`)
* Extracts director, top cast, genres, and keywords from JSON-style fields
* Adds release date features: year, month, and weekend flag
* Builds a machine learning pipeline with:

  * **CountVectorizer** for text
  * **StandardScaler** for numerics
  * **LabelEncoder** for language
* Trains a **Random Forest Classifier** with \~75% accuracy
* Includes a CLI-based prediction script using user input or default values

---

## 🔧 Technologies Used

* **Language:** Python 3.10
* **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
* **Model:** RandomForestClassifier
* **Vectorizer:** CountVectorizer (bag of words for genres/keywords/etc.)
* **Scaler & Encoder:** StandardScaler, LabelEncoder
* **Interaction:** Command-line input via `input()` in a loop

---

## 🧠 Model Input Features

| Feature              | Type        | Description                               |
| -------------------- | ----------- | ----------------------------------------- |
| `budget`             | Numerical   | Movie's budget in USD                     |
| `popularity`         | Numerical   | Popularity metric from TMDB               |
| `runtime`            | Numerical   | Duration in minutes                       |
| `original_language`  | Categorical | Encoded via `LabelEncoder`                |
| `release_year`       | Numerical   | Extracted from `release_date`             |
| `release_month`      | Numerical   | Extracted from `release_date`             |
| `is_weekend_release` | Boolean     | `1` if released on Sat/Sun                |
| `soup`               | Text        | Combined genres, keywords, cast, director |

---

## 🧪 Sample Prediction CLI (Interactive)

```
🎬 Movie Success Predictor
Enter movie details one by one. Leave blank to use defaults.

Movie Title: Space Avengers
Budget (in USD): 180000000
Popularity score (0-100): 60.2
Runtime (in minutes): 125
Original language (e.g. 'en'): en
Release year: 2025
Release month (1-12): 7
Weekend release? (1 for yes, 0 for no): 1

Movie soup (space-separated): Action Marvel Space ChrisEvans Russo

🎯 Prediction for 'Space Avengers': ✅ Successful
```

---

## 🗂️ Repository Structure

```
📁 data/
    ├── movies.csv
    └── credits.csv

📁 model/
    ├── movie_success_model.pkl
    ├── scaler.pkl
    ├── vectorizer.pkl
    └── language_encoder.pkl

📁 notebooks/
    └── movie_success_training.ipynb

📄 predict_interactive.py
📄 batch_predict.py
```

---

## 📄 Notebook

The Jupyter notebook that builds and trains the model is available here:

👉 [`notebooks/movie_success_training.ipynb`](notebooks/movie_success_training.ipynb)

---

## 👥 Author

* [Muhammad Ibrahim Abdullah](https://github.com/Ibrahim5570)

---

## 🔖 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for full details.

---
