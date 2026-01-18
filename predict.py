<<<<<<< HEAD
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# === Load Model and Tools ===
model = joblib.load('model/movie_success_model.pkl')
audience_model = joblib.load('model/audience_model.pkl')
scaler = joblib.load('model/scaler.pkl')
cv = joblib.load('model/vectorizer.pkl')
le = joblib.load('model/language_encoder.pkl')

while True:
    print("\n🔍 Enter movie details (or type 'exit' to quit):")
    title = input("Movie title: ")
    if title.lower() == 'exit':
        break

    try:
        budget = int(input("Budget (USD): "))
        popularity = float(input("Popularity score (e.g. 57.8): "))
        runtime = float(input("Runtime (minutes): "))
        language = input("Original language (e.g. en): ")
        release_year = int(input("Release year: "))
        release_month = int(input("Release month (1-12): "))
        is_weekend = input("Is it a weekend release? (y/n): ").lower() == 'y'
        vote_avg = float(input("Vote average (IMDb style): "))
        vote_count = int(input("Vote count: "))

        genres = input("Genres (comma-separated): ").split(',')
        keywords = input("Keywords (comma-separated): ").split(',')
        production = input("Production companies (comma-separated): ").split(',')
        cast = input("Top cast (comma-separated): ").split(',')
        director = input("Director: ")

        # Clean inputs
        genres = [g.strip() for g in genres]
        keywords = [k.strip() for k in keywords]
        production = [p.strip() for p in production]
        cast = [c.strip() for c in cast]
        director = director.strip()

        # Build soup
        soup = ' '.join(genres + keywords + production + cast + [director])

        # Encode numerics
        language_encoded = le.transform([language])[0] if language in le.classes_ else 0
        num_features = [[
            budget, popularity, runtime, language_encoded,
            release_year, release_month, int(is_weekend),
            vote_avg, vote_count
        ]]
        X_num_scaled = scaler.transform(num_features)
        X_soup = cv.transform([soup]).toarray()
        X_input = np.hstack((X_num_scaled, X_soup))

        # Predict financial success
        success_pred = model.predict(X_input)[0]
        success_proba = model.predict_proba(X_input)[0]

        # Predict audience approval
        audience_pred = audience_model.predict(X_input)[0]
        audience_proba = audience_model.predict_proba(X_input)[0]

        success_result = "✅ Success" if success_pred == 1 else "❌ Flop"
        audience_result = "👍 Liked by audience" if audience_pred == 1 else "👎 Disliked by audience"

        print(f"\n🎬 {title} — {success_result}, {audience_result}")
        print(f"🔥 Financial Success Probability: {success_proba[1]*100:.2f}%")
        print(f"💀 Flop Probability: {success_proba[0]*100:.2f}%")
        print(f"⭐ Audience Approval Probability: {audience_proba[1]*100:.2f}%")
        print(f"💢 Disapproval Probability: {audience_proba[0]*100:.2f}%")

        # Optional flag
        if vote_avg < 4 and vote_count > 20000:
            print("⚠️ Warning: Extremely low audience rating despite high attention. Model may overestimate success.")

    except Exception as e:
        print(f"⚠️ Error: {e}. Please try again.")
=======
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Load models and tools
clf = joblib.load('model/success_model.pkl')
reg = joblib.load('model/budget_model.pkl')
cv = joblib.load('model/vectorizer.pkl')
scaler = joblib.load('model/scaler.pkl')
le = joblib.load('model/language_encoder.pkl')

def get_input(prompt, default=None, cast=str):
    val = input(prompt)
    if not val.strip() and default is not None:
        return default
    try:
        return cast(val)
    except:
        print(f"⚠️ Invalid input. Using default: {default}")
        return default

while True:
    print("🔍 Enter movie details (or type 'exit' to quit):")
    title = input("Movie title: ")
    if title.lower() == 'exit':
        break

    runtime = get_input("Runtime (minutes): ", default=100, cast=float)
    lang = get_input("Original language (e.g. en): ", default="en").lower()
    release_year = get_input("Release year: ", default=2025, cast=int)
    release_month = get_input("Release month (1-12): ", default=1, cast=int)
    is_weekend = get_input("Is it a weekend release? (y/n): ", default="n").lower() == 'y'

    genres = get_input("Genres (comma-separated): ", default="")
    keywords = get_input("Keywords (comma-separated): ", default="")
    prod_companies = get_input("Production companies (comma-separated): ", default="")
    cast_input = get_input("Top cast (comma-separated): ", default="")
    director = get_input("Director: ", default="")

    soup = " ".join([
        genres, keywords, prod_companies, cast_input, director
    ])

    try:
        lang_encoded = le.transform([lang])[0]
    except:
        print("⚠️ Unknown language. Defaulting to 'en'.")
        lang_encoded = le.transform(['en'])[0]

    # Numeric features
    X_num = np.array([
        runtime, lang_encoded, release_year, release_month, int(is_weekend)
    ]).reshape(1, -1)

    X_num_scaled = scaler.transform(X_num)
    soup_vec = cv.transform([soup]).toarray()

    final_input = np.hstack((X_num_scaled, soup_vec))

    # Predict
    pred = clf.predict(final_input)[0]
    prob = clf.predict_proba(final_input)[0][pred]
    budget_suggestion = reg.predict(final_input)[0]

    print(f"\n🎬 {title} Prediction: {'✅ Success' if pred else '💀 Flop'}")
    print(f"🔥 Probability: {prob * 100:.2f}%")
    print(f"💰 Suggested Budget: ${int(budget_suggestion):,}\n")

    print("-" * 50)
>>>>>>> master
