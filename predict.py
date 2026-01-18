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
