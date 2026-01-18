import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
movies = pd.read_csv(r'C:\Users\ADMIN\Desktop\Movie-ml\data\movies.csv')
credits = pd.read_csv(r'C:\Users\ADMIN\Desktop\Movie-ml\data\credits.csv')
df = movies.merge(credits, left_on='id', right_on='movie_id')

# Drop unnecessary/post-release columns
df.drop(columns=[
    'homepage', 'status', 'original_title', 'overview', 'spoken_languages',
    'tagline', 'title_y', 'movie_id', 'production_countries',
    'video', 'adult', 'vote_average', 'vote_count', 'popularity'
], inplace=True, errors='ignore')

# JSON parsing
def extract_names(x):
    try:
        return [d['name'] for d in json.loads(x)]
    except:
        return []

def extract_director(x):
    try:
        for d in json.loads(x):
            if d['job'] == 'Director':
                return d['name']
    except:
        return ''

df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['production_companies'] = df['production_companies'].apply(extract_names)
df['cast'] = df['cast'].apply(lambda x: extract_names(x)[:3])
df['director'] = df['crew'].apply(extract_director)

# Create 'soup'
df['soup'] = (
    df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
    df['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
    df['production_companies'].apply(lambda x: ' '.join(x)) + ' ' +
    df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
    df['director'].fillna('')
)

# Label encode language
le = LabelEncoder()
df['original_language'] = le.fit_transform(df['original_language'])

# Date handling
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['is_weekend_release'] = (df['release_date'].dt.dayofweek >= 5).astype(int)

# Runtime cleanup
df['runtime'] = df['runtime'].fillna(df['runtime'].median())

# Filter invalid budget/revenue
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# Define success label
df['success'] = (df['revenue'] > 1.5 * df['budget']).astype(int)

# Vectorize soup
cv = CountVectorizer(max_features=5000, stop_words='english')
soup_matrix = cv.fit_transform(df['soup']).toarray()

# Pre-release numerical features
num_cols = [
    'runtime', 'original_language',
    'release_year', 'release_month', 'is_weekend_release'
]
X_num = df[num_cols].fillna(0)

# Scale
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Final input
X = np.hstack((X_num_scaled, soup_matrix))

# 🎯 Success classifier
y_success = df['success']

# 💸 Budget regressor target
y_budget = df['budget']

# Split
X_train, X_test, y_success_train, y_success_test, y_budget_train, y_budget_test = train_test_split(
    X, y_success, y_budget, test_size=0.2, random_state=42
)

# Train classifier
success_model = RandomForestClassifier(n_estimators=100, random_state=42)
success_model.fit(X_train, y_success_train)

# Train regressor
budget_model = RandomForestRegressor(n_estimators=100, random_state=42)
budget_model.fit(X_train, y_budget_train)

# Save models
joblib.dump(success_model, 'model/success_model.pkl')
joblib.dump(budget_model, 'model/budget_model.pkl')
joblib.dump(cv, 'model/vectorizer.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/language_encoder.pkl')

# Evaluate success model
y_pred_success = success_model.predict(X_test)
print("\n🎯 Success Model Accuracy:", accuracy_score(y_success_test, y_pred_success))
print("\n📋 Success Model Report:\n", classification_report(y_success_test, y_pred_success))
