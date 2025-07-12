import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load both CSV files
movies = pd.read_csv(r'C:\Users\ADMIN\Desktop\Movie-ml\data\movies.csv')
cr = pd.read_csv(r'C:\Users\ADMIN\Desktop\Movie-ml\data\credits.csv')

# Merge on the common ID
df = movies.merge(cr, left_on='id', right_on='movie_id')

# Drop unnecessary columns
df.drop(columns=[
    'homepage', 'status', 'original_title', 'overview',
    'spoken_languages', 'tagline', 'title_y', 'movie_id', 'production_countries', 'video', 'adult'
], inplace=True, errors='ignore')

def extract_names(x):
    try:
        return [d['name'] for d in json.loads(x)]
    except:
        return []

def extract_director(x):
    try:
        crew_list = json.loads(x)
        for d in crew_list:
            if d['job'] == 'Director':
                return d['name']
    except:
        return ''

df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['production_companies'] = df['production_companies'].apply(extract_names)
df['cast'] = df['cast'].apply(lambda x: extract_names(x)[:3])  # Top 3 actors
df['director'] = df['crew'].apply(extract_director)

df['soup'] = (
    df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
    df['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
    df['production_companies'].apply(lambda x: ' '.join(x)) + ' ' +
    df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
    df['director'].fillna('')
)

le = LabelEncoder()
df['original_language'] = le.fit_transform(df['original_language'])
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Year and weekday
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['is_weekend_release'] = df['release_date'].dt.dayofweek >= 5  # Sat/Sun = 5/6
df['is_weekend_release'] = df['is_weekend_release'].astype(int)

# Filter zero budgets or revenue
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# Define targets
df['success'] = (df['revenue'] > 1.5 * df['budget']).astype(int)
df['audience_liked'] = (df['vote_average'] >= 6).astype(int)

cv = CountVectorizer(max_features=5000, stop_words='english')
soup_matrix = cv.fit_transform(df['soup']).toarray()

# Numerical features
num_cols = ['budget', 'popularity', 'runtime', 'original_language', 'release_year', 'release_month', 'is_weekend_release', 'vote_average', 'vote_count']
X_num = df[num_cols].fillna(0)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Final X and y
X = np.hstack((X_num_scaled, soup_matrix))
y_success = df['success']
y_audience = df['audience_liked']

# Split 80% train, 20% test
X_train, X_test, y_success_train, y_success_test = train_test_split(
    X, y_success, test_size=0.2, random_state=42, stratify=y_success
)
_, _, y_audience_train, y_audience_test = train_test_split(
    X, y_audience, test_size=0.2, random_state=42, stratify=y_audience
)

# Train models
clf_success = RandomForestClassifier(n_estimators=100, random_state=42)
clf_success.fit(X_train, y_success_train)

audience_clf = RandomForestClassifier(n_estimators=100, random_state=42)
audience_clf.fit(X_train, y_audience_train)

# Save models and tools
joblib.dump(clf_success, 'model/movie_success_model.pkl')
joblib.dump(audience_clf, 'model/audience_model.pkl')
joblib.dump(cv, 'model/vectorizer.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/language_encoder.pkl')

# Evaluate
from sklearn.metrics import classification_report
print("\n🎯 Success Model Accuracy:", accuracy_score(y_success_test, clf_success.predict(X_test)))
print("\n📋 Success Model Report:\n", classification_report(y_success_test, clf_success.predict(X_test)))
print("\n🎯 Audience Model Accuracy:", accuracy_score(y_audience_test, audience_clf.predict(X_test)))
print("\n📋 Audience Model Report:\n", classification_report(y_audience_test, audience_clf.predict(X_test)))
