import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# =============================
# CARGA DE DATOS
# =============================
df = pd.read_excel("denuncias.xlsx", sheet_name=0)
df.head(10)
print('hello')
X = df["texto"]
y = df["delito"]

# =============================
# DIVISIÓN ENTRENAMIENTO
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =============================
# PIPELINE NLP + MODELO
# =============================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
    lowercase=True,
    ngram_range=(1,2)
)),
    ("clf", MultinomialNB())
])

# =============================
# ENTRENAMIENTO
# =============================
pipeline.fit(X_train, y_train)

# =============================
# EVALUACIÓN
# =============================
y_pred = pipeline.predict(X_test)

print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =============================
# GUARDAR MODELO
# =============================
joblib.dump(pipeline, "modelo_denuncias.pkl")
print("✅ Modelo guardado como modelo_denuncias.pkl")