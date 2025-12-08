import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# =============================
# CARGA DE DATOS
# =============================
df = pd.read_excel("denuncias.xlsx", sheet_name=0)
df.head(10)

X = df["texto"]
y = df["delito"]

# =============================
# DIVISIÓN ENTRENAMIENTO
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y 
) # stratify es para que no se pierdan las clases de 'y'

# =============================
# PIPELINE NLP + MODELO
# =============================
pipeline = Pipeline([ # Se le indica un camino a seguir
    ("tfidf", TfidfVectorizer( # el nombre del camino es tfidf y 
    lowercase=True, # realiza la operación de vectorizar
    ngram_range=(1,2)
)),
    ("clf", MultinomialNB()) # el nombre del camino es clf y aplica
]) # la distribución multinomial

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


cm = confusion_matrix(y_test, y_pred)
labels = pipeline.classes_  # nombres reales de tus clases

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title("Matriz de Confusión - Clasificación de Denuncias")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =============================
# GUARDAR MODELO
# =============================
joblib.dump(pipeline, "modelo_denuncias.pkl")
print("✅ Modelo guardado como modelo_denuncias.pkl")