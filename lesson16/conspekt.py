import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

data = pd.read_csv("./lesson20/spam.csv")
print(data)
print(data.columns)
data.info()

data["Spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0)

print(data.head())
vect = CountVectorizer()
X = vect.fit_transform(data["Message"])

w = vect.get_feature_names_out()


model = Pipeline([("vect", CountVectorizer()), ("nb", MultinomialNB())])

X_train, X_test, y_train, y_test = train_test_split(
    data["Message"], data["Spam"], test_size=0.3
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(accuracy_score(y_test, y_pred))

message = [
    "Free entry to a draw to win a £1000 cash prize",
    "You are awarded a £9000 prize",
    "Please call our customer service representative ongay 0800 123 4567",
    "You have won a £1000 cash prize",
    "Please call our customer service representative on 0800 123 4567",
]

print(model.predict(message))


