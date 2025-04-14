from django.shortcuts import render
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from .forms import MessageForm
import os

# Get the base directory of your project (the folder containing manage.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build the path to the CSV file relative to your base directory
csv_path = os.path.join(BASE_DIR, 'detector', 'emails.csv')

# Load the dataset
dataset = pd.read_csv(csv_path)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text'])

X_train, X_test, Y_train, Y_test = train_test_split(X, dataset['spam'], test_size=0.2)

model = MultinomialNB()
model.fit(X_train, Y_train)

def predictMessage(message):
    messageVector = vectorizer.transform([message])
    prediction = model.predict(messageVector)
    return 'Spam' if prediction[0] == 1 else 'Ham'

def Home(request):
    result = None
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']
            result = predictMessage(message)
    else:
        form = MessageForm()
    return render(request, 'home.html', {'form': form, 'result': result})
