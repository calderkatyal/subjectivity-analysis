import nltk
import random
import pickle
from nltk.corpus import subjectivity
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# Downloading necessary NLTK datasets
nltk.download('subjectivity')
nltk.download('punkt')

# Function to extract features from the dataset
def extract_features(words):
    return dict([(word.lower(), True) for word in words])

# Load the subjectivity sentences from nltk
subjectivity_data = [(sentence, 'subjective' if label == 'subj' else 'objective')
                     for label in subjectivity.categories()
                     for sentence in subjectivity.sents(categories=label)]

# Shuffle the data
random.shuffle(subjectivity_data)

# Extract features from the data
feature_sets = [(extract_features(sen), category) for (sen, category) in subjectivity_data]

# Split the data into training and testing sets (70% training, 30% testing)
train_size = int(len(feature_sets) * 0.7)
train_set, test_set = feature_sets[:train_size], feature_sets[train_size:]

# Function to save the trained model
def save_trained_model(classifier):
    with open('subjectivity_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

# Function to load the trained model
def load_trained_model():
    try:
        with open('subjectivity_classifier.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load or train the Naive Bayes classifier
classifier = load_trained_model()
if not classifier:
    classifier = NaiveBayesClassifier.train(train_set)
    save_trained_model(classifier)

# Evaluate the classifier
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy:.2f}")

# Function to classify user input and learn from corrections
def classify_user_input():
    while True:
        sentence = input("Enter a sentence to analyze ('quit' to stop): ")
        if sentence.lower() == 'quit':
            break
        words = nltk.word_tokenize(sentence.lower())
        features = extract_features(words)
        sentiment = classifier.classify(features)
        print(f"Sentence: '{sentence}', Category: {sentiment}")

        correction = input("Is this correct? (yes/no/quit): ")
        if correction.lower() == 'quit':
            break
        elif correction.lower() == 'no':
            correct_category = input("What is the correct category? (subjective/objective): ")
            new_data = (features, correct_category)
            retrain_classifier_with_new_data(new_data)
            save_trained_model(classifier)

# Function to retrain the classifier with new data
def retrain_classifier_with_new_data(new_data):
    global classifier, train_set
    train_set += [new_data]
    classifier = NaiveBayesClassifier.train(train_set)

# Classify user input
classify_user_input()
