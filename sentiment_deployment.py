# pip install fastapi uvicorn
# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from string import punctuation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re 
from os.path import dirname, join, realpath
import uvicorn
from fastapi import FastAPI
# Uncomment once
# nltk.download('stopwords')
# nltk.download('wordnet')

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)
model = keras.models.load_model('/mnt/D/Shai/sentiment competition/bilstm_1.h5')

def text_cleaning(text, remove_stop_words=False, lemmatize_words=True):
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers

    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:

        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return text

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(review)


    # Load the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cleaned_review)

    # Convert the cleaned review into a sequence of word indices
    sequences = tokenizer.texts_to_sequences([cleaned_review])
    padded_sequences = pad_sequences(sequences, maxlen=300)

    # Perform sentiment prediction
    prediction = model.predict(padded_sequences)
    print(prediction)
    output = int(prediction[0])
    print(output)
    # probas = model.predict_proba(padded_sequences)
    # output_probability = "{:.2f}".format(float(probas[:, output]))

    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}

    # show results
    result = {"prediction": sentiments[output], "Probability": output}

    return result
