import nltk
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Define the chatbot function
def chatbot(query):
    response = get_most_relevant_sentence(query, processed_sentences)
    return response

# Preprocess the text data
def preprocess(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = ''.join([char for char in sentence if char not in string.punctuation])
        cleaned_sentences.append(sentence)
    
    return cleaned_sentences


# Define the function to calculate similarity
def get_most_relevant_sentence(query, sentences):
    # Combine the query with the sentences from the text
    sentences.insert(0, query)  # Insert the query as the first sentence
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])  # Compare with all sentences
    
    # Get the index of the most relevant sentence
    most_relevant_idx = cosine_sim.argmax()
    return sentences[most_relevant_idx + 1]  # +1 because the query is at index 0

# Streamlit interface
def main():
    st.title("Shakespeare Chatbot")
    user_input = st.text_input("Ask me anything about Shakespeare:")
    
    if user_input:
        response = chatbot(user_input)
        st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    main()