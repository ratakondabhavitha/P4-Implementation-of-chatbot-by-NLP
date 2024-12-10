import streamlit as st
import pandas as pd
import nltk
import ssl
from google_play_scraper import reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# For NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Scrape Reviews from Google Play
@st.cache
def scrape_reviews(app_id, num_reviews=100):
    """
    Scrape Google Play reviews for a specific app.
    """
    result, _ = reviews(app_id, lang='en', country='us', count=num_reviews)
    df = pd.DataFrame(result)
    return df[['content', 'score']]

# Preprocess Reviews
def preprocess_data(df):
    """
    Preprocess review text and categorize scores.
    """
    df['label'] = df['score'].apply(lambda x: 'positive' if x >= 4 else 'negative')
    return df

# Train the Chatbot Model
def train_model(reviews, labels):
    """
    Train a Logistic Regression model for review classification.
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(reviews)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(X_train, y_train)
    return model, vectorizer

# Chatbot Response
def chatbot_response(model, vectorizer, user_input):
    """
    Predict the sentiment of user input.
    """
    input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(input_vectorized)[0]
    return f"The review seems to be {prediction}."

# Streamlit Application
def main():
    st.title("Google Play Feedback Analyzer Chatbot")
    
    # Sidebar for app settings
    st.sidebar.header("Settings")
    app_id = st.sidebar.text_input("Enter Google Play App ID", value="com.whatsapp")
    num_reviews = st.sidebar.slider("Number of Reviews to Scrape", 10, 500, 100)
    
    # Scrape reviews
    st.write("Scraping reviews... This might take a moment.")
    reviews_df = scrape_reviews(app_id, num_reviews)
    reviews_df = preprocess_data(reviews_df)
    
    # Train model
    st.write("Training model...")
    model, vectorizer = train_model(reviews_df['content'], reviews_df['label'])
    st.success("Model trained successfully!")
    
    # Chatbot Interface
    st.subheader("Chatbot Interface")
    user_input = st.text_input("Enter a review or feedback:")
    if user_input:
        response = chatbot_response(model, vectorizer, user_input)
        st.write(f"Chatbot: {response}")
    
    # View Dataset
    if st.checkbox("Show Dataset"):
        st.write(reviews_df.head())

if __name__ == "__main__":
    main()

