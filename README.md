# P4-Implementation-of-chatbot-by-NLP
# Google Play Feedback Analyzer Chatbot

This project is a chatbot that analyzes Google Play Store app reviews using machine learning. It scrapes reviews from the Google Play Store, processes them, and uses a Logistic Regression model to classify reviews as either "positive" or "negative". The chatbot then predicts the sentiment of user input based on the trained model.

## Features

- **Scrape Reviews**: Scrape Google Play reviews for any app by providing its app ID.
- **Sentiment Analysis**: Classify reviews as "positive" or "negative" based on the app's rating.
- **Chatbot Interface**: Predicts the sentiment of user-entered feedback using a trained model.
- **View Dataset**: Option to display a preview of the scraped reviews dataset.

## Requirements

To run the project, you need to install the following libraries:

- `streamlit`: For building the web application.
- `pandas`: For handling the dataset.
- `nltk`: For natural language processing (e.g., tokenization).
- `google_play_scraper`: For scraping reviews from the Google Play Store.
- `sklearn`: For machine learning models and text vectorization.

You can install the necessary libraries using `pip`:

```bash
pip install streamlit pandas nltk google-play-scraper scikit-learn
