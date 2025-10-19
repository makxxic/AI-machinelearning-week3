# ============================================================
# Task 3: NLP with spaCy
# Text Data: User reviews from Amazon Product Reviews.
# Goal: Perform named entity recognition (NER) to extract product names and brands and analyze sentiment (positive/negative) using a rule-based approach.
# ============================================================
# Install spaCy if not already installed
# pip install spacy

# Load spaCy English model (small version is sufficient)
# python -m spacy download en_core_web_sm

import spacy
from spacy import displacy

# Load English NLP pipeline
nlp = spacy.load("en_core_web_sm")

# Sample user reviews
reviews = [
    "I absolutely love my new Samsung Galaxy phone! The camera quality is outstanding.",
    "The Apple AirPods stopped working after a week. Totally disappointed!",
    "Bought a Lenovo laptop last month and it performs really well for the price.",
    "The Sony headphones have great sound but the battery life is too short.",
    "This HP printer is a waste of money — keeps jamming every time I try to print."
]

# ------------------------------------------------------------
# 1️⃣ Named Entity Recognition (NER)
# ------------------------------------------------------------
print("Named Entity Recognition Results:\n") # Print a header for the NER results

# Process each review using the spaCy NLP pipeline
for review in reviews:
    doc = nlp(review) # Process the review to create a Doc object
    # Extract entities and their labels from the Doc object
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"Review: {review}") # Print the original review
    print("Entities:", entities) # Print the extracted entities and their labels
    print("-" * 80) # Print a separator line

# ------------------------------------------------------------
# 2️⃣ Rule-Based Sentiment Analysis
# ------------------------------------------------------------
# Define small lexicons (lists of words) for rule-based sentiment analysis
positive_words = ["love", "great", "amazing", "good", "outstanding", "excellent", "well", "happy"] # List of positive words
negative_words = ["bad", "poor", "terrible", "disappointed", "waste", "broke", "short", "jam", "stopped"] # List of negative words

# Function to analyze sentiment based on word counts
def analyze_sentiment(text):
    text_lower = text.lower() # Convert the input text to lowercase for case-insensitive matching
    pos_count = sum(word in text_lower for word in positive_words) # Count the number of positive words in the text
    neg_count = sum(word in text_lower for word in negative_words) # Count the number of negative words in the text

    # Determine sentiment based on word counts
    if pos_count > neg_count:
        return "Positive" # Return "Positive" if there are more positive words
    elif neg_count > pos_count:
        return "Negative" # Return "Negative" if there are more negative words
    else:
        return "Neutral" # Return "Neutral" if the counts are equal or zero

# Apply sentiment analysis to each review
print("\nSentiment Analysis Results:\n") # Print a header for the sentiment analysis results

for review in reviews:
    sentiment = analyze_sentiment(review) # Analyze the sentiment of the current review
    print(f"Review: {review}") # Print the original review
    print(f"Sentiment: {sentiment}") # Print the determined sentiment
    print("-" * 80) # Print a separator line

# ------------------------------------------------------------
# 3️⃣ Combine Results and Display as DataFrame
# ------------------------------------------------------------
results = [] # Initialize an empty list to store combined results

# Process each review to extract entities and sentiment
for review in reviews:
    doc = nlp(review) # Process the review using the spaCy NLP pipeline
    # Extract entities with "ORG" or "PRODUCT" labels
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    sentiment = analyze_sentiment(review) # Analyze the sentiment of the review
    # Append results as a dictionary to the list
    results.append({"Review": review, "Entities": entities, "Sentiment": sentiment})

import pandas as pd # Import pandas for creating and displaying a DataFrame
df = pd.DataFrame(results) # Create a pandas DataFrame from the results list
# The DataFrame will be displayed automatically in Colab
df

# ------------------------------------------------------------
# 4️⃣ Visualize Named Entities (using displacy)
# ------------------------------------------------------------
# Display named entities for one example review using displacy
doc = nlp(reviews[0]) # Process the first review
displacy.render(doc, style='ent', jupyter=True) # Render the entities with the 'ent' style in a Jupyter environment