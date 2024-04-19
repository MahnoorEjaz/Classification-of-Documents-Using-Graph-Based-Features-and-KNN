# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:08:15 2024

@author: ELITEBOOK
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Removing Punctuation
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    
    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    
    # Removing Numbers
    stemmed_tokens = [token for token in stemmed_tokens if not token.isdigit()]
    
    # Handling Special Characters
    # (Depending on the context, you may add additional handling for special characters)
    stemmed_tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in stemmed_tokens]
    
    # Handling White Spaces
    stemmed_tokens = [token.strip() for token in stemmed_tokens]
    

    
     
    
     # Remove duplicates
    unique_stemmed_tokens = list(set(stemmed_tokens))
    
    return unique_stemmed_tokens

# Read text from file
file_path = 'C:/sem 6/text files/Diseases4.txt'
with open(file_path, 'r') as file:
    text = file.read()

# Preprocess text
preprocessed_tokens = preprocess_text(text)

# Write preprocessed tokens to a new text file
output_file_path = 'C:/sem 6/text files/disease15Output.txt'
with open(output_file_path, 'w') as file:
    for token in preprocessed_tokens:
        file.write(token + '\n')

print("Preprocessed Tokens saved to", output_file_path)
