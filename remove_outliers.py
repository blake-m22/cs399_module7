
"""
Blake Moore
CS399 HW 7
I decieded to use spaCy because it provides pre-trained word vectors that are trained on a large collection of text data
Another big reason that I decided to use this model is because of its integration with NLP pipelines. 

Overall, spaCy provides a convenient way to work with text data, espically for tasks like semantic similarity calculation, 
which was used for identifying outliers based on their  similarity to other words provided by the user and stored in an input list.


"""

import spacy
import streamlit as st

# Load medium-sized English model with word vectors
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")
print("Loaded.")

# Set a lower similarity threshold
SIMILARITY_THRESHOLD = 0.2

def is_outlier(word, words):
    """
    Check if a word is an outlier based on its semantic similarity with other words in the list.
    """
    word_doc = nlp(word)
    if not word_doc.has_vector:
        return True
    similarities = [word_doc.similarity(nlp(other)) for other in words if word != other and nlp(other).has_vector]
    avg_similarity = sum(similarities) / max(1, len(similarities))
    return avg_similarity < SIMILARITY_THRESHOLD
def main():
    st.title("Word Outlier Remover")
    
    # Multiline text input for entering the list of words
    st.write("Enter each word on a new line:")
    words_input = st.text_area("Input words here:", "")

    if words_input:
        words = [word.strip() for word in words_input.split("\n") if word.strip()]  # Split and strip words, filtering out empty strings

        if len(words) < 3:
            st.warning("Please provide at least 3 words.")
        else:
            # Remove outliers
            filtered_words = [word.strip() for word in words if not is_outlier(word.strip(), words)]

            if len(filtered_words) < 3:
                st.error("Unable to remove outliers. Please provide more valid words.")
            else:
                st.success("With outliers removed, your list looks like this:")
                st.write(", ".join(filtered_words))

if __name__ == "__main__":
    main()
