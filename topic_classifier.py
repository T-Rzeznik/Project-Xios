import streamlit as st
from transformers import pipeline

# Initialize the zero-shot classification pipeline
def initialize_classifier():
    if 'topic_classifier' not in st.session_state:
        st.session_state.topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


#ZERO SHOT IMPLEMENTATION
def is_on_topic(text, context, lab_response):
    initialize_classifier()
    
    # Get keywords from session state and debug print
    keywords = st.session_state.get('generated_keywords', [])
    
    # Base candidate labels
    candidate_labels = [
        f"related to {context}",
        f"about {context}",
        f"discussing {context}",
        "linux related question",
        "cyber security question",
        "computer science related question",
        "linux system administration",
        "off topic",
        "unrelated conversation",
        "non-technical discussion"
    ]
    

    if keywords:  # Only try to add if keywords exist
        for keyword in keywords:
            new_labels = [
                f"about {keyword}",
                f"related to {keyword}",
                f"contains {keyword}"
            ]
            candidate_labels.extend(new_labels)
    else:
        st.write("No keywords found in session state!")  # Debug print
    
    
    
    # Classify the text
    result = st.session_state.topic_classifier(text, candidate_labels)
    
    # Debugging: Output the classification result
    # Debugging: Output only top 10 labels and scores
    top_10_results = list(zip(result['labels'][:10], result['scores'][:10]))
    st.write("Top 10 Classification Results:")
    for label, score in top_10_results:
        st.write(f"- {label}: {score:.3f}")
    
    # Get the highest scoring label
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    
    # Debugging: Output the top label and score
    st.write("Top score:", top_score)
    
    is_related = ("related" in top_label.lower() or  
                 "about" in top_label.lower() or
                 "linux related question" in top_label.lower() or
                 "computer science related question" in top_label.lower() or
                 "linux command question" in top_label.lower() or
                 "cyber security question" in top_label.lower() or
                 "linux system administration" in top_label.lower())
    
    return is_related and top_score > 0.15