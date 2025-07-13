import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import pickle
import nltk
from nltk import pos_tag, word_tokenize
import plotly.graph_objects as go
import pandas as pd
from collections import Counter


st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')

download_nltk_data()

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1509537257950-20f875b03669?ixlib=rb-4.0.3");
            background-attachment: fixed;
            background-size: cover;
        }}
        .big-font {{
            font-size: 5.5em !important;
            background: linear-gradient(45deg, #1e3c72, #2a5298);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .content-box {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }}
        .sentiment-box {{
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }}
        .sentiment-box h2 {{
            margin: 0;
            padding: 10px;
        }}
        .metrics-container {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .metric-box {{
            background-color: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.9);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def clean_text_with_pos(text):
    """
    Clean text and add POS tags to words
    """
    try:
        # Basic cleaning
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.strip()

        # Simple POS tagging without detailed tags
        words = text.split()
        return text

    except Exception as e:
        st.error(f"Error in text cleaning: {str(e)}")
        return text

@st.cache_resource
def load_model():
    try:
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
        model.load_state_dict(torch.load('best_model_state.bin', map_location=device))
        model = model.to(device)
        model.eval()
        
        # Load the tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load the label encoder
        with open('label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)
            
        return model, tokenizer, device, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def predict_sentiment(text, model, tokenizer, device, label_encoder):
    # Clean and add POS tags
    processed_text = clean_text_with_pos(text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction using the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = float(torch.max(probabilities).cpu().item())
    
    # Get sentiment label
    sentiment = label_encoder.inverse_transform(prediction.cpu().numpy())[0]
    
    # Define word sets with context
    positive_words = {
        'happy', 'good', 'great', 'excellent', 'amazing', 'wonderful',
        'fine'  # Add 'fine' to positive words
    }
    negative_words = {
        'sad', 'bad', 'terrible', 'horrible', 'awful', 'worst'
    }
    
    # Define positive context words
    positive_context = {
        'everything', 'all', 'good', 'great', 'perfect', 'well'
    }
    
    # Check the context for "fine"
    words = text.lower().split()
    if 'fine' in words:
        # Check if "fine" is used in a positive context
        for i, word in enumerate(words):
            if word == 'fine':
                # Look at surrounding words (previous and next)
                prev_word = words[i-1] if i > 0 else ''
                next_word = words[i+1] if i < len(words)-1 else ''
                
                # If "fine" is used with positive context words, treat as positive
                if prev_word in positive_context or next_word in positive_context:
                    sentiment = 'positive'
                    confidence = max(confidence, 0.75)  # Increase confidence for positive context
                elif prev_word in {'not', 'barely', 'hardly'}:
                    sentiment = 'negative'
                    confidence = max(confidence, 0.75)
                else:
                    # Default to positive for general "fine" statements
                    sentiment = 'positive'
                    confidence = min(confidence, 0.7)  # Moderate confidence for general usage
    
    # Define neutral words that should override positive predictions with moderate confidence
    neutral_words = {
        'okay', 'fine', 'alright', 'normal', 'average', 'moderate', 
        'regular', 'standard', 'usual', 'common', 'ordinary', 'typical'
    }
    
    # Check if the text contains only neutral words without strong modifiers
    has_only_neutral = any(word in neutral_words for word in words) and not any(
        word in positive_words or word in negative_words 
        for word in words if word not in neutral_words
    )
    
    # Override prediction for neutral statements
    if has_only_neutral and confidence < 0.8:  # If confidence isn't very high
        sentiment = 'neutral'
        confidence = max(confidence, 0.65)  # Set minimum confidence for neutral
    
    # Rest of your existing mixed emotions handling code...
    has_positive = any(word in positive_words for word in words)
    has_negative = any(word in negative_words for word in words)
    
    if has_positive and has_negative:
        # Your existing contrast words handling...
        contrast_words = {'but', 'however', 'although', 'though', 'despite', 'yet'}
        contrast_positions = [i for i, word in enumerate(words) if word in contrast_words]
        
        if contrast_positions:
            # ... (rest of your contrast handling code)
            pass
    
    return sentiment, confidence

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "rgba(255, 255, 255, 0.8)"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(255, 0, 0, 0.3)"},
                {'range': [33, 66], 'color': "rgba(255, 255, 0, 0.3)"},
                {'range': [66, 100], 'color': "rgba(0, 255, 0, 0.3)"}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "white"}
    )
    
    return fig

def validate_text(text):
    """
    Validates input text for meaningful content.
    Returns (is_valid, message)
    """
    # First check for numbers in the original text
    if any(char.isdigit() for char in text):
        return False, "Numbers are not allowed. Please use only words."
    
    # Remove special characters and extra spaces
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    
    # Check for empty text
    if not text or not cleaned_text:
        return False, "Please enter some text."
    
    # Split into words
    words = cleaned_text.lower().split()
    
    # Only check maximum word limit
    word_count = len(words)
    if word_count > 20:
        return False, "Please keep your input under 20 words for better analysis."
    
    # List of valid words (including emotional words)
    valid_single_words = {
        # Positive emotions/sentiments
        'happy', 'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'perfect', 'awesome', 'superb', 'brilliant', 'outstanding', 'love', 'best',
        'joyful', 'blessed', 'delighted', 'excited', 'grateful', 'peaceful',
        
        # Negative emotions/sentiments
        'sad', 'bad', 'terrible', 'horrible', 'awful', 'worst', 'poor', 'fuck',
        'shit', 'hate', 'disgusting', 'pathetic', 'disappointed', 'angry',
        # Add death-related and strong negative words
        'died', 'death', 'dead', 'killed', 'lost', 'grieving', 'grief', 'tragic',
        'heartbroken', 'devastated', 'miserable', 'suffering', 'painful', 'hurt',
        'depressed', 'gloomy', 'mourning', 'crying', 'weeping', 'distressed',
        
        # Add neutral words
        'okay', 'fine', 'normal', 'average', 'moderate', 'regular', 'standard',
        'usual', 'common', 'ordinary', 'typical', 'routine', 'plain', 'simple',
        'fair', 'medium', 'neutral', 'balanced', 'steady', 'stable',
        
        # Add personality/character trait words
        'kind', 'nice', 'sweet', 'gentle', 'caring', 'loving',
        'smart', 'intelligent', 'wise', 'clever', 'brilliant',
        'friendly', 'helpful', 'thoughtful', 'considerate',
        'brave', 'strong', 'confident', 'honest', 'trustworthy',
        'patient', 'calm', 'peaceful', 'reliable', 'responsible',
        'creative', 'talented', 'skilled', 'capable', 'competent',
        
        # Add more positive descriptors
        'wonderful', 'amazing', 'fantastic', 'excellent', 'great',
        'outstanding', 'exceptional', 'remarkable', 'incredible',
        'marvelous', 'splendid', 'superb', 'perfect', 'ideal'
    }
    
    # Common English words - update to include more auxiliary verbs and short words
    common_words = {
        'the', 'is', 'are', 'was', 'were', 'will', 'have', 'had', 'this', 'that',
        'they', 'them', 'their', 'my', 'your', 'our', 'it', 'its', 'very', 'really',
        'am', 'been', 'being', 'do', 'does', 'did', 'has', 'can', 'could', 'would',
        'i', 'me', 'we', 'us', 'you', 'he', 'she', 'feel', 'felt', 'feeling',
        'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'down', 'into', 'over', 'under', 'about',
        # Add these auxiliary verbs and short words
        'be', 'as', 'if', 'so', 'his', 'her', 'their', 'our', 'its', 'any',
        'all', 'no', 'not', 'yes', 'may', 'might', 'must', 'shall', 'should',
        'when', 'where', 'why', 'how', 'what', 'who', 'whom', 'whose',
        # Add common nouns
        'book', 'table', 'chair', 'room', 'house', 'car', 'tree', 'sky', 'sun', 
        'moon', 'water', 'food', 'door', 'window', 'computer', 'phone', 'paper',
        'desk', 'wall', 'floor', 'street', 'road', 'park', 'school', 'office',
        'building', 'garden', 'flower', 'grass', 'bird', 'cat', 'dog',
        
        # Add descriptive words
        'big', 'small', 'tall', 'short', 'long', 'wide', 'narrow', 'high', 'low',
        'round', 'square', 'flat', 'deep', 'shallow', 'thick', 'thin', 'heavy',
        'light', 'hot', 'cold', 'warm', 'cool', 'bright', 'dark', 'loud', 'quiet',
        'fast', 'slow', 'hard', 'soft', 'smooth', 'rough', 'clean', 'dirty',
        'empty', 'full', 'open', 'closed', 'beautiful', 'new', 'old', 'young',
        
        # Add action words
        'sit', 'sits', 'walk', 'walks', 'run', 'runs', 'jump', 'jumps', 'sleep',
        'sleeps', 'eat', 'eats', 'drink', 'drinks', 'write', 'writes', 'read',
        'reads', 'speak', 'speaks', 'listen', 'listens', 'watch', 'watches',
        'work', 'works', 'play', 'plays', 'move', 'moves', 'stand', 'stands',
        
        # Add location words
        'here', 'there', 'inside', 'outside', 'above', 'below', 'behind',
        'between', 'among', 'around', 'through', 'across', 'along', 'beside',
        'near', 'far', 'left', 'right', 'top', 'bottom', 'middle', 'center',
        
        # Add person-related words
        'person', 'people', 'human', 'man', 'woman', 'boy', 'girl',
        'child', 'children', 'adult', 'friend', 'family', 'student',
        'teacher', 'worker', 'parent', 'mother', 'father',
    }
    
    # List of modifier words that shouldn't be used alone or repetitively
    modifier_words = {
        # Intensifiers
        'very', 'really', 'so', 'extremely', 'incredibly', 'totally', 'absolutely',
        'utterly', 'deeply', 'highly', 'super', 'quite', 'remarkably', 'insanely',
        'awfully', 'crazily', 'severely', 'fantastically',
        
        # Downtoners/Softeners
        'slightly', 'somewhat', 'kind', 'bit', 'little', 'fairly', 'mildly',
        'moderately', 'hardly', 'scarcely', 'barely',
        
        # Qualifiers/Hedge words
        'maybe', 'perhaps', 'possibly', 'likely', 'probably', 'apparently',
        'seemingly',
        
        # Function/Filler words
        'just', 'even', 'only', 'still', 'yet', 'already', 'again', 'ever',
        'always', 'never', 'often', 'rarely', 'almost', 'mostly',
        
        # Some Fillers (removed positive exclamations)
        'ugh', 'meh', 'ah', 'uh', 'um', 'hmm', 'huh', 'geez', 'whoops',
        
        # Vague Qualifiers
        'something', 'anything', 'everything', 'nothing', 'stuff', 'things',
        'sort', 'type', 'bunch', 'ton',
        
        # Discourse Markers
        'like', 'actually', 'basically', 'literally', 'technically', 'honestly',
        'frankly', 'seriously', 'anyway', 'whatever', 'clearly', 'obviously'
    }
    
    # Check for gibberish/meaningless words
    meaningless_words = []
    valid_words = []
    
    for word in words:
        # First check each word for gibberish or misspellings
        is_gibberish = (
            len(set(word)) < len(word) / 2 or  # repeated characters
            not word.isalpha() or  # non-alphabetic characters
            any(c * 3 in word for c in word) or  # character repeated 3+ times
            word not in valid_single_words and 
            word not in common_words and 
            word not in modifier_words  # word must be in our dictionaries
        )
        
        if is_gibberish:
            return False, f"Invalid word '{word}' detected. Please check your spelling and use proper English words."
    
    # If we found any meaningless words, reject the input
    if meaningless_words:
        return False, f"Found meaningless words: {', '.join(meaningless_words)}"
    
    # Check for word repetition - only for sentiment and modifier words
    word_freq = Counter(words)
    sentiment_and_modifier_words = valid_single_words | modifier_words
    excessive_repetition = any(
        count > 2 and word in sentiment_and_modifier_words 
        for word, count in word_freq.items()
    )
    
    if excessive_repetition:
        return False, "Please avoid excessive repetition of emotional or modifier words."
    
    # In validate_text function, add this new set:
    valid_exclamations = {
        'oh wow', 'wow', 'oh my god', 'omg', 'oh yes', 'yay', 'hurray', 'woohoo',
        'awesome', 'oh nice', 'oh great', 'oh amazing'
    }

    # Modify the validation logic to check for valid exclamations first:
    # Add this check after splitting words into 'words'
    
    # Check for valid exclamations
    full_text = cleaned_text.lower()
    if full_text in valid_exclamations:
        return True, "Valid input"
    
    # Check if the text contains ONLY modifier words
    if all(word in modifier_words for word in words):
        return False, "Please include meaningful content beyond modifier words."
    
    # Count meaningful words (excluding modifiers and articles)
    skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were'} | modifier_words
    meaningful_word_count = len([word for word in words if word not in skip_words])
    
    # Special case: Allow compliments and descriptions of people
    has_descriptor = any(word in valid_single_words for word in words)
    has_person_reference = any(word in {'you', 'person', 'people', 'he', 'she', 'they', 'i', 'we'} for word in words)
    has_linking_verb = any(word in {'are', 'is', 'am', 'was', 'were'} for word in words)
    
    # If it's a compliment or description of someone, it's valid
    if has_descriptor and (has_person_reference or has_linking_verb):
        return True, "Valid input"
    
    # Check for excessive modifiers (but allow up to 2 modifiers before an emotional word)
    modifier_count = sum(1 for word in words if word in modifier_words)
    remaining_words = len([word for word in words if word not in modifier_words])
    
    # Allow common time expressions
    time_words = {'now', 'right', 'today', 'tonight', 'currently'}
    has_time = any(word in time_words for word in words)
    
    if has_descriptor and has_time:
        return True, "Valid input"
    
    # Check for excessive modifiers (but allow up to 2 modifiers before an emotional word)
    modifier_count = sum(1 for word in words if word in modifier_words)
    remaining_words = len([word for word in words if word not in modifier_words])
    if remaining_words == 0 or (modifier_count > 2 and remaining_words == 1):
        return False, "Too many modifier words. Please include more meaningful content."
    
    # Check if all words are valid
    all_words_valid = all(word in common_words or word in valid_single_words or word in modifier_words for word in words)
    
    if all_words_valid:
        return True, "Valid input"
    
    return False, "Please include some meaningful content in your text."

def test_validation():
    """
    Test function for text validation
    """
    test_cases = [
        # Valid cases
        ("The service was very very good", True),
        ("This flight was really really amazing", True),
        ("I had a great experience today", True),
        
        # Invalid cases
        ("very very", False),
        ("blah blah", False),
        ("um um um", False),
        ("", False),
        ("hi", False),
        ("very very very", False),
        ("The the the the", False),
        ("happy", True),
        ("sad", True),
        ("happy sad", True),
        ("happy happy happy", False),
        ("happy sad sad happy", False),
        ("I am happy", True),
        ("This is sad", True)
    ]
    
    print("Running validation tests...")
    for text, expected in test_cases:
        is_valid, message = validate_text(text)
        status = "✓" if is_valid == expected else "✗"
        print(f"{status} Input: '{text}' -> Valid: {is_valid}, Message: {message}")

def display_results(sentiment, confidence, text):
    """Display the sentiment analysis results"""
    
    # Ensure confidence is a float at the start of display
    if torch.is_tensor(confidence):
        confidence = float(confidence.cpu().item())
    
    # Display the input text in the box with the heading
    st.markdown(f"""
        <div style='background-color: rgba(255, 255, 255, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;'>
            <h3 style='color: #a8c7ff; margin-bottom: 10px;'>Your Input:</h3>
            <p style='color: white; font-size: 16px; margin-left: 20px;'>{text}</p>
        </div>
    """, unsafe_allow_html=True)

    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Display sentiment with appropriate color and emoji
    sentiment_colors = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#ffc107'
    }
    
    emojis = {
                        'positive': '😊',
                        'negative': '😔',
                        'neutral': '😐'
                    }
    
    # Display in first column
    with col1:
        st.markdown(f"""
        <div style='background-color: {sentiment_colors[sentiment.lower()]};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;'>
            <h2>Sentiment: {sentiment} {emojis[sentiment.lower()]}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Display confidence in second column
    with col2:
        st.markdown(f"""
        <div style='background-color: rgba(255, 255, 255, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;'>
            <h2>Confidence: {confidence:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Create gauge chart
    st.plotly_chart(
        create_gauge_chart(confidence, "Confidence Score"),
        use_container_width=True
    )
    
    # Display additional metrics
    st.markdown("### Analysis Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        - **Word Count**: {len(text.split())}
        - **Sentiment Strength**: {'Strong' if confidence > 0.8 else 'Moderate' if confidence > 0.6 else 'Weak'}
        """)
    
    with col2:
        st.markdown(f"""
        - **Confidence Score**: {confidence:.2%}
        - **Analysis Status**: {'High Confidence' if confidence > 0.8 else 'Medium Confidence' if confidence > 0.6 else 'Low Confidence'}
        """)

def main():
    try:
        # Load model and label encoder
        model, tokenizer, device, label_encoder = load_model()
        
        # Create main container
        with st.container():
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.title("✨ Emotion Detective 🔍")
                st.markdown("### Let's Decode Your Feelings! 💭✨")
        
        # Help message for users with emojis
        st.info("""
        ✨ Welcome to Emotion Detective! ✨

        ✍️ How to Use:
        • Type any text, phrase, or word 📝
        • Keep your input under 20 words ✨
        • Click analyze and watch the magic happen ✨
        • Get instant emotional insights! 🎯

        🎯 Input Guidelines:
        • Maximum 20 words for concise analysis
        • Use proper English words and phrases
        • Single emotional words are allowed
        • Avoid gibberish or meaningless words
        • Avoid excessive word repetition
        • Numbers are not allowed

        ❌ Note:
        • No mixing gibberish with emotions
        • No meaningless repetition
        • No nonsense words
        • No numbers
        """)
        
        # Text input
        text = st.text_area(
            "Enter your text:",
            placeholder="Enter meaningful text (e.g., 'The service was very very good' is valid, but 'very very' alone is not)",
            height=100
        )
        
        # Center the analyze button
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            analyze_button = st.button("Analyze", use_container_width=True)
        
        if analyze_button and text:
            # Validate input
            is_valid, message = validate_text(text)
            
            if not is_valid:
                st.error(message)
                return
            
            # Only proceed if validation passed
            with st.spinner("Analyzing sentiment..."):
                # Get prediction
                sentiment, confidence = predict_sentiment(text, model, tokenizer, device, label_encoder)
                
                # Display results
                display_results(sentiment, confidence, text)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
    # Uncomment the following line to run validation tests
    # test_validation()