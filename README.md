# 🧠 Emotion Detective – NLP Sentiment Analysis App
A sentiment analysis web app using DistilBERT and Streamlit to classify English text as positive, negative, or neutral.


## 📌 Project Objective

To develop a context-aware, interactive **sentiment classification app** that evaluates the emotional tone of English text using a fine-tuned **DistilBERT transformer model**. Built with **Streamlit**, the goal is to create a reliable, user-friendly interface for analyzing emotional expression in everyday language—ideal for social media, feedback systems, or personal insights.

---

## 📂 Dataset Used

The model was trained and tested on a dataset of real-world **tweets**, including a diverse set of sentiments:

- 🔁 Short-form social media text  
- 😄 Emotions like joy, sadness, anger, and neutrality  
- 📊 Preprocessed and labeled for supervised learning  

📁 Dataset: [`Tweets.csv`](./Tweets.csv)

---

## ❓ Key Questions Explored

### 🔍 Sentiment Detection

* Is the given text **Positive**, **Negative**, or **Neutral**?
* What is the **confidence score** of the model's prediction?
* Can the model understand contextual words like *"fine"* or *"okay"*?

### 🛡️ Input Validation

* Is the text **genuine** and **meaningful**, or gibberish?
* Are emotional words **overused or repeated**?
* Does the input contain **prohibited patterns**, like numbers or excessive modifiers?

### 📈 Confidence-Based Feedback

* How strong is the sentiment prediction?
* What does the **gauge chart** reveal about model certainty?
* Can mixed emotions be handled using context clues?

---

## 📊 App Overview

![WhatsApp Image 2025-04-21 at 10 17 36 PM](https://github.com/user-attachments/assets/e58813fb-b9de-46fd-8a7f-d328c8d56417)


### 🧠 Functionalities

* Text validation: Prevents analysis of meaningless or invalid input
* DistilBERT inference pipeline: Tokenizes, classifies, and scores
* POS tagging integration for improved preprocessing
* Customized confidence logic based on linguistic context
* Dynamic UI with emoji and confidence meter feedback

---

## 🔧 Process

### 1. **Data Collection & Training**

* Cleaned and labeled tweet data for supervised learning
* Used HuggingFace Transformers (`DistilBERTForSequenceClassification`)
* Encoded target labels and trained for multi-class classification (3 classes)
* Saved model and label encoder for deployment

### 2. **App Development (Streamlit)**

* Created a fully responsive frontend using **Streamlit**
* Added validation for:
  * Excessive repetition
  * Modifier-only input (e.g., "very very")
  * Numeric values
  * Meaningless or offensive input
* Implemented confidence score with a **gauge chart**
* Styled UI with CSS and background images

  ![WhatsApp Image 2025-04-21 at 10 30 23 PM](https://github.com/user-attachments/assets/411a766a-fa62-4e1b-8ac2-1bc5bbc7d7d6)


### 3. **Deployment-Ready Setup**

* Prepared `requirements.txt` for easy environment setup
* Tested with various input samples to ensure robustness
* Compatible with deployment on Streamlit Cloud or Hugging Face Spaces

---

## 📌 Key Insights

* ✅ **Context-Aware Model**: Recognizes subtle language nuances (e.g., *"I'm fine"* in different moods)
* 🎯 **High Precision**: Accurately classifies distinct emotional tones
* 🧼 **Clean UI**: Encourages user-friendly analysis with validations and visual feedback
* 🛑 **Input Filtering**: Stops meaningless entries before analysis
* 🔍 **Insight-Rich Output**: Displays word count, sentiment strength, and model confidence

---

## ✅ Final Conclusion

This app merges **NLP, deep learning, and UX design** into a seamless tool for emotion classification. It doesn't just label input—it interprets it with contextual logic, validation, and engaging visuals.

A perfect fit for:
- Social media sentiment tracking  
- User feedback sentiment analysis  
- Emotion-aware chatbot systems  
- Classroom or academic NLP demonstration projects

---


