# 💬 Sentiment Analysis & Rating Coherence App

A Streamlit web application that analyzes the sentiment of textual feedback and verifies its coherence with the provided numerical rating.

## 📋 Project Overview

This application uses **hybrid sentiment analysis** combining:
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)** - Rule-based sentiment analysis optimized for social media and short texts
- **Machine Learning (TF-IDF + Linear SVM)** - Trained on hotel feedback data

### Key Features
✅ **Multi-model Sentiment Analysis** - Compares VADER and ML predictions  
✅ **Emoji Visualization** - Shows sentiment with emotion icons (😊 happy, 😐 neutral, 😠 angry)  
✅ **Image Assets** - Displays sentiment-based images (happy.png, neutral.png, angry.png)  
✅ **Coherence Detection** - Identifies inconsistencies between text sentiment and rating  
✅ **Detailed Analysis** - Expandable section showing all prediction details and confidence scores  
✅ **Interactive Chatbot** - Auto-engages reviewers for negative/neutral feedback to understand root causes  
✅ **Agency Alert System** - Generates detailed reports with identified issues for agency follow-up

---

## 🏗️ Project Structure

```
sentiment-analysis-feedbacks/     
├── requirement.txt                 # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── feedbacks.csv              # Raw feedback dataset
│   └── feedbacks_enriched.csv      # Processed data with sentiment labels
│
├── models/
│   ├── sentiment_model.pkl         # Trained Linear SVM model
│   └── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│
├── notebooks/
│   ├── 01_exploration_nlp.ipynb    # Data exploration & VADER analysis
│   ├── 02_modelisation_nlp.ipynb   # Model training & evaluation
│   └── 03_bert_experiment.ipynb    # BERT experimentation
│
├── assets/
│   ├── happy.png                   # Positive sentiment image
│   ├── neutral.png                 # Neutral sentiment image
│   └── angry.png                   # Negative sentiment image
│
└── src/
    ├── app.py                    # Main Streamlit application
```

---

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory**
```bash
cd sentiment-analysis-feedbacks
```

2. **Install dependencies**
```bash
pip install -r requirement.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

4. **Access the app**
Open `http://localhost:8501` in your browser

---

## 📊 How It Works

### Input
1. **Text Feedback** - Enter any customer feedback text
2. **Rating** - Provide a numerical rating (1-5)

### Processing
1. **VADER Analysis** - Analyzes sentiment using lexicon-based approach
2. **ML Prediction** - Applies pre-trained TF-IDF + SVM model
3. **Hybrid Decision** - **Uses VADER as primary** (more reliable for negation and neutral detection)

### Output Phase 1: Sentiment Analysis
1. **Sentiment Metrics** - Displays ML and VADER predictions
2. **Sentiment Image** - Visual representation based on text sentiment
3. **Rating Sentiment** - What the numerical rating suggests
4. **Coherence Check** - Compares text sentiment vs rating sentiment
5. **Detailed Analysis** - Shows confidence scores and model adjustments

### Output Phase 2: Chatbot Engagement (for Negative/Neutral Feedback)
1. **Auto-triggered Chat** - Chatbot initiates conversation for negative or neutral sentiment
2. **Interactive Discussion** - Multi-turn dialogue to understand issues
3. **Issue Extraction** - System identifies specific problems from conversation
4. **End Discussion** - User clicks button to finalize conversation
5. **Agency Alert Generation** - Comprehensive report with identified issues

### Output Phase 3: Agency Alert Report
1. **Issue Summary** - Categorized problems identified from conversation
2. **Conversation Transcript** - Full dialogue history
3. **Recommendations** - Actionable next steps
4. **Export** - Download as timestamped text file

---

## � Interactive Chatbot Feature

### What is it?
An automated conversational agent that engages customers when negative or neutral sentiment is detected to understand the root causes of dissatisfaction.

### How It Works

**Trigger:** Chatbot automatically appears when sentiment is detected as "negative" or "neutral"

**Conversation Flow:**

1. **Greeting Phase**
   - Negative: "I see you had a negative experience. Could you tell me more about what went wrong?"
   - Neutral: "It seems the service was average. What could we improve?"

2. **Discussion Phase**
   - Chatbot asks clarifying questions
   - Multiple turns of conversation
   - System listens for specific issues

3. **Issue Extraction**
   - Automatically identifies problems from conversation
   - Categories include:
     - Slow service / delays
     - Rude / unprofessional staff
     - Quality issues
     - Pricing concerns
     - Cleanliness problems
     - Food quality
     - Service issues

4. **Closing Phase**
   - "Thank you for this valuable feedback. We'll address this issue with our team."
   - Prepares alert generation

### User Controls

- **✅ End Discussion Button** - Finalizes conversation and triggers alert generation
- **Chat Input Box** - Type responses to chatbot questions
- **🔄 Start New Discussion** - Begin fresh conversation after alert is generated

---

## 🚨 Agency Alert System

### Purpose
Automatically generates detailed reports for agency teams to address identified customer issues.

### Alert Contents

```
🚨 AGENCY ALERT - Negative/Neutral Feedback

Original Feedback: [Customer's text]
Sentiment: NEGATIVE/NEUTRAL
Rating: X/5

Identified Issues:
  • Issue 1 (e.g., Slow Service)
  • Issue 2 (e.g., Rude Staff)
  • Issue 3 (e.g., Quality)

Discussion Summary:
• Reviewer: [Message]
• Chatbot: [Response]
...

Recommendation: Prioritize addressing the identified issues. 
Follow up with the reviewer to confirm resolution.
```

### Features

- **Automatic Trigger** - Generates when user clicks "End Discussion"
- **Issue Identification** - Extracts specific problems from conversation
- **Full Context** - Includes complete conversation transcript
- **Download Option** - Export as text file with timestamp
- **Multiple Discussions** - Can start new conversation for same feedback

### Use Cases

1. **Service Recovery** - Identify failures and reach out proactively
2. **Root Cause Analysis** - Understand specific reasons for dissatisfaction
3. **SLA Management** - Automated alerts for critical feedback
4. **Performance Monitoring** - Track recurring issues
5. **Staff Training** - Identify training needs from conversation patterns

---

### Why VADER is Primary?
VADER excels at:
- ✅ Detecting **negation** ("not good" → negative)
- ✅ Capturing **neutral sentiment** (phrases with no strong opinion)
- ✅ Handling **simple phrases** ("bad service", "normal service", "great service")
- ✅ Processing **informal language** (emojis, slang, contractions)

### Example Results

#### ✅ Positive
**Text:** "Great service and friendly staff"  
**Rating:** 5  
**ML:** positive | **VADER:** positive | **Coherent:** ✅

#### ❌ Negative
**Text:** "Bad services"  
**Rating:** 4  
**ML:** positive | **VADER:** negative | **Final:** negative | **Coherent:** ❌

#### ⚫ Neutral
**Text:** "Normal services"  
**Rating:** 3  
**ML:** might vary | **VADER:** neutral | **Final:** neutral | **Coherent:** ✅

---

## 📈 Model Performance

### Training Data
- **Source:** Hotel reviews dataset
- **Size:** Multiple feedback entries with ratings
- **Labels:** Sentiment categories (positive, neutral, negative)

### Vectorization
- **Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features:** 5,000
- **N-grams:** 1-2 (unigrams and bigrams)

### Classification Model
- **Algorithm:** Linear SVM (Support Vector Machine)
- **Advantage:** Effective for high-dimensional text data
- **Version:** scikit-learn >= 1.8.0

---

## 🎨 UI Components

### Results Section
- **Metrics Display:** Shows sentiment predictions side-by-side
- **Sentiment Image:** Visual feedback (happy/neutral/angry face)
- **Caption:** Clearly labels the expected sentiment

### Coherence Analysis
- **✅ Success Message:** Feedback is coherent with rating
- **❌ Error Message:** Incoherence detected between text and rating

### Analysis Details (Expandable)
- ML Sentiment prediction
- VADER Sentiment prediction
- Final Sentiment Used
- VADER Confidence Score
- Sentiment derived from rating
- Info about model adjustments

---

## 📦 Requirements

```
streamlit              # Web framework
scikit-learn>=1.8.0    # ML models and TF-IDF
pandas                 # Data manipulation
numpy                  # Numerical computing
joblib                 # Model serialization
nltk                   # Natural Language Toolkit (VADER)
Pillow                 # Image processing
```

Install all with:
```bash
pip install -r requirement.txt
```

---

## 🔧 Configuration

### scikit-learn Version
Models were trained with scikit-learn 1.8.0. Using version 1.3.2 or older will cause unpickling errors. The requirement file specifies `>=1.8.0` to prevent this.

### NLTK Data
The app automatically downloads required NLTK data:
- `vader_lexicon` - Sentiment intensity lexicon

---

## 📝 Notes

### Text Processing
- Input text is analyzed as-is (no mandatory preprocessing)
- VADER handles informal language well
- ML model expects clean English text (trained on hotel reviews)

### Limitations
- **Language:** Primarily trained on English text
- **Domain:** Optimized for hotel/service feedback
- **Negation:** VADER better at negation than bag-of-words ML approach
- **Sarcasm:** Both models struggle with sarcasm detection

### Future Improvements
- [ ] Support for multilingual sentiment analysis and chatbot
- [ ] BERT model integration for complex NLP tasks
- [ ] Training on domain-specific data
- [ ] Advanced LLM-based chatbot (GPT-powered) for more natural conversations
- [ ] Confidence intervals for predictions
- [ ] Named Entity Recognition (NER) for better issue extraction
- [ ] CRM integration for automatic alert routing
- [ ] Issue trend dashboard showing patterns over time
- [ ] Automated solution suggestions based on issue types

---

## 📚 Notebooks

### 01_exploration_nlp.ipynb
- Dataset loading and exploration
- Rating distribution analysis
- Text length analysis
- VADER sentiment extraction
- Coherence analysis (rating vs text sentiment)
- Generates enriched dataset with sentiment labels

### 02_modelisation_nlp.ipynb
- Train-test split (80-20)
- TF-IDF vectorization
- Model training:
  - Logistic Regression
  - Linear SVM
  - Naive Bayes
- Model evaluation and comparison
- Confusion matrix visualization
- VADER qualitative analysis
- Model serialization (pickle)

### 03_bert_experiment.ipynb
- BERT model experimentation
- Advanced NLP approaches
- Transformer-based analysis

---

## 🎯 Use Cases

1. **Customer Feedback Analysis** - Detect inconsistencies in customer reviews
2. **Quality Assurance** - Verify rating accuracy against written feedback
3. **Customer Service** - Identify dissatisfied customers despite high ratings (or vice versa)
4. **Data Validation** - Clean and validate feedback datasets
5. **Proactive Service Recovery** - Automatically engage negative/neutral reviewers to understand issues
6. **Issue Management** - Categorize and track specific problems across feedback
7. **Agency Operations** - Receive actionable alerts for issues requiring follow-up
8. **Performance Monitoring** - Track recurring service issues over time

---

## 👤 Author
Developed as a sentiment analysis and NLP project

## 📄 License
Open source project

---

## ❓ Troubleshooting

### Issue: Model unpickling error
**Error:** `InconsistentVersionWarning: Trying to unpickle estimator...`  
**Solution:** Install scikit-learn >= 1.8.0
```bash
pip install --upgrade scikit-learn
```

### Issue: Image not displaying
**Error:** `FileNotFoundError: assets/happy.png`  
**Solution:** Ensure you're running the app from the project root directory
```bash
cd sentiment-analysis-feedbacks
streamlit run app.py
```

### Issue: NLTK data missing
**Error:** `LookupError: Resource vader_lexicon not found`  
**Solution:** The app auto-downloads it, but you can manually download:
```python
import nltk
nltk.download('vader_lexicon')
```

---

## 📞 Support

For issues or questions:
1. Check the notebooks for detailed analysis workflows
2. Review the app.py for implementation details
3. Examine the data files in the `data/` directory

---

**Last Updated:** January 10, 2026  
**Project Status:** ✅ Active & Functional with Interactive Chatbot & Agency Alert System
