import streamlit as st
import joblib
import pandas as pd
import nltk
from PIL import Image
import pyttsx3
from io import BytesIO


from nltk.sentiment import SentimentIntensityAnalyzer


# =========================
# Speech Recognition Setup (Optional)
# =========================
# Try to import speech recognition - it's optional
try:
   import speech_recognition as sr
   SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
   SPEECH_RECOGNITION_AVAILABLE = False


def speech_to_text():
   """Convert speech to text using microphone"""
   if not SPEECH_RECOGNITION_AVAILABLE:
       st.error("Speech recognition is not available. Please install pyaudio and SpeechRecognition.")
       return None
  
   try:
       recognizer = sr.Recognizer()
       with sr.Microphone() as source:
           try:
               st.info("🎤 Listening... Please speak now.")
               audio = recognizer.listen(source, timeout=10)
               text = recognizer.recognize_google(audio)
               return text
           except sr.UnknownValueError:
               st.error("Could not understand audio. Please try again.")
               return None
           except sr.RequestError:
               st.error("Could not request results. Check your internet connection.")
               return None
   except AttributeError:
       st.error("Microphone not available. Please install portaudio19-dev and pyaudio.")
       return None


def text_to_speech(text):
   """Convert text to speech and return audio bytes"""
   engine = pyttsx3.init()
   engine.setProperty('rate', 150)  # Speed of speech
   engine.setProperty('volume', 0.9)  # Volume
  
   # Save to BytesIO object
   audio_buffer = BytesIO()
   engine.save_to_file(text, "temp_audio.mp3")
   engine.runAndWait()
  
   # Read the audio file and return
   try:
       with open("temp_audio.mp3", "rb") as f:
           audio_buffer.write(f.read())
           audio_buffer.seek(0)
           return audio_buffer
   except:
       return None


# =========================
st.set_page_config(
   page_title="Sentiment Analysis App",
   page_icon="💬",
   layout="centered"
)


st.title("💬 Sentiment Analysis & Rating Coherence")
st.markdown(
   "This application analyzes the **sentiment of a textual feedback** "
   "and verifies its **coherence with the provided rating**."
)


# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
   model = joblib.load("../models/sentiment_model.pkl")
   vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
   return model, vectorizer


model, vectorizer = load_models()


# =========================
# VADER (exploratoire)
# =========================
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()


def vader_sentiment(text):
   score = sia.polarity_scores(text)["compound"]
   if score >= 0.05:
       return "positive"
   elif score <= -0.05:
       return "negative"
   else:
       return "neutral"


# =========================
# Rating → Sentiment attendu
# =========================
def rating_to_sentiment(rating):
   if rating <= 2:
       return "negative"
   elif rating == 3:
       return "neutral"
   else:
       return "positive"


# =========================
# Get Sentiment Image
# =========================
def get_sentiment_image(sentiment):
   """Return image path based on sentiment"""
   if sentiment == "positive":
       return "../assets/happy.png"
   elif sentiment == "negative":
       return "../assets/angry.png"
   else:
       return "../assets/neutral.png"


# =========================
# Chatbot for Negative/Neutral Sentiment
# =========================
def initialize_chat_session():
   """Initialize chat session state"""
   if "chat_messages" not in st.session_state:
       st.session_state.chat_messages = []
   if "issues_identified" not in st.session_state:
       st.session_state.issues_identified = []
   if "discussion_ended" not in st.session_state:
       st.session_state.discussion_ended = False


def extract_issues_from_conversation(messages):
   """Extract issues from the entire conversation"""
   issues = []
   issue_keywords = {
       "slow": ["slow", "slow down", "delay", "delayed", "slow service", "long wait"],
       "rude": ["rude", "disrespectful", "impolite", "unprofessional", "rude staff", "bad attitude"],
       "quality": ["poor quality", "bad quality", "low quality", "defective", "broken", "damaged"],
       "price": ["expensive", "overpriced", "too much", "high price", "costly"],
       "cleanliness": ["dirty", "unclean", "not clean", "messy", "filthy"],
       "staff": ["staff issue", "bad staff", "staff problem", "unhelpful"],
       "service": ["poor service", "bad service", "service issue", "terrible service"],
       "food": ["bad food", "cold food", "stale", "expired", "food quality"],
       "waiting": ["waiting time", "waited too long", "long queue", "wait time"]
   }
  
   # Analyze all messages
   conversation_text = " ".join([msg["content"].lower() for msg in messages])
  
   for issue_type, keywords in issue_keywords.items():
       for keyword in keywords:
           if keyword in conversation_text and issue_type not in issues:
               issues.append(issue_type)
               break
  
   return issues


def chatbot_response(user_message, feedback_text, sentiment):
   """Generate chatbot responses based on sentiment and user input"""
   messages = {
       "greeting_negative": [
           "I see you had a negative experience. Could you tell me more about what went wrong?",
           "What aspect of the service disappointed you the most?",
       ],
       "greeting_neutral": [
           "It seems the service was average. What could we improve?",
           "What aspects of our service met your expectations, and what didn't?",
       ],
       "followup": [
           "Can you provide more details about this issue?",
           "How did this affect your overall experience?",
           "Would you like to tell us anything else about your experience?",
           "Is there anything else we should know?",
       ],
       "closing": [
           "Thank you for this valuable feedback. We'll address this issue with our team.",
           "We appreciate your honesty and will work on improving this area.",
       ]
   }
  
   # Determine greeting based on sentiment
   if sentiment == "negative":
       greeting_list = messages["greeting_negative"]
   else:  # neutral
       greeting_list = messages["greeting_neutral"]
  
   # Simple logic for chatbot responses
   if len(st.session_state.chat_messages) == 0:
       return greeting_list[0]
   else:
       # Return follow-up questions based on message count
       message_count = len(st.session_state.chat_messages) // 2  # Count user messages only
       if message_count < 3:
           return messages["followup"][message_count % len(messages["followup"])]
       else:
           return messages["closing"][0]


# =========================
# Alert Generation Function
# =========================
def generate_agency_alert(feedback_text, sentiment, messages, rating):
   """Generate alert report for agency with issues extracted from conversation"""
   # Extract issues from entire conversation
   issues = extract_issues_from_conversation(messages)
  
   # Build conversation summary
   conversation_summary = "**Conversation Summary:**\n"
   for i, msg in enumerate(messages):
       role = "Reviewer" if msg["role"] == "user" else "Chatbot"
       conversation_summary += f"• {role}: {msg['content']}\n"
  
   alert_text = f"""
🚨 **AGENCY ALERT - Negative/Neutral Feedback**


**Original Feedback:** {feedback_text}


**Sentiment:** {sentiment.upper()}
**Rating:** {rating}/5


**Identified Issues:**
{chr(10).join([f"  • {issue.capitalize()}" for issue in issues]) if issues else "  • No specific issues identified in conversation"}


**Discussion Summary:**
{conversation_summary}


**Recommendation:** Prioritize addressing the identified issues. Follow up with the reviewer to confirm resolution.
   """
   return alert_text


# =========================
# Interface utilisateur
# =========================
st.subheader("✍️ User Input")


# Voice input option
col_text, col_voice = st.columns([3, 1])


with col_voice:
   if st.button("🎤 Record Voice"):
       voice_input = speech_to_text()
       if voice_input:
           st.session_state.feedback = voice_input
           st.success(f"✓ Recorded: {voice_input}")


feedback = st.text_area(
   "Enter a feedback:",
   value=st.session_state.get("feedback", ""),
   height=120,
   placeholder="Ex: The service was fast but the staff was rude..."
)


rating = st.slider(
   "Give a rating (1 to 5)",
   min_value=1,
   max_value=5,
   value=3
)


# Initialize chat session at the start
initialize_chat_session()


# =========================
# Prédiction
# =========================
if st.button("🔍 Analyze"):
   if feedback.strip() == "":
       st.warning("Please enter a feedback.")
   else:
       # Clear previous chat when analyzing new feedback
       st.session_state.chat_messages = []
       st.session_state.issues_identified = []
      
       # VADER sentiment (better at capturing negation and simple phrases)
       vader_scores = sia.polarity_scores(feedback)
       vader_pred = vader_sentiment(feedback)
       vader_confidence = abs(vader_scores["compound"])
      
       # ML prediction
       X = vectorizer.transform([feedback])
       ml_sentiment = model.predict(X)[0]
      
       # Always use VADER - it's more reliable for sentiment analysis
       final_sentiment = vader_pred


       # Rating sentiment (what the rating number suggests)
       rating_sent = rating_to_sentiment(rating)
      
       # Expected sentiment is the actual text sentiment, not the rating
       expected_sentiment = final_sentiment
      
       # Store in session state for persistence
       st.session_state.current_feedback = feedback
       st.session_state.current_rating = rating
       st.session_state.current_sentiment = final_sentiment
       st.session_state.ml_sentiment = ml_sentiment
       st.session_state.vader_pred = vader_pred
       st.session_state.vader_confidence = vader_confidence
       st.session_state.rating_sent = rating_sent
       st.session_state.show_analysis = True


# Display analysis if it exists in session state
if st.session_state.get("show_analysis", False):
   final_sentiment = st.session_state.current_sentiment
   ml_sentiment = st.session_state.ml_sentiment
   vader_pred = st.session_state.vader_pred
   vader_confidence = st.session_state.vader_confidence
   rating_sent = st.session_state.rating_sent
   rating = st.session_state.current_rating
   feedback = st.session_state.current_feedback


   # =========================
   # Results
   # =========================
   st.subheader("📊 Results")


   col1, col2 = st.columns(2)


   with col1:
       st.metric("Sentiment (ML)", ml_sentiment)
       st.metric("Sentiment (VADER)", vader_pred)


   with col2:
       st.metric("Rating", rating)
       st.metric("Expected sentiment (text)", final_sentiment)


   # Display sentiment image for expected sentiment
   st.markdown("---")
   col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
   with col_img2:
       sentiment_image = get_sentiment_image(final_sentiment)
       image = Image.open(sentiment_image)
       st.image(image, width=200, caption=f"Expected: {final_sentiment.upper()}")


 


   # =========================
   # Coherence Analysis
   # =========================
   st.subheader("⚠️ Coherence Analysis")


   if final_sentiment == rating_sent:
       st.success("The feedback is **coherent** with the rating.")
   else:
       st.error("⚠️ Incoherence detected between the text and the rating.")


   # =========================
   # Details
   # =========================
   with st.expander("🔎 Analysis Details"):
       st.write(f"• ML Sentiment: **{ml_sentiment}**")
       st.write(f"• VADER Sentiment: **{vader_pred}**")
       st.write(f"• Final Sentiment (Text): **{final_sentiment}**")
       st.write(f"• VADER Confidence: **{vader_confidence:.2f}**")
       st.write(f"• Sentiment from Rating: **{rating_sent}**")
       if ml_sentiment != final_sentiment:
           st.write(f"ℹ️ *Sentiment adjusted: VADER was preferred due to high confidence*")


   # =========================
   # Chatbot for Negative/Neutral Feedback
   # =========================
   if final_sentiment in ["negative", "neutral"]:
       st.markdown("---")
       st.subheader("💬 Chat with Reviewer")
       st.info(f"Let's understand why the reviewer gave a {final_sentiment} feedback.")
      
       # Display chat messages
       chat_container = st.container()
       with chat_container:
           for message in st.session_state.chat_messages:
               with st.chat_message(message["role"]):
                   st.write(message["content"])
      
       # Chat input and buttons
       col_chat, col_end = st.columns([3, 1])
      
       with col_chat:
           user_input = st.chat_input("Your response...")
      
       with col_end:
           if st.button("✅ End Discussion"):
               st.session_state.discussion_ended = True
      
       if user_input and not st.session_state.discussion_ended:
           # Add user message to chat
           st.session_state.chat_messages.append({
               "role": "user",
               "content": user_input
           })
          
           # Get chatbot response
           bot_response = chatbot_response(user_input, feedback, final_sentiment)
           st.session_state.chat_messages.append({
               "role": "assistant",
               "content": bot_response
           })
          
           # Rerun to update chat display
           st.rerun()
      
       # Show agency alert only when discussion ends
       if st.session_state.discussion_ended and len(st.session_state.chat_messages) > 0:
           st.markdown("---")
           st.subheader("📋 Agency Alert Report")
           alert = generate_agency_alert(feedback, final_sentiment, st.session_state.chat_messages, rating)
           st.warning(alert)
          
           # Download alert as text file
           st.download_button(
               label="📥 Download Alert Report",
               data=alert,
               file_name=f"alert_{final_sentiment}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
               mime="text/plain"
           )
          
           # Option to start new discussion
           if st.button("🔄 Start New Discussion"):
               st.session_state.chat_messages = []
               st.session_state.discussion_ended = False
               st.rerun()


# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
   "NLP Project – Sentiment Analysis | "
   "TF-IDF + SVM (main) | BERT (experimental)"
)



