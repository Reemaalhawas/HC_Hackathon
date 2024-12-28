import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load spaCy Model
nlp = spacy.load("en_core_web_sm")

# Load Training Programs Data
courses = pd.read_csv('Training_Programs.csv')

# Preprocessing Function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Recommendation Function
def recommend_courses(text):
    processed_text = preprocess_text(text)
    descriptions = courses['Description'].tolist()
    descriptions.append(processed_text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(descriptions)
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])

    scores = similarity_scores[0]
    recommended_courses = sorted(
        zip(courses['Course Name'], courses['Provider'], scores),
        key=lambda x: x[2],
        reverse=True
    )

    # Keyword-based forced suggestions
    forced_suggestions = []
    if any(keyword in text.lower() for keyword in ["tax", "excel", "حساب", "ضرائب"]):
        forced_suggestions.append(("Advanced Excel Training", "Coursera"))
        forced_suggestions.append(("Tax Accounting Certification", "LinkedIn Learning"))
    if any(keyword in text.lower() for keyword in ["database", "بيانات", "data management"]):
        forced_suggestions.append(("SQL Database Management", "Udemy"))
        forced_suggestions.append(("Data Analysis Training", "edX"))
    if any(keyword in text.lower() for keyword in ["human resources", "hr", "الموارد البشرية"]):
        forced_suggestions.append(("HR Management Training", "LinkedIn Learning"))
        forced_suggestions.append(("Leadership Skills Training", "Skillshare"))
    if any(keyword in text.lower() for keyword in ["cloud storage", "cloud", "تخزين سحابي"]):
        forced_suggestions.append(("Cloud Computing", "Google Digital Garage"))
        forced_suggestions.append(("File Management in Cloud Platforms", "Coursera"))
    if any(keyword in text.lower() for keyword in ["presentation", "عروض تقديمية", "اجتماعات"]):
        forced_suggestions.append(("Presentation Skills Training", "Skillshare"))
        forced_suggestions.append(("Public Speaking Masterclass", "Coursera"))

    final_recommendations = forced_suggestions + [rec[:2] for rec in recommended_courses if rec[2] > 0.1][:5]
    return final_recommendations

# Streamlit Interface
st.title("AI-Powered Chat Analysis")
st.write("Analyze chats and recommend training programs.")

# Simulated Chat Data
chats = [
    {"sender": "User", "message": "مرحبًا، أثناء إعداد التقارير المالية للشهر الماضي، لاحظت وجود تناقضات في الأرقام المتعلقة بالضرائب. أحتاج إلى بعض المساعدة في مراجعة العمليات الحسابية والتأكد من الامتثال للوائح. شكرًا."},
    {"sender": "AI", "message": "نوصي بدورة تدريبية في برنامج Excel المتقدم ودورة حول الضرائب والشهادات المالية."},
    {"sender": "User", "message": "السلام عليكم، نواجه مشاكل متكررة مع تنظيم البيانات داخل نظام قواعد البيانات الحالي. البيانات تتكرر وهناك صعوبة في الوصول إلى المعلومات الدقيقة بسرعة. نحتاج إلى حلول فعالة لإدارة البيانات."},
    {"sender": "AI", "message": "نوصي بدورات تدريبية في إدارة قواعد البيانات مثل SQL والتحليل البياني لإدارة البيانات."}
]

# Display Chat Messages and Recommendations
st.markdown("<style> .chat-bubble-user {padding: 10px; background-color: #E3F2FD; border-radius: 10px; margin: 5px 0; color: black;} .chat-bubble-ai {padding: 10px; background-color: #BBDEFB; border-radius: 10px; margin: 5px 0; color: black;} .chat-container {display: flex; flex-direction: column;} .recommendation-box {padding: 10px; background-color: #F5F5F5; border: 1px solid #E0E0E0; border-radius: 10px; margin-top: 10px;}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Chat Messages")
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for i, chat in enumerate(chats):
        if chat['sender'] == "User":
            st.markdown(f"<div class='chat-bubble-user'>{chat['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>{chat['message']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section for Adding New Chat
    st.header("Add New Chat Message")
    new_chat = st.text_area("Enter New Chat Message:", "")
    if new_chat:
        st.markdown(f"<div class='chat-bubble-user'>{new_chat}</div>", unsafe_allow_html=True)
        recommendations = recommend_courses(new_chat)

with col2:
    st.header("Course Recommendations")
    st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
    if new_chat:
        for course, provider in recommendations:
            st.markdown(f"<p>- {course} by {provider}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

