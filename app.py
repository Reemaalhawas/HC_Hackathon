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


def recommend_courses(email):
    if not email.strip():
        return []  

    processed_email = preprocess_text(email)

    
    descriptions = courses['Description'].tolist()
    descriptions.append(processed_email)

    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(descriptions)
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])

    
    scores = similarity_scores[0]
    recommended_courses = sorted(
        zip(courses['Course Name'], courses['Provider'], scores),
        key=lambda x: x[2],
        reverse=True
    )


    forced_suggestions = []
    if any(keyword in email.lower() for keyword in ["ضرائب", "tax", "excel", "حساب", "محاسبة", "calculate", "calculations", "financial report"]):
        forced_suggestions.append(("Advanced Excel Training", "Coursera"))
        forced_suggestions.append(("Tax Accounting Certification", "LinkedIn Learning"))
        forced_suggestions.append(("Financial Modeling", "edX"))
    if any(keyword in email.lower() for keyword in ["database", "قواعد البيانات", "بيانات", "sql", "data management", "data analysis"]):
        forced_suggestions.append(("SQL Database Management", "Udemy"))
        forced_suggestions.append(("Data Analysis Training", "edX"))
        forced_suggestions.append(("Database Optimization", "Coursera"))
    if any(keyword in email.lower() for keyword in ["human resources", "الموارد البشرية", "الموظفين", "team management", "hr"]):
        forced_suggestions.append(("HR Management Training", "LinkedIn Learning"))
        forced_suggestions.append(("Leadership Skills Training", "Skillshare"))
        forced_suggestions.append(("Employee Engagement Strategies", "Udemy"))
    if any(keyword in email.lower() for keyword in ["cloud storage", "تخزين سحابي", "ملفات سحابية", "cloud computing", "backup", "file management"]):
        forced_suggestions.append(("Cloud Computing", "Google Digital Garage"))
        forced_suggestions.append(("File Management in Cloud Platforms", "Coursera"))
        forced_suggestions.append(("AWS Cloud Practitioner", "AWS Training"))
    if any(keyword in email.lower() for keyword in ["presentation", "عروض تقديمية", "اجتماعات", "slides", "public speaking", "meetings"]):
        forced_suggestions.append(("Presentation Skills Training", "Skillshare"))
        forced_suggestions.append(("Public Speaking Masterclass", "Coursera"))
        forced_suggestions.append(("Effective Communication", "LinkedIn Learning"))

    
    final_recommendations = forced_suggestions + [rec[:2] for rec in recommended_courses if rec[2] > 0.1][:5]
    
    
    if not final_recommendations:
        final_recommendations.append(("General Skills Training", "Udemy"))
        final_recommendations.append(("Professional Development", "Coursera"))
    
    return final_recommendations


st.set_page_config(layout="wide", page_title="Skill Bot Email Analyizer")
st.title("Skill Bot Email Analyizer ")
st.write("Analyze emails and get training program recommendations.")


with st.sidebar:
    st.subheader("Inbox")
    email_list = [
        "تأخير في تسليم الحسابات الضريبية",
        "مشاكل في قواعد البيانات",
        "الموارد البشرية وتوزيع المهام",
        "التخزين السحابي وعروض تقديمية"
    ]
    email_content = [
      "  السلام عليكم ، أثناء إعداد التقارير المالية للشهر الماضي، لاحظت وجود تناقضات في الأرقام المتعلقة بالضرائب. أحتاج إلى مراجعة العمليات الحسابية والتاكد من  اللوائح  و شكرا "," والتنواجه مشاكل متكررة مع تنظيم البيانات داخل نظام قواعد البيانات الحالي. البيانات تتكرر وهناك صعوبة في الوصول إلى المعلومات الدقيقة بسرعة. نحتاج إلى حلول فعالة لإدارة البيانات.",
        "لاحظت وجود بعض التحديات في توزيع المهام بين الموظفين وعدم وضوح الأدوار في الفريق. نحتاج إلى تطوير المهارات القيادية لتحسين الأداء.",
        "أواجه صعوبة في إدارة الملفات على التخزين السحابي وإنشاء عروض تقديمية فعالة للاجتماعات القادمة."
    ]
    selected_email_index = st.radio("Select an Email:", range(len(email_list)), format_func=lambda x: email_list[x])


st.subheader("Email Content")
st.write(email_content[selected_email_index])


st.subheader("Recommendations")
recommendations = recommend_courses(email_content[selected_email_index])
for course, provider in recommendations:
    st.write(f"- {course} by {provider}")


st.subheader("Add New Email for Analysis")
new_email = st.text_area("Enter New Email Content:", "")
if new_email:
    st.subheader("Recommendations for New Email")
    recommendations = recommend_courses(new_email)
    for course, provider in recommendations:
        st.write(f"- {course} by {provider}")
