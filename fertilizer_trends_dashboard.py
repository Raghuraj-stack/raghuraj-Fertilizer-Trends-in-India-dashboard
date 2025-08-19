import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords")
from nltk.corpus import stopwords

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    df1 = pd.read_csv("fertiliser-status-in-india-from-2014-to-2024/fertiliser_production_capacity_2014_to_2024.csv")
    df2 = pd.read_csv("fertiliser-status-in-india-from-2014-to-2024/requirement_availability_sales_fertilisers_2014_to_2024.csv")
    return df1, df2

prod_df, req_df = load_data()

# ===============================
# Dashboard Title
# ===============================
st.title("📊 Fertilizer Trends in India (2014–2024)")

# ===============================
# Production Capacity Trends
# ===============================
st.header("Fertilizer Production Capacity")
st.line_chart(prod_df.set_index("Year"))

# ===============================
# Requirement & Availability
# ===============================
st.header("Requirement, Availability & Sales")
st.line_chart(req_df.set_index("Year"))

# ===============================
# Topic Modeling (Scikit-Learn LDA)
# ===============================
st.header("Topic Modeling from Fertilizer Reports")

# Example text corpus from CSV (replace with actual text column if available)
sample_docs = [
    "Fertilizer demand increased in 2020 due to higher crop requirements",
    "Government subsidy boosted fertilizer sales across states",
    "Production capacity improved with new urea plants",
    "Fertilizer import dependency decreased in recent years",
    "Organic fertilizers gaining importance alongside chemical fertilizers"
]

# Preprocess & Vectorize
stop_words = stopwords.words("english")
vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(sample_docs)

# Train LDA model
lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
lda_model.fit(X)

terms = vectorizer.get_feature_names_out()

st.subheader("Discovered Topics")
for idx, topic in enumerate(lda_model.components_):
    top_terms = [terms[i] for i in topic.argsort()[-8:]]
    st.write(f"**Topic {idx+1}:**", ", ".join(top_terms))

