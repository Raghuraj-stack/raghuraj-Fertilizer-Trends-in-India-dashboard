import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# Download stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    df1 = pd.read_csv(
        "fertiliser-status-in-india-from-2014-to-2024/fertiliser_production_capacity_2014_to_2024.csv"
    )
    df2 = pd.read_csv(
        "fertiliser-status-in-india-from-2014-to-2024/requirement_availability_sales_fertilisers_2014_to_2024.csv"
    )
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
st.header("Topic Modeling from Fertilizer Dataset")

# Combine text from requirement, availability, and sales columns
if all(col in req_df.columns for col in ["Requirement", "Availability", "Sales"]):
    text_data = (
        req_df["Requirement"].astype(str) + " " +
        req_df["Availability"].astype(str) + " " +
        req_df["Sales"].astype(str)
    ).tolist()
else:
    # fallback: join all columns into text per row
    text_data = req_df.astype(str).apply(" ".join, axis=1).tolist()

# Sidebar for topic selection
num_topics = st.sidebar.slider("Number of Topics", 2, 6, 3)

# Vectorization
stop_words = stopwords.words("english")
vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(text_data)

# Train LDA model
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(X)

terms = vectorizer.get_feature_names_out()

# Define color themes for topics
color_maps = ["Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"]

st.subheader("Discovered Topics from Fertilizer Data")
for idx, topic in enumerate(lda_model.components_):
    top_terms = [terms[i] for i in topic.argsort()[-15:]]  # top 15 words
    st.write(f"**Topic {idx+1}:**", ", ".join(top_terms))

    # Generate weighted word frequencies
    word_freq = {terms[i]: topic[i] for i in topic.argsort()[-30:]}  # top 30 words

    # Create word cloud with a distinct color map per topic
    wc = WordCloud(
        width=600,
        height=400,
        background_color="white",
        colormap=color_maps[idx % len(color_maps)]
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

