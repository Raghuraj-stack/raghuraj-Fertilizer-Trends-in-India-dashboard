import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# =====================================
# Streamlit Page Config
# =====================================
st.set_page_config(page_title="Fertilizer Trends Dashboard", layout="wide")

# =====================================
# Load Data
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("fertiliser-status-in-india-from-2014-2024.csv")
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔧 Filters")
year_range = st.sidebar.slider("Select Year Range", 2014, 2024, (2014, 2024))
fertilizer_type = st.sidebar.selectbox("Select Fertilizer Type", df.columns[2:])
num_topics = st.sidebar.slider("Number of Topics", 2, 6, 2)

# Filtered data
filtered_df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

# =====================================
# Dashboard Title
# =====================================
st.title("🌱 Fertilizer Trends in India (2014–2024)")

tab1, tab2, tab3 = st.tabs(["📈 Fertilizer Trends", "🧾 Topic Modeling", "⚖️ Compare Places"])

# =====================================
# Fertilizer Trends Tab
# =====================================
with tab1:
    st.subheader(f"Fertilizer Trend: {fertilizer_type}")
    fig = px.line(filtered_df, x="Year", y=fertilizer_type, color="State",
                  title=f"Trend of {fertilizer_type} (2014–2024)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Distribution Across States")
    fig_bar = px.bar(filtered_df, x="State", y=fertilizer_type, color="State",
                     title=f"State-wise {fertilizer_type} Usage")
    st.plotly_chart(fig_bar, use_container_width=True)

# =====================================
# Topic Modeling Tab
# =====================================
with tab2:
    st.subheader("Discovered Themes in Fertilizer Data")

    # Combine all text columns for topic modeling
    text_data = df["State"].astype(str) + " " + df["Year"].astype(str) + " " + df[fertilizer_type].astype(str)

    vectorizer = CountVectorizer(stop_words="english")
    dtm = vectorizer.fit_transform(text_data)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    terms = vectorizer.get_feature_names_out()

    for idx, topic in enumerate(lda.components_):
        st.markdown(f"### 🏷️ Topic {idx+1}")
        # Get top words
        top_indices = topic.argsort()[-15:]
        top_terms = [terms[i] for i in top_indices]
        st.write(", ".join(top_terms))

        # Wordcloud
        word_freq = {terms[i]: topic[i] for i in top_indices}
        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# =====================================
# Compare Places Tab
# =====================================
with tab3:
    st.subheader("⚖️ Compare Fertilizer Usage Between Two Places")

    states = df["State"].unique()
    col1, col2 = st.columns(2)

    with col1:
        place1 = st.selectbox("Select First Place", states, key="place1")
    with col2:
        place2 = st.selectbox("Select Second Place", states, key="place2")

    compare_df = filtered_df[filtered_df["State"].isin([place1, place2])]

    if not compare_df.empty:
        fig_compare = px.line(compare_df, x="Year", y=fertilizer_type, color="State",
                              title=f"Comparison of {fertilizer_type} Usage ({place1} vs {place2})")
        st.plotly_chart(fig_compare, use_container_width=True)
