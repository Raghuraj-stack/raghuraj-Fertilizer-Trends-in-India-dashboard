import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import StringIO

# set_page_config must come here, after import streamlit as st
st.set_page_config(page_title="Fertilizer Trends Dashboard", layout="wide")

# now the rest of your code
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
# Helper: Detect Year column safely
# ===============================
def get_year_column(df):
    for col in df.columns:
        if "year" in col.lower():
            return col
    return None

year_col_prod = get_year_column(prod_df)
year_col_req = get_year_column(req_df)

# ===============================
# Dashboard Title
# ===============================
st.set_page_config(page_title="Fertilizer Trends Dashboard", layout="wide")
st.title("🌱 Fertilizer Trends in India (2014–2024)")

# ===============================
# Tabs for Navigation
# ===============================
tab1, tab2 = st.tabs(["📊 Fertilizer Trends", "🧠 Topic Modeling"])

# ===============================
# Fertilizer Trends Tab
# ===============================
with tab1:
    st.subheader("Production Capacity and Usage Trends")

    if year_col_prod:
        prod_df_clean = prod_df.copy()
        for col in prod_df_clean.columns:
            if col != year_col_prod:
                prod_df_clean[col] = pd.to_numeric(prod_df_clean[col], errors="coerce")

        fig1 = px.line(
            prod_df_clean,
            x=year_col_prod,
            y=[c for c in prod_df_clean.columns if c != year_col_prod],
            markers=True,
            title="Fertilizer Production Capacity Over Time"
        )
        fig1.update_layout(legend_title_text="Fertilizer Type", height=500)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("⚠️ No 'Year' column found in Production dataset.")
        st.write("Columns:", prod_df.columns.tolist())

    if year_col_req:
        req_df_clean = req_df.copy()
        for col in req_df_clean.columns:
            if col != year_col_req:
                req_df_clean[col] = pd.to_numeric(req_df_clean[col], errors="coerce")

        fig2 = px.line(
            req_df_clean,
            x=year_col_req,
            y=[c for c in req_df_clean.columns if c != year_col_req],
            markers=True,
            title="Requirement, Availability & Sales Over Time"
        )
        fig2.update_layout(legend_title_text="Category", height=500)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ No 'Year' column found in Requirement dataset.")
        st.write("Columns:", req_df.columns.tolist())

# ===============================
# Topic Modeling Tab
# ===============================
with tab2:
    st.subheader("Discovered Themes in Fertilizer Data")

    # Combine text data
    if all(col in req_df.columns for col in ["Requirement", "Availability", "Sales"]):
        text_data = (
            req_df["Requirement"].astype(str) + " " +
            req_df["Availability"].astype(str) + " " +
            req_df["Sales"].astype(str)
        ).tolist()
    else:
        text_data = req_df.astype(str).apply(" ".join, axis=1).tolist()

    num_topics = st.sidebar.slider("Number of Topics", 2, 6, 3)

    # Vectorization
    stop_words = stopwords.words("english")
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(text_data)

    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)

    terms = vectorizer.get_feature_names_out()
    color_maps = ["Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"]

    topics_data = []

    for idx, topic in enumerate(lda_model.components_):
        top_terms = [terms[i] for i in topic.argsort()[-15:]]
        st.markdown(f"### 🏷️ Topic {idx+1}")
        st.write(", ".join(top_terms))

        topics_data.append({"Topic": f"Topic {idx+1}", "Top Words": ", ".join(top_terms)})

        word_freq = {terms[i]: topic[i] for i in topic.argsort()[-30:]}
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

    # Download Topics as CSV
    if topics_data:
        topics_df = pd.DataFrame(topics_data)
        csv_buffer = StringIO()
        topics_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Topics as CSV",
            data=csv_buffer.getvalue(),
            file_name="fertilizer_topics.csv",
            mime="text/csv"
        )
