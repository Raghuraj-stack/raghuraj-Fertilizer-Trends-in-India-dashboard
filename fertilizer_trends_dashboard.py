import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import StringIO

# ✅ Page config
st.set_page_config(page_title="Fertilizer Trends Dashboard", layout="wide")

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
# Clean Year Column
# ===============================
def clean_year(val):
    try:
        return int(str(val).split("-")[0])  # take first part if it's like "2014-15"
    except:
        return None

def get_year_column(df):
    for col in df.columns:
        if "year" in col.lower():
            return col
    return None

year_col_prod = get_year_column(prod_df)
year_col_req = get_year_column(req_df)

if year_col_prod:
    prod_df[year_col_prod] = prod_df[year_col_prod].apply(clean_year)
if year_col_req:
    req_df[year_col_req] = req_df[year_col_req].apply(clean_year)

# ===============================
# Dashboard Title
# ===============================
st.title("🌱 Fertilizer Trends in India (2014–2024)")

# ===============================
# Sidebar Filters
# ===============================
if year_col_req:
    years_sorted = sorted([y for y in req_df[year_col_req].dropna().unique()])
    min_y, max_y = min(years_sorted), max(years_sorted)
    year_range = st.sidebar.slider("Select Year Range", min_value=min_y, max_value=max_y, value=(min_y, max_y))
else:
    year_range = None

fertilizer_options = [c for c in req_df.columns if c not in [year_col_req, "Place", "State"]]
fertilizer_type = st.sidebar.selectbox("Select Fertilizer Type", fertilizer_options)

place_col = None
for col in req_df.columns:
    if col.lower() in ["place", "state", "region"]:
        place_col = col
        break

place1, place2 = None, None
if place_col:
    places = sorted(req_df[place_col].dropna().unique())
    place1 = st.sidebar.selectbox("Select Place 1", places, index=0)
    place2 = st.sidebar.selectbox("Select Place 2", places, index=min(1, len(places)-1))

# ===============================
# Tabs
# ===============================
tab1, tab2 = st.tabs(["📊 Fertilizer Trends", "🧠 Topic Modeling"])

# ===============================
# Fertilizer Trends Tab
# ===============================
with tab1:
    st.subheader("Production Capacity Trends")

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

    st.subheader("Requirement, Availability & Sales Trends")

    if year_col_req:
        req_df_clean = req_df.copy()
        for col in req_df_clean.columns:
            if col != year_col_req:
                req_df_clean[col] = pd.to_numeric(req_df_clean[col], errors="coerce")

        fig2 = px.line(
            req_df_clean,
            x=year_col_req,
            y=[c for c in req_df_clean.columns if c != year_col_req and c != place_col],
            markers=True,
            title="Requirement, Availability & Sales Over Time"
        )
        fig2.update_layout(legend_title_text="Category", height=500)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Compare Fertilizer Usage Between Two Places")

    if year_col_req and fertilizer_type and place_col:
        filtered_df = req_df[(req_df[year_col_req] >= year_range[0]) & (req_df[year_col_req] <= year_range[1])]

        # Show summary metrics
        st.markdown("### 📈 Summary Metrics")
        colm1, colm2, colm3 = st.columns(3)
        try:
            max_val = filtered_df[fertilizer_type].max()
            min_val = filtered_df[fertilizer_type].min()
            avg_val = filtered_df[fertilizer_type].mean()
            colm1.metric("Peak Usage", f"{max_val:.2f}")
            colm2.metric("Lowest Usage", f"{min_val:.2f}")
            colm3.metric("Average Usage", f"{avg_val:.2f}")
        except:
            st.warning("⚠️ Unable to calculate metrics for selected fertilizer.")

        # Compare two places
        col1, col2 = st.columns(2)

        with col1:
            df1 = filtered_df[filtered_df[place_col] == place1]
            if not df1.empty:
                figp1 = px.line(df1, x=year_col_req, y=fertilizer_type, markers=True,
                               title=f"{fertilizer_type} Trend in {place1}")
                st.plotly_chart(figp1, use_container_width=True)
            else:
                st.warning(f"No data available for {place1}")

        with col2:
            df2 = filtered_df[filtered_df[place_col] == place2]
            if not df2.empty:
                figp2 = px.line(df2, x=year_col_req, y=fertilizer_type, markers=True,
                               title=f"{fertilizer_type} Trend in {place2}")
                st.plotly_chart(figp2, use_container_width=True)
            else:
                st.warning(f"No data available for {place2}")

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
       # Get top word indices sorted by importance
        top_indices = topic.argsort()[-15:]
# Map them back to actual words
        top_terms = [terms[i] for i in top_indices]

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
