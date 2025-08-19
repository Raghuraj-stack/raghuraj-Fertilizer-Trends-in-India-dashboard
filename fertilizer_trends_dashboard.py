import os
import nltk
import streamlit as st
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Sector Keywords (adjusted for fertilizer/agriculture context)
sector_keywords = {
    'fertilizer_policy': ['fertilizer use', 'subsidies', 'nitrogen', 'phosphorus', 'potassium', 'production'],
    'agriculture': ['crop yield', 'irrigation', 'agroforestry', 'sustainable farming', 'fertilizer application'],
    'government_regulation': ['policy', 'regulation', 'subsidy scheme', 'import', 'export'],
    'environmental_impact': ['soil health', 'pollution', 'sustainability', 'climate effect', 'nutrient loss']
}

# Load New Kaggle Dataset
dataset_path = '/content/fertiliser-status-in-india-from-2014-to-2024/fertiliser_production_capacity_2014_to_2024.csv'
df = pd.read_csv(dataset_path)

# Generate text from available columns
df['text'] = df.apply(lambda row: f"{row['fertiliser_type']} production in {row['state']} with capacity {row['fy_2023_24']} tons in {row['sector']} sector", axis=1)
df = df.dropna(subset=['text', 'fertiliser_type', 'state', 'fy_2023_24'])  # Drop rows with missing key data

# Extract texts and filenames
texts = df['text'].tolist()
filenames = [f"{row['fy_2023_24']}_{row['state']}_fertilizer_{i}.txt" if 'fy_2023_24' in df.columns else f"doc_{i}_fertilizer.txt" for i, row in df.iterrows()]

# Preprocess and LDA
processed_texts = [word_tokenize(text.lower()) for text in texts if text]
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=20)
topic_labels = [f"Topic {i}: {lda_model.print_topic(i, topn=3)}" for i in range(5)]

# Sector Classification
sector_data = defaultdict(list)
for i, filename in enumerate(filenames):
    text = ' '.join(texts[i]) if texts[i] else ''
    sector = 'unknown'
    for sec, keywords in sector_keywords.items():
        if any(keyword in text for keyword in keywords):
            sector = sec
            break
    year = filename.split('_')[0] if '_' in filename else 'unknown_year'
    topics = lda_model[corpus[i]]
    sector_data[sector].append({'year': year, 'topics': topics})

# Dashboard
st.title("Fertilizer Trends Dashboard - Kaggle Dataset")

# Sidebar for filters
st.sidebar.header("Filters")
selected_sector = st.sidebar.text_input("Search Sector (e.g., fertilizer_policy, agriculture, government_regulation):").lower()
if not selected_sector or selected_sector not in sector_data:
    st.sidebar.error("Invalid sector. Choose from available sectors.")
    st.stop()

years_available = sorted(set(data['year'] for data in sector_data[selected_sector]))
selected_years = st.sidebar.multiselect("Select Years (or all for combined):", years_available, default=years_available)
selected_topic = st.sidebar.selectbox("Select Topic", list(range(5)) + [-1], format_func=lambda x: topic_labels[x] if x >= 0 else "All Topics")

# Filter Data
topic_data = defaultdict(list)
for data in sector_data[selected_sector]:
    year = data['year']
    if year in selected_years or not selected_years:
        for topic_id, prob in data['topics']:
            topic_data[topic_id].append(prob)

# Debug
st.write("Selected Sector:", selected_sector)
st.write("Selected Years:", selected_years)
st.write("Topic Data:", dict(topic_data))

# Trend Line Chart
plt.figure(figsize=(10, 6))
for topic_id in topic_data:
    years = [d['year'] for d in sector_data[selected_sector] if d['year'] in selected_years]
    probs = topic_data[topic_id]
    plt.plot(years, probs, label=topic_labels[topic_id], marker='o')
plt.xlabel('Year')
plt.ylabel('Topic Probability')
plt.title(f'Topic Trends - {selected_sector.replace("_", " ").capitalize()} Sector')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Combined Bar Chart
if len(selected_years) > 1 and topic_data:
    combined_data = [sum(topic_data[tid]) / len(topic_data[tid]) if topic_data[tid] else 0 for tid in range(5)]
    plt.figure(figsize=(10, 6))
    plt.bar(topic_labels, combined_data)
    plt.title(f'Combined Topic Probabilities - {selected_sector.replace("_", " ").capitalize()} Sector')
    st.pyplot(plt)
else:
    st.warning("No data for combined chart. Select multiple years or check data.")
