import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from io import StringIO
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ──────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fertilizer Trends Dashboard", layout="wide")

# NLTK stopwords (safe download)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
from nltk.corpus import stopwords


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Update these paths if your repo structure differs
    prod = pd.read_csv(
        "fertiliser-status-in-india-from-2014-to-2024/fertiliser_production_capacity_2014_to_2024.csv"
    )
    req = pd.read_csv(
        "fertiliser-status-in-india-from-2014-to-2024/requirement_availability_sales_fertilisers_2014_to_2024.csv"
    )
    return prod, req

prod_df, req_df = load_data()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: detect columns robustly
# ──────────────────────────────────────────────────────────────────────────────
def find_year_col(df: pd.DataFrame):
    for c in df.columns:
        if "year" in str(c).lower():
            return c
    return None

def find_place_col(df: pd.DataFrame):
    # try common place columns
    candidates = ["State", "STATE", "state", "UT", "District", "Region"]
    for c in df.columns:
        if c in candidates or str(c).lower() in ["state", "ut", "district", "region"]:
            return c
    # if nothing, return None (national-level)
    return None

def coerce_numeric(df: pd.DataFrame, exclude: list):
    out = df.copy()
    for c in out.columns:
        if c not in exclude:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def melt_long(df: pd.DataFrame, year_col: str, place_col: str | None):
    # long format: Year, Place(optional), Fertilizer, Value
    id_vars = [year_col] + ([place_col] if place_col else [])
    value_vars = [c for c in df.columns if c not in id_vars]
    long = df.melt(id_vars=id_vars, value_vars=value_vars,
                   var_name="Fertilizer", value_name="Value")
    return long

def list_numeric_fertilizers(df: pd.DataFrame, exclude: list):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]


# ──────────────────────────────────────────────────────────────────────────────
# Figure out schema
# ──────────────────────────────────────────────────────────────────────────────
year_col_req = find_year_col(req_df)
place_col_req = find_place_col(req_df)
req_df = coerce_numeric(req_df, exclude=[year_col_req, place_col_req] if place_col_req else [year_col_req])

# If production file is needed, we’ll keep it available but the main “usage across places” is from req_df
year_col_prod = find_year_col(prod_df)
place_col_prod = find_place_col(prod_df)
if year_col_prod:
    prod_df = coerce_numeric(prod_df, exclude=[year_col_prod, place_col_prod] if place_col_prod else [year_col_prod])

# Fertilizer columns (numeric)
fert_cols_req = list_numeric_fertilizers(req_df, exclude=[year_col_req, place_col_req] if place_col_req else [year_col_req])

# Early sanity
if not year_col_req:
    st.error("Could not detect a 'Year' column in the requirement/availability/sales dataset. Please ensure a Year field exists.")
    st.stop()

if not fert_cols_req:
    st.error("No numeric fertilizer columns found to plot. Please check your dataset.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar filters & controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filters")

# Year range
years_sorted = sorted(req_df[year_col_req].dropna().unique())
min_y, max_y = int(min(years_sorted)), int(max(years_sorted))
year_range = st.sidebar.slider("Year range", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1)

# Place filter (if exists)
selected_places = None
if place_col_req:
    all_places = sorted([p for p in req_df[place_col_req].dropna().unique()])
    selected_places = st.sidebar.multiselect("Place (State/Region)", options=all_places, default=all_places[:5] if len(all_places) > 5 else all_places)

# Fertilizer(s)
selected_ferts = st.sidebar.multiselect("Fertilizer types", options=fert_cols_req, default=fert_cols_req[:3])

# Highlight which fertilizer (thicker line)
highlight_fert = st.sidebar.selectbox("Highlight fertilizer", options=selected_ferts if selected_ferts else fert_cols_req, index=0)

# Chart type
chart_type = st.sidebar.radio("Chart type", options=["Line", "Area", "Bar"], index=0)

# Data/Table toggle
show_table = st.sidebar.checkbox("Show filtered data table", value=False)

# Word cloud options
st.sidebar.markdown("---")
wc_places_mode = st.sidebar.radio("Word cloud by", ["Selected places", "All places combined"], index=0)
wc_max_words = st.sidebar.slider("Word cloud max words", 20, 200, 80, 10)

# Topics slider
st.sidebar.markdown("---")
num_topics = st.sidebar.slider("Number of topics (topic modeling tab)", 2, 8, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Title & layout
# ──────────────────────────────────────────────────────────────────────────────
st.title("🌱 Fertilizer Trends in India (2014–2024)")
st.caption("Filter by year, place, and fertilizer. Highlight a fertilizer to compare its trend. Export data and view word clouds and topics.")


# ──────────────────────────────────────────────────────────────────────────────
# Filter data
# ──────────────────────────────────────────────────────────────────────────────
mask_year = (req_df[year_col_req].astype(int) >= year_range[0]) & (req_df[year_col_req].astype(int) <= year_range[1])
df_filt = req_df.loc[mask_year].copy()
if place_col_req and selected_places:
    df_filt = df_filt[df_filt[place_col_req].isin(selected_places)]

# Prepare long format for plotting
long_df = melt_long(df_filt[[year_col_req] + ([place_col_req] if place_col_req else []) + selected_ferts], year_col_req, place_col_req)


# ──────────────────────────────────────────────────────────────────────────────
# KPI cards (simple, understandable)
# ──────────────────────────────────────────────────────────────────────────────
def kpi_area():
    # Total usage across selected fertilizers & places in end year vs start year
    cols = st.columns(3)
    # Aggregate by year
    agg = long_df.groupby(year_col_req)["Value"].sum(min_count=1).reset_index()
    total_selected = agg["Value"].sum()
    start_val = float(agg.loc[agg[year_col_req] == year_range[0], "Value"].sum()) if (agg[year_col_req] == year_range[0]).any() else float("nan")
    end_val = float(agg.loc[agg[year_col_req] == year_range[1], "Value"].sum()) if (agg[year_col_req] == year_range[1]).any() else float("nan")
    growth = (end_val - start_val) / start_val * 100 if (start_val and start_val == start_val) else float("nan")

    cols[0].metric("Total (selected range)", f"{total_selected:,.0f}")
    cols[1].metric(f"Value in {year_range[0]}", f"{start_val:,.0f}" if start_val == start_val else "—")
    cols[2].metric(f"Growth to {year_range[1]}", f"{growth:,.1f}%" if growth == growth else "—")

kpi_area()


# ──────────────────────────────────────────────────────────────────────────────
# Charts (highlight selected fertilizer)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Trends by Fertilizer")
help_txt = "Use the sidebar to change the chart type and the highlighted fertilizer. Hover for exact values; drag to zoom."
st.caption(help_txt)

def make_chart(df_long: pd.DataFrame, chart_kind: str):
    # If places exist, facet by place to make it easy to read
    facet_kwargs = {"facet_col": place_col_req, "facet_col_wrap": 3} if place_col_req else {}

    # Split highlighted vs others for line styling
    sel = df_long[df_long["Fertilizer"] == highlight_fert]
    oth = df_long[df_long["Fertilizer"] != highlight_fert]

    figs = []

    if chart_kind == "Line":
        # Others (faded)
        if not oth.empty:
            f_oth = px.line(
                oth, x=year_col_req, y="Value", color="Fertilizer",
                line_group="Fertilizer", markers=True, opacity=0.35,
                **facet_kwargs
            )
            figs.append(f_oth)

        # Highlighted (thick)
        if not sel.empty:
            f_sel = px.line(
                sel, x=year_col_req, y="Value", color="Fertilizer",
                line_group="Fertilizer", markers=True, **facet_kwargs
            )
            f_sel.update_traces(line=dict(width=4))
            figs.append(f_sel)

    elif chart_kind == "Area":
        # Stack selected fertilizers (others + highlighted; highlighted drawn last and thicker)
        f = px.area(df_long, x=year_col_req, y="Value", color="Fertilizer", groupnorm=None, **facet_kwargs)
        # Increase highlighted trace width
        f.for_each_trace(lambda tr: tr.update(line=dict(width=4)) if tr.name == highlight_fert else tr.update(opacity=0.5))
        figs.append(f)

    else:  # Bar
        f = px.bar(df_long, x=year_col_req, y="Value", color="Fertilizer", barmode="group", **facet_kwargs)
        # Make highlighted bars stand out
        f.for_each_trace(lambda tr: tr.update(marker_line_width=2, marker_line_color="black") if tr.name == highlight_fert else tr.update(opacity=0.6))
        figs.append(f)

    # Combine (if multiple figures)
    base = None
    for i, fig in enumerate(figs):
        if i == 0:
            base = fig
        else:
            for tr in fig.data:
                base.add_trace(tr)
            base.layout.update(fig.layout)
    if base:
        base.update_layout(legend_title_text="Fertilizer", height=520)
    return base

chart = make_chart(long_df, chart_type)
st.plotly_chart(chart, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data table + download
# ──────────────────────────────────────────────────────────────────────────────
if show_table:
    st.subheader("🧾 Filtered Data")
    st.dataframe(long_df.sort_values([year_col_req, "Fertilizer"]))

    csv_buf = StringIO()
    long_df.to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Download filtered data (CSV)",
        data=csv_buf.getvalue(),
        file_name="fertilizer_filtered_trends.csv",
        mime="text/csv"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Word clouds (place-wise or combined)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("☁️ Word Clouds — fertilizer prominence")
st.caption("Each cloud uses fertilizer names as words, sized by their average usage in your current selection. This keeps words correct and easy to understand.")

def draw_wordcloud(weights: dict, title: str):
    if not weights:
        st.info(f"No data for {title}.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(weights)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.markdown(f"**{title}**")
    st.pyplot(fig)

if wc_places_mode == "Selected places" and place_col_req:
    # One cloud per place (grid)
    # Compute average by place across selected years
    with st.container():
        if selected_places:
            cols = st.columns(2)
            col_idx = 0
            for place in selected_places:
                sub = df_filt[df_filt[place_col_req] == place]
                if sub.empty:
                    continue
                weights = {f: float(sub[f].mean(skipna=True)) for f in selected_ferts}
                with cols[col_idx % 2]:
                    draw_wordcloud(weights, title=f"{place}")
                col_idx += 1
        else:
            st.info("Select at least one place to render place-wise word clouds.")
else:
    # Combined cloud for all selected places/years
    weights = {f: float(df_filt[f].mean(skipna=True)) for f in selected_ferts}
    draw_wordcloud(weights, title="All selected places combined")


# ──────────────────────────────────────────────────────────────────────────────
# Topic Modeling (optional, separate tab)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("🧠 Topic Modeling (experimental)"):
    st.caption("Creates themes from text derived from numeric columns. For clearer topics, prefer the word clouds above.")
    # Build a text corpus per row from numeric columns: "FertilizerName" repeated ~ scaled value
    # This keeps words readable and relevant.
    scaled_docs = []
    scale_base = df_filt[selected_ferts].max(numeric_only=True).replace(0, 1).to_dict()

    for _, row in df_filt.iterrows():
        tokens = []
        for fert in selected_ferts:
            val = pd.to_numeric(row.get(fert, 0), errors="coerce")
            if pd.isna(val):
                continue
            # scale repetitions (keep bounded)
            reps = int(max(1, min(10, round(val / (scale_base[fert] if scale_base[fert] else 1) * 10))))
            tokens.extend([fert] * reps)
        # include place name as context, lightly
        if place_col_req and pd.notna(row.get(place_col_req, None)):
            tokens.extend([str(row[place_col_req])] * 2)
        scaled_docs.append(" ".join(tokens) if tokens else "")

    # Vectorize + LDA
    sw = stopwords.words("english")
    vect = CountVectorizer(stop_words=sw)
    X = vect.fit_transform([d for d in scaled_docs if isinstance(d, str)])
    if X.shape[0] >= num_topics and X.shape[1] > 0:
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)
        terms = vect.get_feature_names_out()

        for t_idx, topic in enumerate(lda.components_):
            top_terms = [terms[i] for i in topic.argsort()[-10:]]
            st.markdown(f"**Topic {t_idx+1}:** " + ", ".join(top_terms))
    else:
        st.info("Not enough textual variety to run LDA with current filters. Try widening the year range or selecting more fertilizers.")


# ──────────────────────────────────────────────────────────────────────────────
# (Optional) Production tab (if you want to expose production file too)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("🏭 Production capacity (optional)"):
    if year_col_prod:
        prod_filt = prod_df[(prod_df[year_col_prod].astype(int) >= year_range[0]) & (prod_df[year_col_prod].astype(int) <= year_range[1])]
        # Detect fertilizer-like columns
        fert_cols_prod = list_numeric_fertilizers(prod_filt, exclude=[year_col_prod, place_col_prod] if place_col_prod else [year_col_prod])
        if fert_cols_prod:
            prod_long = melt_long(prod_filt[[year_col_prod] + fert_cols_prod], year_col_prod, None)
            figp = px.line(prod_long, x=year_col_prod, y="Value", color="Fertilizer", markers=True, title="Production Capacity Over Time")
            figp.update_layout(legend_title_text="Fertilizer", height=480)
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.info("No numeric production columns detected.")
    else:
        st.info("No 'Year' column detected in production dataset.")
