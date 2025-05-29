import streamlit as st
import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# App title
st.set_page_config(page_title="Scopus Keywords Network", layout="wide")
st.title("🔗 Scopus Keywords Co-occurrence Network")

# Sidebar: settings
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Minimum co-occurrence frequency", min_value=1, max_value=10, value=2
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload Scopus CSV file", type=["csv"], help="Export from Scopus with keyword columns"
)

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file, low_memory=False)

    # Identify keyword columns
    keyword_cols = [col for col in df.columns if "keyword" in col.lower()]
    if not keyword_cols:
        st.error("No keyword column found. Make sure your CSV has a column named 'Author Keywords' or similar.")
    else:
        # Let user choose which column
        chosen_col = st.sidebar.selectbox("Select keyword column", keyword_cols)
        # Extract and preprocess keywords
        raw_lists = df[chosen_col].dropna().astype(str).str.split(r";|,|")
        # Normalize
        keyword_lists = []
        for kws in raw_lists:
            cleaned = [kw.strip().lower() for kw in kws if kw.strip()]
            if cleaned:
                keyword_lists.append(cleaned)

        # Build co-occurrence counts
        pairs = []
        for kws in keyword_lists:
            for a, b in combinations(set(kws), 2):
                pairs.append(tuple(sorted((a, b))))
        co_counts = Counter(pairs)

        # Build network graph
        G = nx.Graph()
        for (a, b), w in co_counts.items():
            if w >= threshold:
                G.add_edge(a, b, weight=w)

        if G.number_of_nodes() == 0:
            st.warning("No edges with the given threshold. Try lowering the threshold.")
        else:
            st.subheader("Network Preview")
            # Generate pyvis network
            net = Network(height="600px", width="100%", notebook=False)

            # add the physics control panel:
            net.show_buttons(filter_=['physics'])

            # 2. Tune the ForceAtlas2 layout a bit (optional)
            net.force_atlas_2based(
                gravity=-50,         # repulsion from center
                central_gravity=0.01,
                spring_length=150,    # desired link length
                spring_strength=0.08,
                damping=0.4
            )
            
            # 3. Size nodes by their degree (so big hubs stand out)
            for node in net.nodes:
                node['size'] = G.degree(node['id']) * 5  # adjust multiplier as needed
                
            net.from_nx(G)
            # Save to temporary HTML
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                html_path = tmp.name
            # Display in Streamlit
            components.html(
                open(html_path, 'r', encoding='utf-8').read(),
                height=600,
                scrolling=True
            )

else:
    st.info("Upload a Scopus CSV to get started.")
