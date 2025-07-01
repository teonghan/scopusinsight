import streamlit as st
import pandas as pd
import plotly.express as px
import re
from st_aggrid import AgGrid, GridOptionsBuilder
from scipy.stats import linregress
from burst_detection import burst_detection
import numpy as np

# ========================
#     HELPER FUNCTIONS
# ========================

def clean_issn(val):
    """Clean and format ISSN/EISSN values to 'XXXX-XXXX' or None."""
    if pd.isna(val):
        return None
    val = str(val).strip().replace("-", "")
    if val.upper() == "NULL" or val == "":
        return None
    # Allow ISSN with 'X' as check digit
    if not (val[:-1].isdigit() and (val[-1].isdigit() or val[-1].upper() == "X")):
        return None
    val = val.zfill(8)
    return val[:4] + '-' + val[4:]

def extract_id(id_str):
    """Extract Scopus author ID from 'Name (ID)' or return trimmed input."""
    m = re.search(r'\((\d+)\)$', id_str)
    return m.group(1) if m else id_str.strip()

def extract_name(id_str):
    """Extract author name from 'Name (ID)' or return trimmed input."""
    m = re.match(r'(.+)\s\(\d+\)$', id_str.strip())
    return m.group(1) if m else id_str.strip()

def author_name_variants(surname_first):
    """Given 'Surname Initials', return possible variants for matching."""
    parts = surname_first.split()
    if len(parts) < 2:
        return {surname_first}
    surname = parts[0]
    initials = " ".join(parts[1:])
    return {surname_first, f"{initials} {surname}".strip()}

def unique_concatenate(x):
    """Concatenate unique, non-empty (case-insensitive) values as '; '."""
    seen = set()
    result = []
    for item in x:
        cleaned = item.lower().strip()
        if cleaned not in seen and str(item).strip() != "":
            seen.add(cleaned)
            result.append(item.strip())
    return "; ".join(result)

def parse_asjc_list(asjc_str):
    """Parse ASJC string to list of integer codes."""
    return [int(code) for code in str(asjc_str).replace(" ", "").replace(",", ";").split(";") if code.isdigit()]

def get_author_canonical_info(df):
    """
    Given a dataframe with author rows, return a dataframe mapping Author ID to:
    - unique Author Name (mode, or sorted first if no mode)
    - Author Name (from ID): all unique variations, concatenated
    - Affiliation: all unique variations, concatenated

    Returns dataframe with columns:
    [Author ID, Author Name, Author Name (from ID), Affiliation]
    """
    if df.empty:
        return pd.DataFrame(columns=["Author ID", "Author Name", "Author Name (from ID)", "Affiliation"])
    def concat_uniques(x):
        return "; ".join(sorted(set(xx.strip() for xx in x if pd.notna(xx) and str(xx).strip() != "")))
    author_ref = (
        df.groupby("Author ID")
        .agg({
            "Author Name": lambda x: pd.Series.mode(x)[0] if not pd.Series(x).mode().empty else sorted(set(x))[0],
            "Author Name (from ID)": concat_uniques,
            "Affiliation": concat_uniques,
        })
        .reset_index()
    )
    return author_ref

def quadrant_plot_total_vs_slope(table):
    st.subheader("Quadrant Analysis")
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        total_cut = st.number_input("Total publication threshold (x-axis cut)", min_value=0, max_value=int(table['Total'].max()), value=int(table['Total'].median()), step=1)
    with col2:
        slope_cut = st.number_input("Slope threshold (y-axis cut)", min_value=float(table['Slope'].min()), max_value=float(table['Slope'].max()), value=0.0, step=0.1)

    # Assign quadrant label
    def quad(row):
        if row["Total"] >= total_cut and row["Slope"] >= slope_cut:
            return "(Q1) Established & Emerging"
        elif row["Total"] < total_cut and row["Slope"] >= slope_cut:
            return "(Q2) Emerging"
        elif row["Total"] < total_cut and row["Slope"] < slope_cut:
            return "(Q3) Marginal/Other"
        else:
            return "(Q4) Established but stable/declining"
    table["Quadrant"] = table.apply(quad, axis=1)

    fig = px.scatter(
        table, x="Total", y="Slope", text="ASJC", color="Quadrant",
        labels={"Total": "Total Publications", "Slope": "Annual Growth Slope"},
        title="Quadrant Analysis: Total vs. Slope"
    )
    fig.update_traces(textposition='bottom center')
    
    # Add threshold lines
    fig.add_vline(x=total_cut, line_dash="dash", line_color="gray")
    fig.add_hline(y=slope_cut, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(table)

def burst_detection_asjc(
    df_summary,
    recent_years=3,
    gamma=1.0,
    smooth_win=1,
    burst_level_option="All nonzero burst levels",
    min_years=3,
    min_papers=2,
):
    """
    Run burst detection for each ASJC in df_summary.
    Flags ASJC as 'Emerging' if a burst is detected in the recent period.
    Allows user to set burst parameters and min data filters.
    """
    years = sorted(df_summary["Year"].unique())
    recent_window = years[-recent_years:]

    results = []
    for asjc, group in df_summary.groupby("ASJC"):
        # Filter out fields with too little data
        group = group.set_index("Year").reindex(years, fill_value=0)
        counts = group["Unique_Paper_Count"].values
        total_pubs = counts.sum()
        n_years = np.count_nonzero(counts)

        if n_years < min_years or total_pubs < min_papers:
            results.append({
                "ASJC": asjc,
                "Burst_in_recent": False,
                "Burst_years": [],
                "Details": f"Not enough data (years={n_years}, pubs={total_pubs})"
            })
            continue

        # Prepare events array (year repeated by count)
        events = []
        for y, c in zip(years, counts):
            events.extend([y] * c)
        n = len(events)
        s = [years[0], years[-1]]

        if n < 2:
            results.append({
                "ASJC": asjc,
                "Burst_in_recent": False,
                "Burst_years": [],
                "Details": "Not enough events"
            })
            continue

        # Run burst detection
        q, d, r = burst_detection(events, n, s, gamma, smooth_win)

        # Decide which bursts to use
        burst_years = []
        if burst_level_option == "Only highest burst level":
            if r:
                max_level = max(level for _, _, level in r)
                for start, end, level in r:
                    if level == max_level and level > 0:
                        burst_years.extend(range(start, end + 1))
        else:  # All nonzero burst levels
            for start, end, level in r:
                if level > 0:
                    burst_years.extend(range(start, end + 1))

        burst_years = set(burst_years)
        burst_in_recent = any(y in recent_window for y in burst_years)
        results.append({
            "ASJC": asjc,
            "Burst_in_recent": burst_in_recent,
            "Burst_years": sorted(burst_years),
            "Details": ""
        })

    burst_df = pd.DataFrame(results)
    return burst_df

# ========================
#     DATA LOADING
# ========================

@st.cache_data
def read_scopus_excel(file):
    """Read Scopus Source Excel and return two DataFrames: sources and ASJC codes."""
    wanted_cols = [
        "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
        "Active or Inactive", "Source Type", "Publisher",
        "Publisher Imprints Grouped to Main Publisher",
        "All Science Journal Classification Codes (ASJC)",
    ]
    excel_file = pd.ExcelFile(file)
    df_source_full = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])
    cols_present = [col for col in wanted_cols if col in df_source_full.columns]
    df_source = df_source_full[cols_present]
    df_source["ISSN"] = df_source["ISSN"].apply(clean_issn)
    df_source["EISSN"] = df_source["EISSN"].apply(clean_issn)
    # ASJC code list
    asjc_df = pd.read_excel(
        excel_file,
        sheet_name=excel_file.sheet_names[-1],
        usecols=[0, 1],
        skiprows=8,
        nrows=362
    )
    asjc_cleaned = asjc_df.dropna(subset=["Code"]).copy()
    asjc_cleaned["Code"] = asjc_cleaned["Code"].astype(int)
    return df_source, asjc_cleaned

@st.cache_data
def read_and_merge_scopus_csv(files):
    """Read multiple Scopus export CSVs and merge into one DataFrame. Checks for consistent columns."""
    if not files:
        return None, "No files uploaded."
    if len(files) > 10:
        return None, "Please upload no more than 10 CSV files at once."
    dataframes = []
    base_columns = None
    for i, file in enumerate(files):
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return None, f"Error reading file '{file.name}': {str(e)}"
        if base_columns is None:
            base_columns = df.columns.tolist()
        else:
            if df.columns.tolist() != base_columns:
                return None, (
                    f"Column mismatch detected!\n"
                    f"File '{file.name}' has columns:\n{df.columns.tolist()}\n"
                    f"First file has columns:\n{base_columns}\n"
                    "Please ensure all CSVs have the same columns in the same order."
                )
        dataframes.append(df)
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df, None

# ===========================
#     DATA MAPPING/PREP
# ===========================

def add_asjc_to_export_csv(df_export, df_source, df_asjc):
    """
    Map ISSN/EISSN in export CSV to ASJC codes & descriptions using Scopus source & ASJC masterlist.
    Adds 'Matched_ASJC' (list) and 'Matched_ASJC_Description' (list) columns to df_export.
    """
    # Clean ISSN/EISSN in both source and export for matching
    df_source = df_source.copy()
    df_source["ISSN_clean"] = df_source["ISSN"].apply(clean_issn)
    df_source["EISSN_clean"] = df_source["EISSN"].apply(clean_issn)
    df_export = df_export.copy()
    df_export["ISSN"] = df_export["ISSN"].apply(clean_issn)
    if "EISSN" in df_export.columns:
        df_export["EISSN"] = df_export["EISSN"].apply(clean_issn)
    else:
        df_export["EISSN"] = None

    issn_map = df_source.set_index('ISSN_clean')["All Science Journal Classification Codes (ASJC)"].to_dict()
    eissn_map = df_source.set_index('EISSN_clean')["All Science Journal Classification Codes (ASJC)"].to_dict()
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))

    def get_asjc_codes(row):
        issn, eissn = row["ISSN"], row["EISSN"]
        codes = None
        if issn and issn in issn_map and pd.notna(issn_map[issn]):
            codes = issn_map[issn]
        elif eissn and eissn in eissn_map and pd.notna(eissn_map[eissn]):
            codes = eissn_map[eissn]
        return parse_asjc_list(codes) if codes else []
    df_export["Matched_ASJC"] = df_export.apply(get_asjc_codes, axis=1)
    df_export["Matched_ASJC_Description"] = df_export["Matched_ASJC"].apply(
        lambda codes: [asjc_dict.get(code, str(code)) for code in codes]
    )
    return df_export

# ===========================
#   AUTHOR DATAFRAME BUILD
# ===========================

def build_author_df(df_export_with_asjc):
    """
    Build a DataFrame with one row per author, paper, ASJC, and author type.
    For use with Scopus export CSV with mapped ASJC columns.
    """
    author_rows = []
    df_expanded = df_export_with_asjc.copy()
    # If needed, convert ASJC description string to list
    if "Matched_ASJC_Description" in df_expanded.columns and isinstance(df_expanded["Matched_ASJC_Description"].iloc[0], str) and "[" in df_expanded["Matched_ASJC_Description"].iloc[0]:
        import ast
        df_expanded["Matched_ASJC_Description"] = df_expanded["Matched_ASJC_Description"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df_expanded = df_expanded.explode("Matched_ASJC_Description")
    for idx, row in df_expanded.iterrows():
        names = [x.strip() for x in str(row.get("Authors", "")).split(";")]
        ids_full = [x.strip() for x in str(row.get("Author full names", "")).split(";")]
        authors_with_affil = [x.strip() for x in str(row.get("Authors with affiliations", "")).split(";")]
        correspondence_address = str(row.get("Correspondence Address", ""))
        asjc = row.get("Matched_ASJC_Description", None)
        corresponding_names_raw = correspondence_address.split(";", 1)[0]
        corresponding_names = [x.strip() for x in corresponding_names_raw.split(";") if x.strip()]
        n = min(len(names), len(ids_full), len(authors_with_affil))
        for i in range(n):
            name = names[i]
            id_full = ids_full[i]
            author_id = extract_id(id_full)
            name_variant = extract_name(id_full)
            split_affil = authors_with_affil[i].split(",", 1)
            affiliation = split_affil[1].strip() if len(split_affil) > 1 else ""
            author_type = "First Author" if i == 0 else "Co-author"
            variants = author_name_variants(name)
            if any(v in corresponding_names for v in variants):
                author_type = "Corresponding Author"
            author_rows.append({
                "Author ID": author_id,
                "Author Name": name,
                "Author Name (from ID)": name_variant,
                "Affiliation": affiliation,
                "ASJC": asjc,
                "Author Type": author_type,
                "EID": row.get("EID", None)
            })
            
    df_authors = pd.DataFrame(author_rows)
    
    # --- CANONICALIZE Author fields ---
    if not df_authors.empty:
        author_ref = get_author_canonical_info(df_authors)
        df_authors = df_authors.drop(columns=["Author Name", "Author Name (from ID)", "Affiliation"], errors="ignore")
        df_authors = df_authors.merge(author_ref, on="Author ID", how="left")
    return df_authors

def build_author_df_w_year(df_export_with_asjc):
    """
    Build a DataFrame with one row per author, paper, ASJC, year, and author type.
    For use with Scopus export CSV with mapped ASJC columns.
    """
    author_rows = []
    df_expanded = df_export_with_asjc.copy()
    # If needed, convert ASJC description string to list
    if "Matched_ASJC_Description" in df_expanded.columns and isinstance(df_expanded["Matched_ASJC_Description"].iloc[0], str) and "[" in df_expanded["Matched_ASJC_Description"].iloc[0]:
        import ast
        df_expanded["Matched_ASJC_Description"] = df_expanded["Matched_ASJC_Description"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df_expanded = df_expanded.explode("Matched_ASJC_Description")
    for idx, row in df_expanded.iterrows():
        names = [x.strip() for x in str(row.get("Authors", "")).split(";")]
        ids_full = [x.strip() for x in str(row.get("Author full names", "")).split(";")]
        authors_with_affil = [x.strip() for x in str(row.get("Authors with affiliations", "")).split(";")]
        correspondence_address = str(row.get("Correspondence Address", ""))
        asjc = row.get("Matched_ASJC_Description", None)
        year = row.get("Year", None)

        # Add more information here !!!
        title = row.get("Title", None)
        source = row.get("Source title", None)
        cited_by = row.get("Cited by", None)
        
        corresponding_names_raw = correspondence_address.split(";", 1)[0]
        corresponding_names = [x.strip() for x in corresponding_names_raw.split(";") if x.strip()]
        n = min(len(names), len(ids_full), len(authors_with_affil))
        for i in range(n):
            name = names[i]
            id_full = ids_full[i]
            author_id = extract_id(id_full)
            name_variant = extract_name(id_full)
            split_affil = authors_with_affil[i].split(",", 1)
            affiliation = split_affil[1].strip() if len(split_affil) > 1 else ""
            author_type = "First Author" if i == 0 else "Co-author"
            variants = author_name_variants(name)
            if any(v in corresponding_names for v in variants):
                author_type = "Corresponding Author"
            author_rows.append({
                "Author ID": author_id,
                "Author Name": name,
                "Author Name (from ID)": name_variant,
                "Affiliation": affiliation,
                "ASJC": asjc,
                "Author Type": author_type,
                "EID": row.get("EID", None),
                "Year": year,
                "Title": title,
                "Source": source,
                "Cited by": cited_by
            })
    df_authors = pd.DataFrame(author_rows)
    
    # --- CANONICALIZE Author fields as in detailed table ---
    if not df_authors.empty:
        author_ref = get_author_canonical_info(df_authors)
        # Remove current possibly non-canonical values and merge canonical ones
        df_authors = df_authors.drop(columns=["Author Name", "Author Name (from ID)", "Affiliation"], errors="ignore")
        df_authors = df_authors.merge(author_ref, on="Author ID", how="left")
    desired_order = ["EID", "Title", "Source", "Cited by", "Year", "ASJC", "Author Type", "Author ID", "Author Name", "Author Name (from ID)", "Affiliation"]
    df_authors = df_authors[desired_order]
    return df_authors

# ===========================
#       UI SECTIONS
# ===========================

def section_journal_filter(df_source, df_asjc):
    """UI for journal filtering by ASJC codes."""
    st.header("Journal Filter Section")
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))
    all_asjc_codes = list(df_asjc["Code"])
    select_all = st.checkbox("Select All ASJC Categories", key="select_all_journal")
    selected = st.multiselect(
        "Select ASJC Categories",
        options=all_asjc_codes,
        default=all_asjc_codes if select_all else [],
        format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
        key="journal_filter_asjc"
    )
    filter_now = st.button("Filter Journals", key="journal_filter_btn")
    if filter_now:
        if not selected:
            st.warning("Please select at least one ASJC category before filtering.")
        else:
            df_filtered = filter_and_collect_matches_with_desc(df_source, selected, asjc_dict)
            st.write(f"Journals matching selected ASJC categories ({len(df_filtered)}):")
            st.dataframe(df_filtered)
    else:
        st.info("Select one or more ASJC categories, then click 'Filter Journals'.")

def filter_and_collect_matches_with_desc(df_source, selected_codes, asjc_dict):
    """
    Filter journals in df_source for matching ASJC codes, returns a displayable DataFrame.
    """
    col = "All Science Journal Classification Codes (ASJC)"
    df = df_source.copy()
    df[col] = df[col].astype(str).str.replace(" ", "").str.replace(",", ";").replace("nan", "")
    df["ASJC_list"] = df[col].apply(parse_asjc_list)
    df["Matched_ASJC"] = df["ASJC_list"].apply(lambda codes: [code for code in codes if code in selected_codes])
    df["Matched_ASJC_Description"] = df["Matched_ASJC"].apply(lambda codes: [asjc_dict.get(code, str(code)) for code in codes])
    df_filtered = df[df["Matched_ASJC"].apply(lambda x: len(x) > 0)].copy()
    display_cols = [
        "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
        "Active or Inactive", "Source Type", "Publisher",
        "Publisher Imprints Grouped to Main Publisher",
        "Matched_ASJC", "Matched_ASJC_Description"
    ]
    return df_filtered[display_cols]

def section_map_export_csv(df_export_with_asjc, df_asjc):
    """UI to display mapped export CSV with ASJC filter."""
    st.header("Map Export CSV to Scopus Source & ASJC")
    all_codes = sorted(set(code for codes in df_export_with_asjc["Matched_ASJC"] for code in codes))
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))
    select_all_csv = st.checkbox("Select All ASJC Categories", key="select_all_csv")
    selected = st.multiselect(
        "Filter by ASJC Categories",
        options=all_codes,
        default=all_codes if select_all_csv else [],
        format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
        key="csv_asjc"
    )
    df_show = df_export_with_asjc.copy()
    if selected:
        df_show = df_show[df_show["Matched_ASJC"].apply(lambda codes: any(code in selected for code in codes))]
    st.dataframe(df_show)

def section_author_asjc_summary(author_df):
    """
    Author summary table (by author ID) and detailed table (by ID/ASJC/type) with download.
    Uses canonicalization: 
        - Author Name: most frequent value,
        - Author Name (from ID): all unique variants concatenated,
        - Affiliation: all unique variants concatenated.
    """
    st.subheader("Author Analysis Summary (with robust Corresponding Author detection)")

    # Canonical reference table for each author ID
    author_ref = get_author_canonical_info(author_df)
    # Replace columns in author_df with canonicalized ones
    author_df = author_df.drop(columns=["Author Name", "Author Name (from ID)", "Affiliation"], errors='ignore')
    author_df = author_df.merge(author_ref, on="Author ID", how="left")

    # --- Summary Table ---
    author_info = (
        author_df.groupby("Author ID")
        .agg({
            "Author Name": "first",  # already canonical
            "Author Name (from ID)": "first",
            "Affiliation": "first",
            "ASJC": lambda x: "; ".join(sorted(set(str(xx) for xx in x if pd.notna(xx) and str(xx).strip() != ""))),
            "Author Type": lambda x: "; ".join(sorted(set(x))),
            "EID": lambda x: len(set(x))
        })
        .rename(columns={"EID": "Unique Paper Count"})
        .reset_index()
    )
    author_info = author_info[[
        "Author ID", "Author Name", "Author Name (from ID)",
        "Affiliation", "ASJC", "Author Type", "Unique Paper Count"
    ]]
    st.write("**Summary Table:** (All variations, grouped by Scopus Author ID)")
    gb = GridOptionsBuilder.from_dataframe(author_info)
    gb.configure_default_column(filterable=True, sortable=True)
    for col in author_info.columns:
        gb.configure_column(col, filter=True, editable=False)
    grid_options = gb.build()
    AgGrid(author_info, gridOptions=grid_options, enable_enterprise_modules=True, allow_unsafe_jscode=True, fit_columns_on_grid_load=True, key="author_summary_grid")
    st.download_button("Download Author Summary Table as CSV", data=author_info.to_csv(index=False), file_name="author_summary_by_id.csv")

    # --- Detailed Table ---
    summary = (
        author_df.groupby(["Author ID", "ASJC", "Author Type"])
        .agg({
            "EID": lambda x: len(set(x)),
            "Author Name": "first",
            "Author Name (from ID)": "first",
            "Affiliation": "first"
        })
        .reset_index()
        .sort_values(["Author ID", "ASJC"])
        .rename(columns={"EID": "Unique Paper Count"})
    )
    summary = summary.merge(
        author_info[["Author ID", "Author Name", "Author Name (from ID)"]],
        on="Author ID", how="left", suffixes=("", "_Summary")
    )
    summary = summary[[
        "Author ID", "Author Name", "Author Name (from ID)",
        "Affiliation", "ASJC", "Author Type", "Unique Paper Count"
    ]]
    st.write("**Detailed Table:** (Each Author-ASJC-Type combination, canonicalized names/affiliations)")
    gb = GridOptionsBuilder.from_dataframe(summary)
    gb.configure_default_column(filterable=True, sortable=True)
    for col in summary.columns:
        gb.configure_column(col, filter=True, editable=False)
    grid_options = gb.build()
    AgGrid(summary, gridOptions=grid_options, enable_enterprise_modules=True, allow_unsafe_jscode=True, fit_columns_on_grid_load=True, key="author_detailed_grid")
    st.download_button("Download Detailed Author-ASJC-Type Table as CSV", data=summary.to_csv(index=False), file_name="author_asjc_type_summary.csv")
    return author_df

def section_author_dashboard(filtered_author_df):
    """Single-author dashboard with top ASJC categories bar chart (for selected author and type)."""
    st.header("Author Dashboard")
    if filtered_author_df.empty:
        st.info("No data for the selected author and type(s).")
        return
    top_asjc = (
        filtered_author_df.groupby("ASJC")
        .size()
        .reset_index(name="Paper Count")
        .sort_values("Paper Count", ascending=False)
        .head(10)
    )
    st.subheader("Top 10 ASJC Categories (for this author, by author type selection)")
    fig = px.bar(top_asjc, x="ASJC", y="Paper Count", title="Top 10 ASJC Categories for Selected Author")
    st.plotly_chart(fig, use_container_width=True)


def section_show_author_df_from_source(df_export_with_asjc, selected_author_id, selected_types):
    """Show author-paper-ASJC table and summary, filtered by selected author and type."""
    st.subheader("Author-Paper-ASJC Table (from Source)")
    df_authors = build_author_df_w_year(df_export_with_asjc)
    df_authors = df_authors[df_authors["Author ID"] == selected_author_id]
    df_authors = df_authors[df_authors["Author Type"].isin(selected_types)]
    st.write("Table below shows one row per author, per paper, per ASJC, per author type:")
    st.dataframe(df_authors, use_container_width=True)

    # Group and summarize
    st.subheader("Summary: Unique Paper Count by Year, ASJC, and Author Type")
    if not df_authors.empty:
        summary = (
            df_authors
            .groupby(["ASJC", "Author Type", "Year"])
            .agg(Unique_Paper_Count=("EID", lambda x: len(pd.unique(x))))
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)
        # Optionally add download button:
        st.download_button(
            "Download Summary as CSV",
            data=summary.to_csv(index=False),
            file_name="author_paper_asjc_summary.csv"
        )
        return df_authors, summary
    else:
        st.info("No author data to summarize.")
        return df_authors, pd.DataFrame()

def section_emerging_established_fields(df_summary, asjc_name_map=None, n_recent_years=3, max_year=None):
    """
    Displays a classification table for each ASJC: Total, Recent, Previous, Growth, Classification.
    df_summary: DataFrame with columns ['Year', 'ASJC', 'Unique_Paper_Count']
    asjc_name_map: dict mapping ASJC code to name (optional, else use ASJC code)
    n_recent_years: int, how many years to consider as 'recent'
    max_year: int, last year to include (optional; if None, uses latest in data)
    """

    if df_summary.empty:
        st.info("No data available.")
        return

    df_summary = df_summary.copy()
    df_summary["Year"] = df_summary["Year"].astype(int)
    
    if max_year is not None:
        df_summary = df_summary[df_summary["Year"] <= max_year]
    years = sorted(df_summary["Year"].unique())
    if len(years) <= n_recent_years:
        st.warning("Not enough years to classify emerging/established fields.")
        return

    recent_years = years[-n_recent_years:]
    pre_years = years[:-n_recent_years]

    recent_label = f"Recent {n_recent_years}y ({min(recent_years)}–{max(recent_years)})"
    pre_label = f"Pre-{n_recent_years}y (<{min(recent_years)})"

    st.markdown(f"""
    - **{recent_label}**: Total papers in {', '.join(str(y) for y in recent_years)}
    - **{pre_label}**: Total papers before {min(recent_years)}
    - **Max year considered:** {max_year if max_year else max(years)}
    """)

    recent = df_summary[df_summary["Year"].isin(recent_years)].groupby("ASJC")["Unique_Paper_Count"].sum()
    pre = df_summary[df_summary["Year"].isin(pre_years)].groupby("ASJC")["Unique_Paper_Count"].sum()
    total = df_summary.groupby("ASJC")["Unique_Paper_Count"].sum()

    table = pd.DataFrame({
        "Total": total,
        recent_label: recent,
        pre_label: pre
    }).fillna(0).astype(int)

    table["Growth"] = np.where(
        table[pre_label] == 0,
        np.where(table[recent_label] > 0, 1.0, 0.0),
        (table[recent_label] - table[pre_label]) / table[pre_label]
    )
    table["Growth"] = (table["Growth"] * 100).round().astype(int).astype(str) + "%"

    def classify(row):
        pre = row[pre_label]
        recent = row[recent_label]
        growth = int(row["Growth"].replace("%", ""))
        if pre <= 2 and recent >= 3 and growth >= 30:
            return "Emerging"
        elif pre >= 3 and recent >= 3 and growth > -30:
            return "Established"
        elif pre > 0 and recent == 0:
            return "Declining"
        else:
            return ""

    table["Classification"] = table.apply(classify, axis=1)
    table = table.reset_index()
    if asjc_name_map:
        table["ASJC"] = table["ASJC"].map(asjc_name_map).fillna(table["ASJC"])
    table = table[["ASJC", "Total", recent_label, pre_label, "Growth", "Classification"]]

    st.dataframe(table, use_container_width=True)
    return table
    
def section_trend_slope_classification(
    df_summary, asjc_name_map=None, min_total_default=5, last_year_default=None, slope_threshold_default=0.5
):
    """
    Interactive classification of fields by trend slope.
    User can set minimum total output, Last Year considered, and Slope threshold.
    """
    st.subheader("Trend Slope Classification Table")

    if df_summary.empty:
        st.info("No data available.")
        return

    all_years = sorted(df_summary["Year"].unique())
    min_year, max_year = min(all_years), max(all_years)
    if last_year_default is None:
        last_year_default = max_year

    # --- UI controls: for min_total, last_year, and slope threshold
    col1, col2, col3 = st.columns(3)
    with col1:
        min_total = st.number_input(
            "Minimum total publications to consider",
            min_value=1, max_value=100, value=min_total_default, step=1, key="min_total_slope"
        )
    with col2:
        last_year = st.number_input(
            "Last year to include",
            min_value=min_year, max_value=max_year, value=last_year_default, step=1, key="last_year_slope"
        )
    with col3:
        slope_threshold = st.number_input(
            "Slope threshold for 'Emerging' (per year)",
            min_value=-5.0, max_value=5.0, value=slope_threshold_default, step=0.1, key="slope_threshold"
        )

    # Filter data
    filtered = df_summary[df_summary["Year"] <= last_year].copy()
    if filtered.empty:
        st.info("No data in selected year range.")
        return

    # --- Slope analysis per ASJC
    records = []
    for asjc, group in filtered.groupby("ASJC"):
        group = group.sort_values("Year")
        years = group["Year"].values
        counts = group["Unique_Paper_Count"].values
        total = counts.sum()
        if len(years) > 1:
            slope, _, r_value, p_value, _ = linregress(years, counts)
        else:
            slope, r_value, p_value = 0, 0, 1
        records.append({
            "ASJC": asjc,
            "Total": int(total),
            "First_Year": int(years[0]),
            "Last_Year": int(years[-1]),
            "Slope": round(slope, 2),
            "R": round(r_value, 2),
            "p-value": round(p_value, 3)
        })

    table = pd.DataFrame(records)

    # --- Classification logic
    def classify(row):
        if row["Total"] >= min_total and row["Slope"] > slope_threshold:
            return "Emerging"
        elif row["Total"] >= min_total and -slope_threshold <= row["Slope"] <= slope_threshold:
            return "Established"
        elif row["Total"] >= min_total and row["Slope"] < -slope_threshold:
            return "Declining"
        else:
            return "Other"

    table["Classification"] = table.apply(classify, axis=1)

    if asjc_name_map:
        table["ASJC"] = table["ASJC"].map(asjc_name_map).fillna(table["ASJC"])

    table = table[["ASJC", "Total", "First_Year", "Last_Year", "Slope", "R", "p-value", "Classification"]]
    st.dataframe(table, use_container_width=True)
    
    quadrant_plot_total_vs_slope(table)
    
    return table

def section_burst_detection_asjc(
    df_summary,
):
    st.subheader("Kleinberg’s Burst Detection")
    gamma = st.number_input("Burst penalty (`gamma`)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    smooth_win = st.number_input("Smoothing window (`smooth_win`)", min_value=1, max_value=10, value=1, step=1)
    burst_level_option = st.radio(
        "Which bursts to flag as 'emerging'?",
        options=["All nonzero burst levels", "Only highest burst level"],
        index=0
    )
    min_years = st.number_input("Minimum years of data per ASJC", min_value=2, max_value=10, value=3, step=1)
    min_papers = st.number_input("Minimum total papers per ASJC", min_value=1, max_value=50, value=2, step=1)
    recent_years = st.number_input(
        "How many recent years to consider?",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )
    
    burst_df = burst_detection_asjc(
        df_summary,
        recent_years=recent_years,
        gamma=gamma,
        smooth_win=1,
        burst_level_option=burst_level_option,
        min_years=min_years,
        min_papers=min_papers,
    )
    st.dataframe(burst_df, use_container_width=True)
    st.info(
        "A field is flagged as 'Emerging' if a burst is detected in the most recent period. "
        "Tune the parameters above for sensitivity."
    )
    return burst_df

# ===========================
#         MAIN APP
# ===========================

def main():
    st.set_page_config(
        page_title="Scopus Analysis Toolkit",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Scopus Analysis Toolkit")
    st.sidebar.header("1. Upload Scopus Source Excel")
    excel_file = st.sidebar.file_uploader("Upload Scopus Source Excel", type=["xlsx"], key="excel_upload")
    df_source, df_asjc = None, None
    df_export, df_export_with_asjc, csv_error = None, None, None

    if excel_file:
        df_source, df_asjc = read_scopus_excel(excel_file)
        st.sidebar.success("Excel file loaded. Please proceed to upload CSV file(s).")
        st.sidebar.header("2. Upload Scopus Export CSV(s)")
        csv_files = st.sidebar.file_uploader(
            "Upload up to 10 Scopus Export CSV files",
            type=["csv"], accept_multiple_files=True, key="csv_upload"
        )
        if csv_files:
            df_export, csv_error = read_and_merge_scopus_csv(csv_files)
            if csv_error:
                st.sidebar.error(csv_error)
            else:
                st.sidebar.success(f"Successfully merged {len(csv_files)} CSV files ({len(df_export)} rows).")
                df_export_with_asjc = add_asjc_to_export_csv(df_export, df_source, df_asjc)
    else:
        st.sidebar.info("Please upload the Scopus Source Excel first.")

    tabs = st.tabs(["Journal Filter", "Map Export CSV", "Author Analysis"])
    with tabs[0]:
        if df_source is not None and df_asjc is not None:
            section_journal_filter(df_source, df_asjc)
        else:
            st.info("Please upload the Scopus Source Excel to use this section.")
    with tabs[1]:
        if df_export_with_asjc is not None and df_asjc is not None:
            section_map_export_csv(df_export_with_asjc, df_asjc)
        else:
            st.info("Please upload both the Scopus Source Excel and Export CSV(s) to use this section.")
    with tabs[2]:
        if df_export_with_asjc is not None:
            # Build author table (canonicalized)
            author_df_full = build_author_df(df_export_with_asjc)
            author_ref = get_author_canonical_info(author_df_full)
            author_ref["Selector"] = author_ref["Author ID"] + " | " + author_ref["Author Name (from ID)"]
    
            # Select author
            selected = st.selectbox("Select an Author", options=author_ref["Selector"].tolist(), index=0)
            selected_author_id = selected.split(" | ")[0]
    
            # Filter for dashboard, detailed table, and summary
            df_author = author_df_full[author_df_full["Author ID"] == selected_author_id]
            author_types = sorted(df_author["Author Type"].dropna().unique())
            selected_types = st.multiselect(
                "Filter by Author Type",
                options=author_types,
                default=author_types,
                key="source_author_type"
            )
    
            # Final filtered dataframe for display in all sections
            filtered_author_df = df_author[df_author["Author Type"].isin(selected_types)]
    
            # Pass to all dashboard sections
            section_author_dashboard(filtered_author_df)
            df_authors_table, df_summary = section_show_author_df_from_source(
                df_export_with_asjc, selected_author_id, selected_types
            )
            section_author_asjc_summary(filtered_author_df)
            
            years = sorted(df_summary["Year"].unique())
            default_max_year = max(years)
            
            st.subheader("Summary Table with Classification")
            max_year = st.number_input(
                "Max year to consider:",
                min_value=min(years), max_value=max(years), value=default_max_year, step=1
            )
            n_recent_years = st.number_input(
                "Number of recent years:", min_value=2, max_value=10, value=3, step=1
            )
            section_emerging_established_fields(
                df_summary, asjc_name_map=None,
                n_recent_years=n_recent_years, max_year=max_year
            )
            section_trend_slope_classification(
                df_summary, asjc_name_map=None, min_total_default=5, last_year_default=None
            )
            section_burst_detection_asjc(
                df_summary,
            )
            
        else:
            st.info("Please upload both the Scopus Source Excel and Export CSV(s) to use this section.")

if __name__ == "__main__":
    main()
