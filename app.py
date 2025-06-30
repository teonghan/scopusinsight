import streamlit as st
import pandas as pd
import plotly.express as px
import re
from st_aggrid import AgGrid, GridOptionsBuilder

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
                "Year": year
            })
    df_authors = pd.DataFrame(author_rows)
    
    # --- CANONICALIZE Author fields as in detailed table ---
    if not df_authors.empty:
        author_ref = get_author_canonical_info(df_authors)
        # Remove current possibly non-canonical values and merge canonical ones
        df_authors = df_authors.drop(columns=["Author Name", "Author Name (from ID)", "Affiliation"], errors="ignore")
        df_authors = df_authors.merge(author_ref, on="Author ID", how="left")
    desired_order = ["Author ID", "Author Name", "Author Name (from ID)", "Affiliation", "Year", "ASJC", "Author Type", "EID"]
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
    st.header("Author Analysis Summary (with robust Corresponding Author detection)")

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
    AgGrid(author_info, gridOptions=grid_options, enable_enterprise_modules=True, allow_unsafe_jscode=True, fit_columns_on_grid_load=True)
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
    AgGrid(summary, gridOptions=grid_options, enable_enterprise_modules=True, allow_unsafe_jscode=True, fit_columns_on_grid_load=True)
    st.download_button("Download Detailed Author-ASJC-Type Table as CSV", data=summary.to_csv(index=False), file_name="author_asjc_type_summary.csv")
    return author_df

def section_author_dashboard(author_df):
    """Single-author dashboard with top ASJC categories bar chart (for author/type selection)."""
    st.header("Author Dashboard")

    # Use canonical author info for dropdown
    author_ref = get_author_canonical_info(author_df)
    author_ref["Selector"] = author_ref["Author ID"] + " | " + author_ref["Author Name (from ID)"]

    selected = st.selectbox(
        "Select an Author",
        options=author_ref["Selector"].tolist(),
        index=0
    )
    if selected:
        selected_id = selected.split(" | ")[0]
        df_author = author_df[author_df["Author ID"] == selected_id]
        author_types = sorted(df_author["Author Type"].dropna().unique())
        selected_types = st.multiselect(
            "Filter by Author Type",
            options=author_types,
            default=author_types
        )
        filtered = df_author[df_author["Author Type"].isin(selected_types)]
        top_asjc = (
            filtered.groupby("ASJC")
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
            .groupby(["Author ID", "Author Name", "Author Name (from ID)", "Affiliation", "Year", "ASJC", "Author Type"])
            .agg(Unique_Paper_Count=("EID", lambda x: len(pd.unique(x))))
            .reset_index()
            .sort_values(["Year", "ASJC", "Author Type"])
        )
        st.dataframe(summary, use_container_width=True)
        # Optionally add download button:
        st.download_button(
            "Download Summary as CSV",
            data=summary.to_csv(index=False),
            file_name="author_paper_asjc_summary.csv"
        )
    else:
        st.info("No author data to summarize.")

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
                default=author_types
            )
    
            # Final filtered dataframe for display in all sections
            filtered_author_df = df_author[df_author["Author Type"].isin(selected_types)]
    
            # Pass to all dashboard sections
            section_author_dashboard(filtered_author_df)
            section_show_author_df_from_source(df_export_with_asjc, selected_author_id, selected_types)
            section_author_asjc_summary(filtered_author_df)
        else:
            st.info("Please upload both the Scopus Source Excel and Export CSV(s) to use this section.")

if __name__ == "__main__":
    main()
