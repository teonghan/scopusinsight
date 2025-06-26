import streamlit as st
import pandas as pd
import plotly.express as px
import re
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Scopus Analysis Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def clean_issn(val):
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
    m = re.search(r'\((\d+)\)$', id_str)
    return m.group(1) if m else id_str.strip()

def extract_name(id_str):
    m = re.match(r'(.+)\s\(\d+\)$', id_str.strip())
    return m.group(1) if m else id_str.strip()

def author_name_variants(surname_first):
    parts = surname_first.split()
    if len(parts) < 2:
        return {surname_first}
    surname = parts[0]
    initials = " ".join(parts[1:])
    return {surname_first, f"{initials} {surname}".strip()}

def unique_concatenate(x):
    seen = set()
    result = []
    for item in x:
        cleaned = item.lower().strip()
        if cleaned not in seen and str(item).strip() != "":
            seen.add(cleaned)
            result.append(item.strip())
    return "; ".join(result)

# --- Data Loading ---
@st.cache_data
def read_scopus_excel(file):
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

def add_asjc_to_export_csv(df_export, df_source, df_asjc):
    # Clean ISSN/EISSN in source
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
    def get_asjc_codes(row):
        issn = row["ISSN"]
        eissn = row["EISSN"]
        codes = None
        if issn and issn in issn_map and pd.notna(issn_map[issn]):
            codes = issn_map[issn]
        elif eissn and eissn in eissn_map and pd.notna(eissn_map[eissn]):
            codes = eissn_map[eissn]
        if codes:
            codes_list = [int(code) for code in str(codes).replace(" ", "").replace(",", ";").split(";") if code.isdigit()]
            return codes_list
        else:
            return []
    df_export["Matched_ASJC"] = df_export.apply(get_asjc_codes, axis=1)
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))
    df_export["Matched_ASJC_Description"] = df_export["Matched_ASJC"].apply(
        lambda codes: [asjc_dict.get(code, str(code)) for code in codes]
    )
    return df_export

# --- Author Table Building ---
def build_author_df(df_export_with_asjc):
    author_rows = []
    df_expanded = df_export_with_asjc.copy()
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
    return pd.DataFrame(author_rows)
    
# --- Author Table Building (NEW) ---
def build_author_df_from_source(df_source):
    author_rows = []
    # Defensive: Some columns may be missing, handle gracefully
    df = df_source.copy()
    df = df.reset_index(drop=True)
    
    # Explode if there are multiple ASJC codes per paper
    if "All Science Journal Classification Codes (ASJC)" in df.columns:
        df["ASJC_list"] = (
            df["All Science Journal Classification Codes (ASJC)"]
            .astype(str)
            .str.replace(" ", "")
            .str.replace(",", ";")
            .replace("nan", "")
            .apply(lambda x: [int(code) for code in x.split(";") if code.isdigit()])
        )
        df = df.explode("ASJC_list").rename(columns={"ASJC_list": "ASJC"})
    else:
        df["ASJC"] = None

    for idx, row in df.iterrows():
        # Parse authors (short), author full names (with IDs), authors with affiliations
        names = [x.strip() for x in str(row.get("Authors", "")).split(";")]
        ids_full = [x.strip() for x in str(row.get("Author full names", "")).split(";")]
        authors_with_affil = [x.strip() for x in str(row.get("Authors with affiliations", "")).split(";")]
        correspondence_address = str(row.get("Correspondence Address", ""))
        asjc = row.get("ASJC", None)
        year = row.get("Year", None)
        eid = row.get("EID", None)

        # Corresponding Author (if desired, you can implement robust detection as before)
        corresponding_names_raw = correspondence_address.split(";", 1)[0]
        corresponding_names = [x.strip() for x in corresponding_names_raw.split(";") if x.strip()]

        n = min(len(names), len(ids_full), len(authors_with_affil))
        for i in range(n):
            name = names[i]
            id_full = ids_full[i]
            # You must define or import extract_id and extract_name from your utilities
            author_id = extract_id(id_full)
            name_variant = extract_name(id_full)
            split_affil = authors_with_affil[i].split(",", 1)
            affiliation = split_affil[1].strip() if len(split_affil) > 1 else ""
            author_type = "First Author" if i == 0 else "Co-author"
            # Add robust corresponding detection if needed:
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
                "Year": year,
                "EID": eid
            })

    return pd.DataFrame(author_rows)

# --- Journal Filter Section ---
def filter_and_collect_matches_with_desc(df_source, selected_codes, asjc_dict):
    col = "All Science Journal Classification Codes (ASJC)"
    df = df_source.copy()
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(" ", "")
        .str.replace(",", ";")
        .replace("nan", "")
    )
    df["ASJC_list"] = df[col].apply(lambda x: [int(code) for code in x.split(";") if code.isdigit()])
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

def section_journal_filter(df_source, df_asjc):
    st.header("Journal Filter Section")
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))
    all_asjc_codes = list(df_asjc["Code"])
    select_all = st.checkbox("Select All ASJC Categories", key="select_all_journal")
    if select_all:
        selected = st.multiselect(
            "Select ASJC Categories",
            options=all_asjc_codes,
            default=all_asjc_codes,
            format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
            key="journal_filter_asjc"
        )
    else:
        selected = st.multiselect(
            "Select ASJC Categories",
            options=all_asjc_codes,
            format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
            key="journal_filter_asjc"
        )
    filter_now = st.button("Filter Journals", key="journal_filter_btn")
    if filter_now:
        if not selected:
            st.warning("Please select at least one ASJC category before filtering.")
        else:
            filtered = filter_and_collect_matches_with_desc(df_source, selected, asjc_dict)
            st.write(f"Journals matching selected ASJC categories ({len(filtered)}):")
            st.dataframe(filtered)
            # Add charts as desired
    else:
        st.info("Select one or more ASJC categories, then click 'Filter Journals'.")

def section_map_export_csv(df_export_with_asjc, df_asjc):
    st.header("Map Export CSV to Scopus Source & ASJC")
    all_codes = sorted(set(code for codes in df_export_with_asjc["Matched_ASJC"] for code in codes))
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))
    select_all_csv = st.checkbox("Select All ASJC Categories", key="select_all_csv")
    if select_all_csv:
        selected = st.multiselect(
            "Filter by ASJC Categories",
            options=all_codes,
            default=all_codes,
            format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
            key="csv_asjc"
        )
    else:
        selected = st.multiselect(
            "Filter by ASJC Categories",
            options=all_codes,
            format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
            key="csv_asjc"
        )
    df_show = df_export_with_asjc.copy()
    if selected:
        df_show = df_show[df_show["Matched_ASJC"].apply(lambda codes: any(code in selected for code in codes))]
    st.dataframe(df_show)

# --- Author Summary and Detailed Table Section ---
def section_author_asjc_summary(author_df):
    st.header("Author Analysis Summary (with robust Corresponding Author detection)")

    # Build canonical reference
    author_ref = (
        author_df.groupby("Author ID")
        .agg({
            "Author Name": lambda x: pd.Series.mode(x)[0] if not pd.Series(x).mode().empty else sorted(set(x))[0],
            "Author Name (from ID)": lambda x: pd.Series.mode(x)[0] if not pd.Series(x).mode().empty else sorted(set(x))[0],
            "Affiliation": lambda x: pd.Series.mode(x)[0] if not pd.Series(x).mode().empty else sorted(set(x))[0]
        })
        .reset_index()
    )
    # Merge back
    author_df = author_df.drop(columns=["Author Name", "Author Name (from ID)", "Affiliation"], errors='ignore')
    author_df = author_df.merge(author_ref, on="Author ID", how="left")
    
    # --- Summary Table (grouped by Author ID, all variations, unique paper count) ---
    author_info = (
        author_df.groupby("Author ID")
        .agg({
            "Author Name": "first",
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

    # -------------------------------------------------------------------------
    # Assume author_info is your DataFrame, enable advanced filtering to the table
    # -------------------------------------------------------------------------
    gb = GridOptionsBuilder.from_dataframe(author_info)
    gb.configure_default_column(filterable=True, sortable=True)
    
    # (Optional) Force text columns to be filterable
    for col in author_info.columns:
        gb.configure_column(col, filter=True, editable=False)
    
    grid_options = gb.build()
    
    AgGrid(
        author_info,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True
    )
    
    st.download_button(
        "Download Author Summary Table as CSV",
        data=author_info.to_csv(index=False),
        file_name="author_summary_by_id.csv"
    )
    # --- Detailed Table (one row per Author-ASJC-Type, with unique paper count, with name variants) ---
    summary = (
        author_df.groupby(["Author ID", "ASJC", "Author Type"])
        .agg({
            "EID": lambda x: len(set(x)),
            "Author Name": unique_concatenate,
            "Author Name (from ID)": unique_concatenate,
            "Affiliation": unique_concatenate
        })
        .reset_index()
        .sort_values(["Author ID", "ASJC"])
        .rename(columns={"EID": "Unique Paper Count"})
    )
    # Merge main name variants from author_info for each Author ID
    summary = summary.merge(
        author_info[["Author ID", "Author Name", "Author Name (from ID)"]],
        on="Author ID",
        how="left",
        suffixes=("", "_Summary")
    )
    summary = summary[[
        "Author ID", "Author Name", "Author Name (from ID)",
        "Affiliation", "ASJC", "Author Type", "Unique Paper Count"
    ]]
    st.write("**Detailed Table:** (Each Author-ASJC-Type combination, with name variants)")

    # -------------------------------------------------------------------------
    # Assume summary is your DataFrame, enable advanced filtering to the table
    # -------------------------------------------------------------------------
    gb = GridOptionsBuilder.from_dataframe(summary)
    gb.configure_default_column(filterable=True, sortable=True)
    
    # (Optional) Force text columns to be filterable
    for col in summary.columns:
        gb.configure_column(col, filter=True, editable=False)
    
    grid_options = gb.build()
    
    AgGrid(
        summary,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True
    )

    st.download_button(
        "Download Detailed Author-ASJC-Type Table as CSV",
        data=summary.to_csv(index=False),
        file_name="author_asjc_type_summary.csv"
    )
    return author_df

# --- Author Dashboard ---
def section_author_dashboard(author_df):
    st.header("Author Dashboard")

    # Build canonical author_ref table inside this function
    author_ref = (
        author_df.groupby("Author ID")
        .agg({
            "Author Name": lambda x: pd.Series.mode(x)[0] if not pd.Series(x).mode().empty else sorted(set(x))[0],
            "Author Name (from ID)": lambda x: pd.Series.mode(x)[0] if not pd.Series(x).mode().empty else sorted(set(x))[0],
            "Affiliation": lambda x: pd.Series.mode(x)[0] if not pd.Series(x).mode().empty else sorted(set(x))[0]
        })
        .reset_index()
    )

    # Use only the canonicalized author_ref for the dropdown
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

# --- Main App ---
def main():
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
            type=["csv"],
            accept_multiple_files=True,
            key="csv_upload"
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
            author_df = build_author_df(df_export_with_asjc)
            section_author_asjc_summary(author_df)
            section_author_dashboard(author_df)
        else:
            st.info("Please upload both the Scopus Source Excel and Export CSV(s) to use this section.")

if __name__ == "__main__":
    main()
