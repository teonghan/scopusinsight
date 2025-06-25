import streamlit as st
import pandas as pd
import plotly.express as px

# ===============================
# --- Data Loading and Helpers ---
# ===============================

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

    # Format ISSN/EISSN columns to XXXX-XXXX
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

def clean_issn(val):
    if pd.isna(val): return None
    val = str(val).replace("-", "").strip()
    if not val.isdigit() or len(val) < 7:
        return None
    val = val.zfill(8)
    return val[:4] + '-' + val[4:]

# ===============================
# --- Journal Filter Section ----
# ===============================

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

    select_all = st.checkbox("Select All ASJC Categories")
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

            # --- Pie Chart: Active vs Inactive Journals ---
            st.subheader("Journal Activity Status")
            if "Active or Inactive" in filtered.columns:
                status_counts = filtered["Active or Inactive"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig_status = px.pie(status_counts, names="Status", values="Count", title="Active vs Inactive Journals")
                st.plotly_chart(fig_status, use_container_width=True)

            # --- Stacked Bar Chart: Active/Inactive by Source Type ---
            st.subheader("Journal Activity Status by Source Type")
            if "Active or Inactive" in filtered.columns and "Source Type" in filtered.columns:
                type_status_counts = (
                    filtered
                    .groupby(['Source Type', 'Active or Inactive'])
                    .size()
                    .reset_index(name='Count')
                )
                fig_stack = px.bar(
                    type_status_counts,
                    x="Source Type",
                    y="Count",
                    color="Active or Inactive",
                    title="Active/Inactive Journals by Source Type",
                    barmode="stack"
                )
                st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Select one or more ASJC categories, then click 'Filter Journals'.")

# ==========================
# --- Map Export CSV ------
# ==========================
def add_asjc_to_export_csv(df_export, df_source, df_asjc):
    # Clean ISSN/EISSN in source
    df_source = df_source.copy()
    df_source["ISSN_clean"] = df_source["ISSN"].apply(clean_issn)
    df_source["EISSN_clean"] = df_source["EISSN"].apply(clean_issn)

    # Clean ISSN in CSV
    df_export = df_export.copy()
    df_export["ISSN"] = df_export["ISSN"].apply(clean_issn)

    # Build maps
    issn_map = df_source.set_index('ISSN_clean')["All Science Journal Classification Codes (ASJC)"].to_dict()
    eissn_map = df_source.set_index('EISSN_clean')["All Science Journal Classification Codes (ASJC)"].to_dict()

    # For each CSV ISSN: try ISSN match first, then EISSN match
    def get_asjc_codes(row):
        issn = row["ISSN"]
        codes = None
        if issn in issn_map and pd.notna(issn_map[issn]):
            codes = issn_map[issn]
        elif issn in eissn_map and pd.notna(eissn_map[issn]):
            codes = eissn_map[issn]
        if codes:
            codes_list = [int(code) for code in str(codes).replace(" ", "").replace(",", ";").split(";") if code.isdigit()]
            return codes_list
        else:
            return []
    df_export["Matched_ASJC"] = df_export.apply(get_asjc_codes, axis=1)

    # ASJC descriptions
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))
    df_export["Matched_ASJC_Description"] = df_export["Matched_ASJC"].apply(
        lambda codes: [asjc_dict.get(code, str(code)) for code in codes]
    )
    return df_export

def section_map_export_csv(df_export_with_asjc, df_asjc):
    st.header("Map Export CSV to Scopus Source & ASJC")
    df_export_with_asjc = add_asjc_to_export_csv(df_export, df_source, df_asjc)

    # Filtering by ASJC code
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

def section_author_analysis(df_export_with_asjc):
    st.header("Author Analysis")

    # ---- Prepare data ----
    # Explode ASJC so each row is 1 paper, 1 author, 1 ASJC
    df_expanded = df_export_with_asjc.copy()
    df_expanded = df_expanded.explode("Matched_ASJC_Description")

    # Parse authors (assuming 'Authors' column is "Last F.; Name S.; ..." with '; ' separator)
    author_rows = []
    for idx, row in df_expanded.iterrows():
        # Split authors
        authors = [a.strip() for a in str(row.get("Authors", "")).split(";") if a.strip()]
        asjc = row.get("Matched_ASJC_Description", None)
        # If you have a "Corresponding Author" or "Correspondence Address" column, update this!
        corresponding = row.get("Corresponding Author", None)
        for i, author in enumerate(authors):
            # Author type logic
            if i == 0:
                author_type = "First Author"
            else:
                author_type = "Co-author"
            # Corresponding author logic (example, adjust as needed)
            if corresponding and author in corresponding:
                author_type = "Corresponding Author"
            # If author is both first and corresponding, prioritize "Corresponding Author"
            author_rows.append({
                "Author": author,
                "ASJC": asjc,
                "Author Type": author_type
            })
    # Build DataFrame
    author_df = pd.DataFrame(author_rows)

    # If no explicit "Corresponding Author" info, just classify as First or Co-author
    # You can also deduplicate as needed

    st.write("All unique authors and their roles (first, corresponding, co-author):")
    st.dataframe(author_df.head(20))

    # ---- Aggregate: How many papers per author per ASJC and author type ----
    summary = (
        author_df.groupby(["Author", "ASJC", "Author Type"])
        .size()
        .reset_index(name="Paper Count")
        .sort_values(["Author", "ASJC"])
    )
    st.write("Table: Paper count per author, per ASJC, and author type")
    st.dataframe(summary)

    # Optional: Download as CSV
    st.download_button(
        "Download Author-ASJC-Type Table as CSV",
        data=summary.to_csv(index=False),
        file_name="author_asjc_type_summary.csv"
    )

# ===================
# --- Main App ------
# ===================

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
                # Create the mapped CSV ONCE here!
                df_export_with_asjc = add_asjc_to_export_csv(df_export, df_source, df_asjc)
    else:
        st.sidebar.info("Please upload the Scopus Source Excel first.")

    tabs = st.tabs(["Journal Filter", "Map Export CSV"])
    with tabs[0]:
        if df_source is not None and df_asjc is not None:
            section_journal_filter(df_source, df_asjc)
        else:
            st.info("Please upload the Scopus Source Excel to use this section.")
    with tabs[1]:
        if df_export_with_asjc is not None:
            section_map_export_csv(df_export_with_asjc, df_asjc)
            section_author_analysis(df_export_with_asjc)
        else:
            st.info("Please upload both the Scopus Source Excel and Export CSV(s) to use this section.")

if __name__ == "__main__":
    main()
