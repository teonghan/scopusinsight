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

# ==================================
# --- Table Display with Filtering --
# ==================================

def display_journal_table(df, asjc_dict, filter_label="Filter by ASJC Categories"):
    all_codes = sorted(set(code for codes in df["Matched_ASJC"] for code in codes))
    selected = st.multiselect(
        filter_label,
        options=all_codes,
        format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
        key=filter_label  # Unique per section
    )
    if selected:
        df = df[df["Matched_ASJC"].apply(lambda codes: any(code in selected for code in codes))]
    st.dataframe(df)

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
            
            # ... any charts or stats here ...
            st.subheader("Journal Activity Status")
            if "Active or Inactive" in filtered.columns:
                status_counts = filtered["Active or Inactive"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig_status = px.pie(status_counts, names="Status", values="Count", title="Active vs Inactive Journals")
                st.plotly_chart(fig_status, use_container_width=True)

            # --- Stacked Bar Chart: Active/Inactive by Source Type ---
            st.subheader("Journal Activity Status by Source Type")
            if "Active or Inactive" in filtered.columns and "Source Type" in filtered.columns:
                # Prepare data
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
# --- ISSN-ASJC Match ------
# ==========================
def add_asjc_to_export_csv(df_export, df_source, df_asjc):
    # Standardize ISSN/EISSN in source
    df_source = df_source.copy()
    df_source["ISSN_clean"] = df_source["ISSN"].apply(clean_issn)
    df_source["EISSN_clean"] = df_source["EISSN"].apply(clean_issn)

    # Create mapping from ISSN/EISSN to ASJC codes
    issn_map = df_source.set_index('ISSN_clean')["All Science Journal Classification Codes (ASJC)"].to_dict()
    eissn_map = df_source.set_index('EISSN_clean')["All Science Journal Classification Codes (ASJC)"].to_dict()

    # Ensure CSV columns are standardized
    df_export = df_export.copy()
    df_export["ISSN"] = df_export["ISSN"].apply(clean_issn)
    if "EISSN" in df_export.columns:
        df_export["EISSN"] = df_export["EISSN"].apply(clean_issn)
    else:
        df_export["EISSN"] = None

    # Add ASJC list to each CSV row
    def get_asjc_codes(row):
        issn = row["ISSN"]
        eissn = row["EISSN"]
        codes = None
        if issn in issn_map and pd.notna(issn_map[issn]):
            codes = issn_map[issn]
        elif eissn in eissn_map and pd.notna(eissn_map[eissn]):
            codes = eissn_map[eissn]
        if codes:
            codes_list = [int(code) for code in str(codes).replace(" ", "").replace(",", ";").split(";") if code.isdigit()]
            return codes_list
        else:
            return []
    df_export["Matched_ASJC"] = df_export.apply(get_asjc_codes, axis=1)

    # Get ASJC descriptions
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))
    df_export["Matched_ASJC_Description"] = df_export["Matched_ASJC"].apply(lambda codes: [asjc_dict.get(code, str(code)) for code in codes])
    return df_export

def section_issn_asjc_export_csv(df_export, df_source, df_asjc):
    st.header("Map Export CSV to Scopus Source & ASJC")
    # Add ASJC codes to CSV
    df_export_with_asjc = add_asjc_to_export_csv(df_export, df_source, df_asjc)

    # Optional: Filtering by ASJC code
    all_codes = sorted(set(code for codes in df_export_with_asjc["Matched_ASJC"] for code in codes))
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))

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

# ======================================
# --- Export CSV Tagged with ASJC -------
# ======================================

def section_issn_asjc_export_tagged(df_export, df_source, df_asjc):
    st.header("Scopus Export CSV Tagged with ASJC")
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))

    # Standardize ISSN/EISSN
    df_source = df_source.copy()
    df_source["ISSN_clean"] = df_source["ISSN"].apply(clean_issn)
    df_source["EISSN_clean"] = df_source["EISSN"].apply(clean_issn)
    df_export = df_export.copy()
    df_export["ISSN_clean"] = df_export["ISSN"].apply(clean_issn)
    if "EISSN" not in df_export.columns:
        df_export["EISSN"] = None
    df_export["EISSN_clean"] = df_export["EISSN"].apply(clean_issn)

    issn_map = df_source.set_index('ISSN_clean')["All Science Journal Classification Codes (ASJC)"].to_dict()
    eissn_map = df_source.set_index('EISSN_clean')["All Science Journal Classification Codes (ASJC)"].to_dict()

    def get_asjc_codes(row):
        issn = row["ISSN_clean"]
        eissn = row["EISSN_clean"]
        codes = None
        if issn in issn_map and pd.notna(issn_map[issn]):
            codes = issn_map[issn]
        elif eissn in eissn_map and pd.notna(eissn_map[eissn]):
            codes = eissn_map[eissn]
        if codes:
            codes_list = [int(code) for code in str(codes).replace(" ", "").replace(",", ";").split(";") if code.isdigit()]
            return codes_list
        else:
            return []
    df_export["Matched_ASJC"] = df_export.apply(get_asjc_codes, axis=1)
    df_export["Matched_ASJC_Description"] = df_export["Matched_ASJC"].apply(lambda codes: [asjc_dict.get(code, str(code)) for code in codes])

    preview_cols = [
        "ISSN", "EISSN", "Cited by", "DOI", "Matched_ASJC", "Matched_ASJC_Description"
    ]
    for col in ["Title", "Authors", "Year", "Source title"]:
        if col in df_export.columns and col not in preview_cols:
            preview_cols = [col] + preview_cols

    all_codes = sorted(set(code for codes in df_export["Matched_ASJC"] for code in codes))
    selected = st.multiselect(
        "Filter by ASJC Categories (Export CSV section)",
        options=all_codes,
        format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
        key="exportcsv_asjc"
    )
    df_show = df_export.copy()
    if selected:
        df_show = df_show[df_show["Matched_ASJC"].apply(lambda codes: any(code in selected for code in codes))]
    st.dataframe(df_show[preview_cols])

# ===================
# --- Main App ------
# ===================

def main():
    st.title("Scopus Analysis Toolkit")

    # 1. Excel uploader appears first
    st.sidebar.header("1. Upload Scopus Source Excel")
    excel_file = st.sidebar.file_uploader("Upload Scopus Source Excel", type=["xlsx"], key="excel_upload")

    df_source, df_asjc = None, None
    df_export, csv_error = None, None

    # 2. Read Excel before showing CSV uploader
    if excel_file:
        df_source, df_asjc = read_scopus_excel(excel_file)
        st.sidebar.success("Excel file loaded. Please proceed to upload CSV file(s).")

        # 3. Now show CSV uploader
        st.sidebar.header("2. Upload Scopus Export CSV(s)")
        csv_files = st.sidebar.file_uploader(
            "Upload up to 10 Scopus Export CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="csv_upload"
        )

        # 4. Read CSV files if uploaded
        if csv_files:
            df_export, csv_error = read_and_merge_scopus_csv(csv_files)
            if csv_error:
                st.sidebar.error(csv_error)
            else:
                st.sidebar.success(f"Successfully merged {len(csv_files)} CSV files ({len(df_export)} rows).")
    else:
        # 5. CSV uploader is hidden until Excel is uploaded
        st.sidebar.info("Please upload the Scopus Source Excel first.")

    # Main tabs
    tabs = st.tabs(["Journal Filter", "ISSN-ASJC Match", "Export CSV Tagged"])
    with tabs[0]:
        if df_source is not None and df_asjc is not None:
            section_journal_filter(df_source, df_asjc)
        else:
            st.info("Please upload the Scopus Source Excel to use this section.")
    with tabs[1]:
        if df_export is not None and df_source is not None and df_asjc is not None:
            section_issn_asjc(df_export, df_source, df_asjc)
        else:
            st.info("Please upload both the Scopus Source Excel and Export CSV(s) to use this section.")
    with tabs[2]:
        if df_export is not None and df_source is not None and df_asjc is not None:
            section_issn_asjc_export_tagged(df_export, df_source, df_asjc)
        else:
            st.info("Please upload both the Scopus Source Excel and Export CSV(s) to use this section.")

if __name__ == "__main__":
    main()

