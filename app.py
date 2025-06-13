import streamlit as st
import pandas as pd

# === Caching functions ===

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

# === Section: Journal Filter ===

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
    selected = st.multiselect(
        "Select ASJC Categories",
        options=df_asjc["Code"],
        format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}"
    )

    filter_now = st.button("Filter Journals")
    if filter_now:
        if not selected:
            st.warning("Please select at least one ASJC category before filtering.")
        else:
            filtered = filter_and_collect_matches_with_desc(df_source, selected, asjc_dict)
            st.write(f"Journals matching selected ASJC categories ({len(filtered)}):")
            display_journal_table(filtered, asjc_dict)
    else:
        st.info("Select one or more ASJC categories, then click 'Filter Journals'.")

def display_journal_table(df, asjc_dict, filter_label="Filter by ASJC Categories"):
    """Display the DataFrame with an ASJC category filter on top."""
    # Collect unique codes
    all_codes = sorted(set(code for codes in df["Matched_ASJC"] for code in codes))
    selected = st.multiselect(
        filter_label,
        options=all_codes,
        format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
        key=filter_label  # So the filter widgets don't conflict across sections
    )
    if selected:
        df = df[df["Matched_ASJC"].apply(lambda codes: any(code in selected for code in codes))]
    st.dataframe(df)

def section_issn_asjc(df_export, df_source, df_asjc):
    st.header("Match ISSN to Scopus Source & ASJC")
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))

    issns = set(df_export["ISSN"].astype(str).dropna())
    df_source = df_source.copy()

    # First round: match by ISSN
    matched = df_source[df_source["ISSN"].astype(str).isin(issns)].copy()
    matched["MatchType"] = "ISSN"
    matched_issns = set(matched["ISSN"].astype(str))
    unmatched_issns = issns - matched_issns

    # Second round: match unmatched by EISSN
    matched2 = df_source[df_source["EISSN"].astype(str).isin(unmatched_issns)].copy()
    matched2["MatchType"] = "EISSN"
    # Concatenate results
    df_matched = pd.concat([matched, matched2], ignore_index=True)

    # Get ASJC lists for display (like in journal filter)
    col = "All Science Journal Classification Codes (ASJC)"
    df_matched[col] = (
        df_matched[col]
        .astype(str)
        .str.replace(" ", "")
        .str.replace(",", ";")
        .replace("nan", "")
    )
    df_matched["ASJC_list"] = df_matched[col].apply(lambda x: [int(code) for code in x.split(";") if code.isdigit()])
    df_matched["Matched_ASJC"] = df_matched["ASJC_list"]
    df_matched["Matched_ASJC_Description"] = df_matched["Matched_ASJC"].apply(lambda codes: [asjc_dict.get(code, str(code)) for code in codes])

    display_cols = [
        "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
        "Active or Inactive", "Source Type", "Publisher",
        "Publisher Imprints Grouped to Main Publisher", "MatchType",
        "Matched_ASJC", "Matched_ASJC_Description"
    ]
    df_final = df_matched[display_cols].copy()

    st.write(f"Found {len(df_final)} sources matched by ISSN/EISSN.")
    display_journal_table(df_final, asjc_dict, filter_label="Filter by ASJC Categories (Matched Section)")

# === Main App ===

def main():
    st.title("Scopus Analysis Toolkit")

    # === Sidebar for file uploads ===
    st.sidebar.header("1. Upload Files")
    excel_file = st.sidebar.file_uploader("Upload Scopus Source Excel", type=["xlsx"])
    csv_files = st.sidebar.file_uploader(
        "Upload up to 10 Scopus Export CSV files", type=["csv"], accept_multiple_files=True
    )

    # Load files
    df_source, df_asjc = None, None
    df_export, csv_error = None, None
    if excel_file:
        df_source, df_asjc = read_scopus_excel(excel_file)
    if csv_files:
        df_export, csv_error = read_and_merge_scopus_csv(csv_files)
        if csv_error:
            st.sidebar.error(csv_error)
        else:
            st.sidebar.success(f"Successfully merged {len(csv_files)} CSV files ({len(df_export)} rows).")

    # === Main: Section navigation ===
    tabs = st.tabs(["Journal Filter", "ISSN-ASJC Match"])
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

if __name__ == "__main__":
    main()
