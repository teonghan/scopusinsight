import streamlit as st
import pandas as pd

def read_scopus_excel(file):
    """Reads the source list (selected columns) and ASJC table, returns as pandas DataFrames."""
    wanted_cols = [
        "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
        "Active or Inactive", "Source Type", "Publisher",
        "Publisher Imprints Grouped to Main Publisher",
        "All Science Journal Classification Codes (ASJC)",
    ]
    excel_file = pd.ExcelFile(file)
    # Source list
    df_source_full = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])
    cols_present = [col for col in wanted_cols if col in df_source_full.columns]
    df_source = df_source_full[cols_present]

    # ASJC code table from last sheet
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

def filter_and_collect_matches_with_desc(df_source, selected_codes, asjc_dict):
    # Clean and split the ASJC codes
    col = "All Science Journal Classification Codes (ASJC)"
    df = df_source.copy()
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(" ", "")
        .str.replace(",", ";")
        .replace("nan", "")
    )
    # Split codes into lists of ints
    df["ASJC_list"] = df[col].apply(lambda x: [int(code) for code in x.split(";") if code.isdigit()])
    # Matched codes (intersection with selected)
    df["Matched_ASJC"] = df["ASJC_list"].apply(lambda codes: [code for code in codes if code in selected_codes])
    df["Matched_ASJC_Description"] = df["Matched_ASJC"].apply(lambda codes: [asjc_dict.get(code, str(code)) for code in codes])
    # Only keep journals with at least one matched code
    df_filtered = df[df["Matched_ASJC"].apply(lambda x: len(x) > 0)].copy()
    display_cols = [
        "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
        "Active or Inactive", "Source Type", "Publisher",
        "Publisher Imprints Grouped to Main Publisher",
        "Matched_ASJC", "Matched_ASJC_Description"
    ]
    return df_filtered[display_cols]

def main():
    st.title("Scopus Journal Filter by ASJC Category (Pandas Edition)")

    uploaded = st.file_uploader("Upload Scopus Source Excel", type=["xlsx"])
    if not uploaded:
        st.info("Please upload a Scopus Source Excel file.")
        return

    df_source, df_asjc = read_scopus_excel(uploaded)
    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))

    selected = st.multiselect(
        "Select ASJC Categories",
        options=df_asjc["Code"],
        format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}"
    )

    if selected:
        filtered = filter_and_collect_matches_with_desc(df_source, selected, asjc_dict)
        st.write(f"Journals matching selected ASJC categories ({len(filtered)}):")
        st.dataframe(filtered)
    else:
        st.info("Select one or more ASJC categories to filter journals.")

if __name__ == "__main__":
    main()
