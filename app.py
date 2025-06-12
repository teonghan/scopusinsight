import streamlit as st
import pandas as pd

def read_scopus_excel(file):
    """Extracts relevant columns from source sheet and cleans ASJC codes."""
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

def explode_asjc(df):
    """Splits ASJC codes into long form for filtering."""
    df = df.copy()
    col = "All Science Journal Classification Codes (ASJC)"
    df["ASJC_list"] = (
        df[col].astype(str).str.replace(" ", "")
        .replace("nan", pd.NA)
        .str.split(";|,")
    )
    df = df.explode("ASJC_list")
    df["ASJC_list"] = pd.to_numeric(df["ASJC_list"], errors="coerce").astype("Int64")
    return df

def filter_by_asjc(df_long, selected_codes):
    """Filters the exploded sources by selected ASJC codes."""
    return df_long[df_long["ASJC_list"].isin(selected_codes)]

def main():
    st.title("Scopus Journal Filter by ASJC Category")

    uploaded = st.file_uploader("Upload Scopus Source Excel", type=["xlsx"])
    if not uploaded:
        st.info("Please upload a Scopus Source Excel file.")
        return

    df_source, df_asjc = read_scopus_excel(uploaded)
    df_long = explode_asjc(df_source)

    asjc_options = df_asjc.set_index("Code")["Description"].to_dict()
    selected = st.multiselect(
        "Select ASJC Categories",
        options=list(asjc_options.keys()),
        format_func=lambda x: f"{x} – {asjc_options.get(x, '')}"
    )

    if selected:
        filtered = filter_by_asjc(df_long, selected)
        st.write(f"Journals matching selected ASJC categories ({filtered['Sourcerecord ID'].nunique()}):")
        st.dataframe(filtered.drop_duplicates(subset=["Sourcerecord ID"]))
    else:
        st.info("Select one or more ASJC categories to filter journals.")

if __name__ == "__main__":
    main()
