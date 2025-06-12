import streamlit as st
import pandas as pd
import polars as pl

def read_scopus_excel(file):
    """Reads the source list (selected columns) and ASJC table, returns as Polars DataFrames."""
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
    pl_source = pl.from_pandas(df_source)

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
    pl_asjc = pl.from_pandas(asjc_cleaned)
    return pl_source, pl_asjc

def explode_asjc(pl_source):
    """
    Splits ASJC code column into long form using Polars.
    """
    col = "All Science Journal Classification Codes (ASJC)"
    # Remove spaces, replace missing with null, and split into lists
    df = pl_source.with_columns([
        pl.col(col).cast(pl.Utf8).str.replace_all(" ", "").alias("ASJC_clean")
    ]).filter(
        pl.col("ASJC_clean").is_not_null() & (pl.col("ASJC_clean") != "nan")
    ).with_columns([
        pl.col("ASJC_clean").str.split(";|,", inclusive=False).alias("ASJC_list")
    ]).explode("ASJC_list").with_columns([
        pl.col("ASJC_list").cast(pl.Int64)
    ])
    return df

def main():
    st.title("Scopus Journal Filter by ASJC Category (Polars Edition)")

    uploaded = st.file_uploader("Upload Scopus Source Excel", type=["xlsx"])
    if not uploaded:
        st.info("Please upload a Scopus Source Excel file.")
        return

    pl_source, pl_asjc = read_scopus_excel(uploaded)
    pl_long = explode_asjc(pl_source)

    # Prepare options for ASJC selection
    asjc_options = dict(zip(pl_asjc["Code"].to_list(), pl_asjc["Description"].to_list()))
    selected = st.multiselect(
        "Select ASJC Categories",
        options=list(asjc_options.keys()),
        format_func=lambda x: f"{x} – {asjc_options.get(x, '')}"
    )

    if selected:
        # Filter journals where any ASJC matches selection
        filtered = pl_long.filter(pl.col("ASJC_list").is_in(selected))
        # Show unique journals (by Sourcerecord ID)
        filtered_unique = filtered.unique(subset=["Sourcerecord ID"])
        st.write(f"Journals matching selected ASJC categories ({filtered_unique.height}):")
        st.dataframe(filtered_unique.to_pandas())
    else:
        st.info("Select one or more ASJC categories to filter journals.")

if __name__ == "__main__":
    main()
