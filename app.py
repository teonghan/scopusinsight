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
    df = pl_source.with_columns([
        pl.col(col)
            .cast(pl.Utf8)
            .str.replace_all(" ", "")
            .str.replace_all(",", ";")  # replace ',' with ';'
            .alias("ASJC_clean")
    ]).filter(
        pl.col("ASJC_clean").is_not_null() & (pl.col("ASJC_clean") != "nan")
    ).with_columns([
        pl.col("ASJC_clean").str.split(";").alias("ASJC_list")
    ]).explode("ASJC_list").with_columns([
        pl.col("ASJC_list").cast(pl.Int64)
    ])
    return df

def filter_and_collect_matches_with_desc(pl_source, selected_codes, asjc_dict):
    col = "All Science Journal Classification Codes (ASJC)"
    df = pl_source.with_columns([
        pl.col(col)
            .cast(pl.Utf8)
            .str.replace_all(" ", "")
            .str.replace_all(",", ";")
            .alias("ASJC_clean")
    ]).filter(
        pl.col("ASJC_clean").is_not_null() & (pl.col("ASJC_clean") != "nan")
    ).with_columns([
        pl.col("ASJC_clean").str.split(";").alias("ASJC_list")
    ])

    # Collect the matching codes for each journal using .apply
    df = df.with_columns([
        pl.col("ASJC_list").apply(
            lambda arr: [int(code) for code in arr if code and int(code) in selected_codes] if isinstance(arr, list) else []
        ).alias("Matched_ASJC")
    ])

    # Map matched codes to descriptions
    def codes_to_desc(codes):
        if not isinstance(codes, list):
            return []
        return [asjc_dict.get(int(code), str(code)) for code in codes if code is not None]

    df = df.with_columns([
        pl.col("Matched_ASJC").apply(codes_to_desc).alias("Matched_ASJC_Description")
    ])

    # Only keep journals with at least one matched code
    df_filtered = df.filter(pl.col("Matched_ASJC").list.lengths() > 0)
    display_cols = [
        "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
        "Active or Inactive", "Source Type", "Publisher",
        "Publisher Imprints Grouped to Main Publisher",
        "Matched_ASJC", "Matched_ASJC_Description"
    ]
    return df_filtered.select([c for c in display_cols if c in df_filtered.columns])

def main():
    st.title("Scopus Journal Filter by ASJC Category (Polars Edition)")

    uploaded = st.file_uploader("Upload Scopus Source Excel", type=["xlsx"])
    if not uploaded:
        st.info("Please upload a Scopus Source Excel file.")
        return

    pl_source, pl_asjc = read_scopus_excel(uploaded)
    # pl_long = explode_asjc(pl_source)

    # Prepare options for ASJC selection
    asjc_options = dict(zip(pl_asjc["Code"].to_list(), pl_asjc["Description"].to_list()))
    asjc_dict = dict(zip(pl_asjc["Code"].to_list(), pl_asjc["Description"].to_list()))
    selected = st.multiselect(
        "Select ASJC Categories",
        options=list(asjc_options.keys()),
        format_func=lambda x: f"{x} – {asjc_options.get(x, '')}"
    )

    if selected:
        filtered = filter_and_collect_matches_with_desc(pl_source, [int(x) for x in selected], asjc_dict)
        st.write(f"Journals matching selected ASJC categories ({filtered.height}):")
        st.dataframe(
            filtered.with_columns(
                pl.col("Matched_ASJC_Description").apply(lambda x: "; ".join(x)).alias("Matched_ASJC_Description")
            ).to_pandas()
        )

    else:
        st.info("Select one or more ASJC categories to filter journals.")

if __name__ == "__main__":
    main()
