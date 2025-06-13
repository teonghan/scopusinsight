import streamlit as st
import pandas as pd

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

def main():
    st.title("Scopus Journal Filter by ASJC Category (Pandas Edition)")

    uploaded = st.file_uploader("Upload Scopus Source Excel", type=["xlsx"])
    if uploaded:
        # Clear session state if a new file is uploaded
        if "last_filename" not in st.session_state or uploaded.name != st.session_state.get("last_filename", ""):
            st.session_state.filtered = None
            st.session_state.selected = []
            st.session_state.last_filename = uploaded.name

        df_source, df_asjc = read_scopus_excel(uploaded)
        asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))

        selected = st.multiselect(
            "Select ASJC Categories",
            options=df_asjc["Code"],
            format_func=lambda x: f"{x} – {asjc_dict.get(x, '')}",
            default=st.session_state.get("selected", [])
        )

        filter_now = st.button("Filter Journals")

        # Store selection in session state
        st.session_state.selected = selected

        if filter_now and selected:
            filtered = filter_and_collect_matches_with_desc(df_source, selected, asjc_dict)
            st.session_state.filtered = filtered
            st.write(f"Journals matching selected ASJC categories ({len(filtered)}):")
            st.dataframe(filtered)
        elif filter_now and not selected:
            st.session_state.filtered = None
            st.warning("Please select at least one ASJC category before filtering.")
        elif st.session_state.get("filtered") is not None:
            filtered = st.session_state.filtered
            st.write(f"Journals matching selected ASJC categories ({len(filtered)}):")
            st.dataframe(filtered)
        else:
            st.info("Select one or more ASJC categories, then click 'Filter Journals'.")
    else:
        st.info("Please upload a Scopus Source Excel file.")

if __name__ == "__main__":
    main()
