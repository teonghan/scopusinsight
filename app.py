import streamlit as st
import pandas as pd
import re
import plotly.express as px # Import Plotly Express

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

def parse_asjc_list(asjc_str):
    """Parse ASJC string to list of integer codes."""
    # Ensure it's a string, replace common delimiters with a consistent one (semicolon), then split
    clean_str = str(asjc_str).strip().replace(" ", "").replace(",", ";")
    return [int(code) for code in clean_str.split(";") if code.isdigit()]

# ========================
#     DATA LOADING
# ========================

@st.cache_data
def read_scopus_excel(file):
    """Read Scopus Source Excel and return two DataFrames: sources and ASJC codes."""
    try:
        excel_file = pd.ExcelFile(file)

        # Read the first sheet (assuming 'Source List' or similar) for journal metadata
        # Adjust skiprows and header if your actual Scopus file structure varies
        df_source_full = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])

        wanted_cols = [
            "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
            "Active or Inactive", "Source Type", "Publisher",
            "Publisher Imprints Grouped to Main Publisher",
            "All Science Journal Classification Codes (ASJC)",
        ]
        # Filter for columns that are actually present in the uploaded Excel
        cols_present = [col for col in wanted_cols if col in df_source_full.columns]
        df_source = df_source_full[cols_present]
        df_source["ISSN"] = df_source["ISSN"].apply(clean_issn)
        df_source["EISSN"] = df_source["EISSN"].apply(clean_issn)

        # Read the last sheet (assuming 'ASJC' or similar) for ASJC code descriptions
        # Adjust skiprows and nrows if your actual ASJC sheet structure varies
        asjc_df = pd.read_excel(
            excel_file,
            sheet_name=excel_file.sheet_names[-1],
            usecols=[0, 1],
            skiprows=8, # This might need adjustment based on your file's header rows
            nrows=362  # This might need adjustment based on the number of ASJC entries
        )
        asjc_df.columns = ["Code", "Description"] # Ensure consistent column names
        asjc_cleaned = asjc_df.dropna(subset=["Code"]).copy()
        asjc_cleaned["Code"] = asjc_cleaned["Code"].astype(int)
        return df_source, asjc_cleaned
    except Exception as e:
        st.error(f"Error reading Scopus Excel file: {e}. Please ensure it's a valid Scopus Source List.")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def read_qs_subject_file(file):
    """Read QS Subject Area file (CSV or Excel) and return a DataFrame."""
    try:
        # Determine file type by extension
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df_qs = pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            df_qs = pd.read_excel(file)
        else:
            st.error("Unsupported file type for QS Subject. Please upload a CSV or Excel file.")
            return pd.DataFrame()

        # Ensure ASJC Code is integer for merging/filtering
        if 'ASJC Code' in df_qs.columns:
            df_qs['ASJC Code'] = df_qs['ASJC Code'].astype(int)
        else:
            st.error("The uploaded QS Subject file does not contain an 'ASJC Code' column. Please check your file.")
            return pd.DataFrame()

        return df_qs
    except Exception as e:
        st.error(f"Error reading QS Subject file: {e}. Please ensure it's a valid CSV or Excel file with an 'ASJC Code' column.")
        return pd.DataFrame()

@st.cache_data
def read_scopus_export_csv(files):
    """
    Reads and merges multiple Scopus export CSV files.
    Performs basic cleaning on ISSN and ensures consistent columns.
    """
    if not files:
        return pd.DataFrame(), "No files provided."

    all_dfs = []
    error_messages = []
    # Only require ISSN for Scopus Export, as EISSN might not be present
    required_cols = ["Source title", "ISSN"]

    for file in files:
        try:
            df = pd.read_csv(file)
            # Check if all required columns are present
            if not all(col in df.columns for col in required_cols):
                error_messages.append(f"File '{file.name}' is missing one or more required columns ({', '.join(required_cols)}).")
                continue

            # Clean ISSN immediately after reading
            df['ISSN'] = df['ISSN'].apply(clean_issn)
            # Check for EISSN in export and clean if present, but do not *require* it
            if 'EISSN' in df.columns:
                df['EISSN'] = df['EISSN'].apply(clean_issn)
            else:
                df['EISSN'] = None # Ensure EISSN column exists, even if empty, for later consistency

            all_dfs.append(df)
        except Exception as e:
            error_messages.append(f"Error reading file '{file.name}': {e}")

    if not all_dfs:
        return pd.DataFrame(), "; ".join(error_messages) if error_messages else "No valid CSV files were loaded."

    # Concatenate all valid dataframes
    try:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        # Drop duplicates based on a combination of identifiers (e.g., Title, Year, Source title, DOI, EID)
        # Assuming 'EID' is a unique document identifier
        if 'EID' in merged_df.columns:
            merged_df.drop_duplicates(subset=['EID'], inplace=True)
        else:
            # Fallback if EID is not present, use a combination that is likely unique
            st.warning("EID column not found in export CSVs. Deduplication might be less precise.")
            merged_df.drop_duplicates(subset=['Title', 'Year', 'Source title', 'DOI'], inplace=True)

        return merged_df, None
    except Exception as e:
        return pd.DataFrame(), f"Error merging CSV files: {e}"


# ===========================
#       FILTERING LOGIC
# ===========================

def filter_journals_by_asjc(df_source, selected_codes, asjc_dict):
    """
    Filter journals in df_source for matching ASJC codes.
    Adds 'Matched_ASJC' and 'Matched_ASJC_Description' columns.
    """
    col = "All Science Journal Classification Codes (ASJC)"
    df = df_source.copy()

    # Handle NaN values and ensure string conversion before splitting
    df[col] = df[col].astype(str).str.replace(" ", "").str.replace(",", ";").replace("nan", "")
    df["ASJC_list"] = df[col].apply(parse_asjc_list)

    df["Matched_ASJC"] = df["ASJC_list"].apply(
        lambda codes: [code for code in codes if code in selected_codes]
    )
    df["Matched_ASJC_Description"] = df["Matched_ASJC"].apply(
        lambda codes: [asjc_dict.get(code, str(code)) for code in codes]
    )

    df_filtered = df[df["Matched_ASJC"].apply(lambda x: len(x) > 0)].copy()

    display_cols = [
        "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
        "Active or Inactive", "Source Type", "Publisher",
        "Publisher Imprints Grouped to Main Publisher",
        "Matched_ASJC", "Matched_ASJC_Description"
    ]

    # Filter display_cols to only include columns that actually exist in df_filtered
    existing_display_cols = [col for col in display_cols if col in df_filtered.columns]

    return df_filtered[existing_display_cols]


def tag_export_with_asjc_and_qs(df_export, df_source, df_asjc, df_qs_subject):
    """
    Tags the Scopus Export DataFrame with ASJC codes and QS Subject Areas
    by looking up ISSN/EISSN in the Scopus Source data,
    following the specific order: df_export['ISSN'] -> df_source['ISSN'],
    then for unmatched, df_export['ISSN'] -> df_source['EISSN'].
    """
    if df_export.empty or df_source.empty or df_asjc.empty or df_qs_subject.empty:
        return pd.DataFrame()

    df_tagged = df_export.copy()

    # Create ASJC Code to Description mapping
    asjc_code_to_desc_map = dict(zip(df_asjc["Code"], df_asjc["Description"]))

    # Create ASJC Code to QS Subject Area(s) mapping
    asjc_to_qs_map = {}
    for idx, row in df_qs_subject.iterrows():
        asjc_code = row['ASJC Code']
        qs_subject_area = row['QS Subject Area']
        if asjc_code not in asjc_to_qs_map:
            asjc_to_qs_map[asjc_code] = set()
        asjc_to_qs_map[asjc_code].add(qs_subject_area)

    # Clean ISSN/EISSN in df_export (already done in read_scopus_export_csv but good to be safe)
    # Ensure ISSN column exists and is cleaned
    if 'ISSN' in df_tagged.columns:
        df_tagged['ISSN'] = df_tagged['ISSN'].apply(clean_issn)
    else:
        df_tagged['ISSN'] = None # Create if missing, fill with None

    # Ensure EISSN column exists and is cleaned (it might be missing in export, handled in read_csv)
    if 'EISSN' in df_tagged.columns:
        df_tagged['EISSN'] = df_tagged['EISSN'].apply(clean_issn)
    else:
        df_tagged['EISSN'] = None # Create if missing, fill with None


    # Prepare lookup dictionaries from df_source for ASJC codes
    # Ensure source ISSN/EISSN columns are clean and unique for mapping
    source_issn_map = df_source[['ISSN', 'All Science Journal Classification Codes (ASJC)']].dropna(subset=['ISSN']).set_index('ISSN')['All Science Journal Classification Codes (ASJC)'].to_dict()
    source_eissn_map = df_source[['EISSN', 'All Science Journal Classification Codes (ASJC)']].dropna(subset=['EISSN']).set_index('EISSN')['All Science Journal Classification Codes (ASJC)'].to_dict()

    # Initialize a temporary column to store matched ASJC codes
    df_tagged['__matched_asjc_codes_raw'] = None

    # Step 1: Lookup df_export['ISSN'] against df_source['ISSN']
    # Ensure ISSN column exists in df_tagged before mapping
    if 'ISSN' in df_tagged.columns:
        df_tagged['__matched_asjc_codes_raw'] = df_tagged['ISSN'].map(source_issn_map)

    # Step 2: For records still unmatched, lookup df_export['ISSN'] against df_source['EISSN']
    # Identify rows where ASJC code is still missing after the first lookup
    # Only proceed if '__matched_asjc_codes_raw' exists and there are NaNs
    if '__matched_asjc_codes_raw' in df_tagged.columns:
        unmatched_mask = df_tagged['__matched_asjc_codes_raw'].isna()
        # Apply the EISSN lookup ONLY to the unmatched rows
        # Ensure ISSN column exists in df_tagged before mapping
        if 'ISSN' in df_tagged.columns:
            df_tagged.loc[unmatched_mask, '__matched_asjc_codes_raw'] = \
                df_tagged.loc[unmatched_mask, 'ISSN'].map(source_eissn_map)

    # Process the 'matched_asjc_codes_raw' to get ASJC_list and descriptions
    # Ensure '__matched_asjc_codes_raw' column exists before applying
    if '__matched_asjc_codes_raw' in df_tagged.columns:
        df_tagged['ASJC_list'] = df_tagged['__matched_asjc_codes_raw'].apply(
            lambda x: parse_asjc_list(x) if pd.notna(x) else []
        )
    else:
        df_tagged['ASJC_list'] = [[]] * len(df_tagged) # Initialize with empty lists if lookup didn't happen

    # Add Matched ASJC Code and Description (as lists for "bubble" effect)
    df_tagged["Matched_ASJC"] = df_tagged["ASJC_list"].apply(
        lambda codes: [code for code in codes if code in asjc_code_to_desc_map]
    )
    df_tagged["Matched_ASJC_Description"] = df_tagged["Matched_ASJC"].apply(
        lambda codes: [asjc_code_to_desc_map.get(code, str(code)) for code in codes]
    )

    # Add Matched QS Subject Area (as list for "bubble" effect)
    df_tagged['Matched_QS_Subject_Area'] = df_tagged['Matched_ASJC'].apply(
        lambda matched_asjcs: sorted(list(set(
            qs_subject
            for asjc_code in matched_asjcs
            for qs_subject in asjc_to_qs_map.get(asjc_code, [])
        )))
    )

    # Clean up temporary columns
    df_tagged.drop(columns=['__matched_asjc_codes_raw', 'ASJC_list'], inplace=True, errors='ignore')

    return df_tagged


# ===========================
#       UI SECTIONS
# ===========================

def section_scopus_asjc_filter(df_source, df_asjc):
    """UI for journal filtering by Scopus ASJC codes."""
    st.subheader("Filter Journals by Scopus ASJC Categories")

    if df_source.empty or df_asjc.empty:
        st.info("Please upload the Scopus Source Excel to use this section.")
        return

    asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))
    all_asjc_codes = sorted(list(df_asjc["Code"])) # Ensure sorted for consistent display

    select_all_asjc = st.checkbox("Select All Scopus ASJC Categories", key="select_all_scopus_asjc_tab1")

    selected_asjc = st.multiselect(
        "Select Scopus ASJC Categories:",
        options=all_asjc_codes,
        default=all_asjc_codes if select_all_asjc else [],
        format_func=lambda x: f"{x} – {asjc_dict.get(x, 'Unknown Description')}",
        key="scopus_asjc_multiselect_tab1"
    )

    filter_button = st.button("Filter by Scopus ASJC", key="filter_scopus_asjc_btn_tab1")

    if filter_button:
        if not selected_asjc:
            st.warning("Please select at least one Scopus ASJC category before filtering.")
        else:
            df_filtered = filter_journals_by_asjc(df_source, selected_asjc, asjc_dict)
            st.write(f"Journals matching selected Scopus ASJC categories ({len(df_filtered)}):")
            st.dataframe(df_filtered, use_container_width=True)
            if not df_filtered.empty:
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Scopus ASJC Journals",
                    data=csv,
                    file_name="filtered_scopus_asjc_journals.csv",
                    mime="text/csv",
                    key="download_scopus_asjc_btn_tab1"
                )
    else:
        st.info("Select one or more Scopus ASJC categories, then click 'Filter by Scopus ASJC'.")

def section_qs_subject_filter(df_source, df_asjc, df_qs_subject, qs_file_uploaded):
    """UI for journal filtering by QS Narrow Subject Area."""
    st.subheader("Filter Journals by QS Narrow Subject Area")

    if df_source.empty or df_asjc.empty:
        st.info("Please upload the Scopus Source Excel to use this section.")
        return

    # Check if QS file is uploaded before proceeding
    if not qs_file_uploaded or df_qs_subject.empty:
        st.warning("This functionality is not available until the QS Subject file has been uploaded.")
        return

    # Use 'QS Subject Area' for the narrow subject filter
    if 'QS Subject Area' in df_qs_subject.columns:
        all_qs_subjects = sorted(df_qs_subject['QS Subject Area'].unique())
    else:
        st.error("The uploaded QS Subject file does not contain a 'QS Subject Area' column. Please check your file.")
        return

    select_all_qs = st.checkbox("Select All QS Subject Areas", key="select_all_qs_subject_tab2")

    selected_qs_subjects = st.multiselect(
        "Select QS Narrow Subject Areas:",
        options=all_qs_subjects,
        default=all_qs_subjects if select_all_qs else [],
        key="qs_subject_multiselect_tab2"
    )

    filter_button = st.button("Filter by QS Subject", key="filter_qs_subject_btn_tab2")

    if filter_button:
        if not selected_qs_subjects:
            st.warning("Please select at least one QS Narrow Subject Area before filtering.")
        else:
            # Get ASJC codes corresponding to selected QS subjects
            asjc_codes_from_qs = df_qs_subject[
                df_qs_subject['QS Subject Area'].isin(selected_qs_subjects)
            ]['ASJC Code'].unique().tolist()

            # Use the existing ASJC dictionary for descriptions
            asjc_dict = dict(zip(df_asjc["Code"], df_asjc["Description"]))

            # Filter Scopus journals using the collected ASJC codes
            df_filtered_qs = filter_journals_by_asjc(df_source, asjc_codes_from_qs, asjc_dict)

            # Create a mapping from ASJC Code to QS Subject Area(s) from df_qs_subject
            asjc_to_qs_map = {}
            for idx, row in df_qs_subject.iterrows():
                asjc_code = row['ASJC Code']
                qs_subject_area = row['QS Subject Area']
                if asjc_code not in asjc_to_qs_map:
                    asjc_to_qs_map[asjc_code] = set()
                asjc_to_qs_map[asjc_code].add(qs_subject_area)

            df_filtered_qs['Matched_QS_Subject_Area'] = df_filtered_qs['Matched_ASJC'].apply(
                lambda matched_asjcs: sorted(list(set(
                    qs_subject
                    for asjc_code in matched_asjcs
                    for qs_subject in asjc_to_qs_map.get(asjc_code, [])
                )))
            )

            st.write(f"Journals matching selected QS Subject Areas ({len(df_filtered_qs)}):")

            display_cols = [
                "Sourcerecord ID", "Source Title", "ISSN", "EISSN",
                "Active or Inactive", "Source Type", "Publisher",
                "Publisher Imprints Grouped to Main Publisher",
                "Matched_ASJC", "Matched_ASJC_Description", "Matched_QS_Subject_Area"
            ]
            existing_display_cols = [col for col in display_cols if col in df_filtered_qs.columns]

            st.dataframe(df_filtered_qs[existing_display_cols], use_container_width=True)

            if not df_filtered_qs.empty:
                df_to_download = df_filtered_qs[existing_display_cols].copy()
                df_to_download['Matched_QS_Subject_Area'] = df_to_download['Matched_QS_Subject_Area'].apply(
                    lambda x: "; ".join(x) if isinstance(x, list) else x
                )
                csv = df_to_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered QS Subject Journals",
                    data=csv,
                    file_name="filtered_qs_subject_journals.csv",
                    mime="text/csv",
                    key="download_qs_subject_btn_tab2"
                )
    else:
        st.info("Select one or more QS Narrow Subject Areas, then click 'Filter by QS Subject'.")

def section_scopus_export_analysis(df_export_tagged, df_qs_subject, export_file_uploaded):
    """UI for analyzing and filtering the uploaded Scopus Export data."""
    st.subheader("Analyze Scopus Export Data by QS Subject")

    if df_export_tagged.empty or not export_file_uploaded:
        st.warning("This functionality is not available until the Scopus Export CSV(s) have been uploaded.")
        return

    if df_qs_subject.empty:
        st.info("Please upload the QS Subject file to filter by QS Narrow Subject Area.")
        return

    # Get unique QS Subject Areas from the QS Subject file
    if 'QS Subject Area' in df_qs_subject.columns:
        all_qs_subjects = sorted(df_qs_subject['QS Subject Area'].unique())
    else:
        st.error("The uploaded QS Subject file does not contain a 'QS Subject Area' column. Please check your file.")
        return

    select_all_export_qs = st.checkbox("Select All QS Subject Areas for Export Analysis", key="select_all_export_qs_tab3")

    selected_export_qs_subjects = st.multiselect(
        "Select QS Narrow Subject Areas to Filter Export:",
        options=all_qs_subjects,
        default=all_qs_subjects if select_all_export_qs else [],
        key="export_qs_subject_multiselect_tab3"
    )

    filter_export_button = st.button("Filter Export by QS Subject", key="filter_export_qs_btn_tab3")

    # Define columns to display for the export analysis at a broader scope
    export_display_cols = [
        "Title", "Source title", "Year", "ISSN", "EISSN", "DOI", "Cited by",
        "Matched_ASJC", "Matched_ASJC_Description", "Matched_QS_Subject_Area"
    ]

    df_chart_data = pd.DataFrame() # Initialize an empty DataFrame
    existing_export_display_cols = [] # Initialize here as well

    # Determine which DataFrame to use for display and charting
    # If the filter button is clicked and subjects are selected, use df_filtered_export
    # Otherwise, use the full df_export_tagged
    if filter_export_button and selected_export_qs_subjects:
        # Filter the already tagged df_export_tagged DataFrame
        df_filtered_export = df_export_tagged[
            df_export_tagged['Matched_QS_Subject_Area'].apply(
                lambda x: any(qs in selected_export_qs_subjects for qs in x) if isinstance(x, list) else False
            )
        ].copy()
        df_chart_data = df_filtered_export # Use filtered data for charts and table
        st.write(f"Publications matching selected QS Subject Areas ({len(df_filtered_export)}):")

        # Filter display_cols to only include columns that actually exist in df_filtered_export
        existing_export_display_cols = [col for col in export_display_cols if col in df_filtered_export.columns]

        st.dataframe(df_filtered_export[existing_export_display_cols], use_container_width=True)

        if not df_filtered_export.empty:
            df_to_download_export = df_filtered_export[existing_export_display_cols].copy()
            df_to_download_export['Matched_ASJC'] = df_to_download_export['Matched_ASJC'].apply(
                lambda x: "; ".join(map(str, x)) if isinstance(x, list) else x
            )
            df_to_download_export['Matched_ASJC_Description'] = df_to_download_export['Matched_ASJC_Description'].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else x
            )
            df_to_download_export['Matched_QS_Subject_Area'] = df_to_download_export['Matched_QS_Subject_Area'].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else x
            )
            csv_export = df_to_download_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Scopus Export Data",
                data=csv_export,
                file_name="filtered_scopus_export_data.csv",
                mime="text/csv",
                key="download_export_qs_btn_tab3"
            )
    elif df_export_tagged is not None and not df_export_tagged.empty:
        df_chart_data = df_export_tagged # Use full data for charts and table if no filter applied
        st.write("Displaying all uploaded and tagged publications:")

        existing_export_display_cols = [col for col in export_display_cols if col in df_export_tagged.columns]
        st.dataframe(df_export_tagged[existing_export_display_cols], use_container_width=True)
    else:
        st.info("Select one or more QS Narrow Subject Areas to filter your uploaded Scopus Export data.")


    st.markdown("---")
    st.subheader("Publication Counts by Category")

    # Radio button for chart metric selection
    chart_metric = st.radio(
        "Choose metric for charts:",
        ("Count Publication", "Count Citation"),
        key="chart_metric_radio"
    )

    metric_column = 'EID' if chart_metric == "Count Publication" else 'Cited by'
    metric_label = 'Number of Unique Publications' if chart_metric == "Count Publication" else 'Total Citations'
    aggregation_func = 'nunique' if chart_metric == "Count Publication" else 'sum'

    if not df_chart_data.empty: # Use df_chart_data for plotting
        # Ensure 'Cited by' column is numeric for sum aggregation
        if 'Cited by' in df_chart_data.columns:
            df_chart_data['Cited by'] = pd.to_numeric(df_chart_data['Cited by'], errors='coerce').fillna(0)
        elif chart_metric == "Count Citation":
            st.warning("The 'Cited by' column is not available in the uploaded export data for citation analysis.")
            return # Exit if citation analysis requested but column is missing


        # Chart 1: Publications per QS Narrow Subject
        st.markdown("#### Publications per QS Narrow Subject")
        # Explode the list column to count each subject per record
        df_qs_exploded = df_chart_data.explode('Matched_QS_Subject_Area')
        # Aggregate based on selected metric
        if aggregation_func == 'nunique':
            qs_counts = df_qs_exploded.dropna(subset=['Matched_QS_Subject_Area']).groupby('Matched_QS_Subject_Area')[metric_column].nunique().reset_index(name=metric_label)
        else: # 'sum' for 'Cited by'
            qs_counts = df_qs_exploded.dropna(subset=['Matched_QS_Subject_Area']).groupby('Matched_QS_Subject_Area')[metric_column].sum().reset_index(name=metric_label)

        if not qs_counts.empty:
            qs_counts = qs_counts.sort_values(by=metric_label, ascending=False)
            order_qs_subjects = qs_counts['Matched_QS_Subject_Area'].tolist()
            fig_qs = px.bar(qs_counts, y='Matched_QS_Subject_Area', x=metric_label, # Flipped axes
                            title=f'{metric_label} by QS Narrow Subject',
                            labels={'Matched_QS_Subject_Area': 'QS Narrow Subject Area', metric_label: metric_label},
                            template='plotly_white',
                            category_orders={'Matched_QS_Subject_Area': order_qs_subjects})
            st.plotly_chart(fig_qs, use_container_width=True)
        else:
            st.info("No QS Narrow Subject data available for charting based on current filter.")

        st.markdown("---")

        # Chart 2: Publications per ASJC Category
        st.markdown("#### Publications per ASJC Category")
        # Explode the list column to count each ASJC description per record
        df_asjc_exploded = df_chart_data.explode('Matched_ASJC_Description')
        # Aggregate based on selected metric
        if aggregation_func == 'nunique':
            asjc_counts = df_asjc_exploded.dropna(subset=['Matched_ASJC_Description']).groupby('Matched_ASJC_Description')[metric_column].nunique().reset_index(name=metric_label)
        else: # 'sum' for 'Cited by'
            asjc_counts = df_asjc_exploded.dropna(subset=['Matched_ASJC_Description']).groupby('Matched_ASJC_Description')[metric_column].sum().reset_index(name=metric_label)

        if not asjc_counts.empty:
            asjc_counts = asjc_counts.sort_values(by=metric_label, ascending=False)
            order_asjc_descriptions = asjc_counts['Matched_ASJC_Description'].tolist()
            fig_asjc = px.bar(asjc_counts, y='Matched_ASJC_Description', x=metric_label, # Flipped axes
                              title=f'{metric_label} by ASJC Category',
                              labels={'Matched_ASJC_Description': 'ASJC Category', metric_label: metric_label},
                              template='plotly_white',
                              category_orders={'Matched_ASJC_Description': order_asjc_descriptions})
            st.plotly_chart(fig_asjc, use_container_width=True)
        else:
            st.info("No ASJC Category data available for charting based on current filter.")

        st.markdown("---")

        # Chart 3: Publications per Source Title
        st.markdown("#### Publications per Source Title")
        # Aggregate based on selected metric
        if aggregation_func == 'nunique':
            source_counts = df_chart_data.dropna(subset=['Source title']).groupby('Source title')[metric_column].nunique().reset_index(name=metric_label)
        else: # 'sum' for 'Cited by'
            source_counts = df_chart_data.dropna(subset=['Source title']).groupby('Source title')[metric_column].sum().reset_index(name=metric_label)

        if not source_counts.empty:
            source_counts = source_counts.sort_values(by=metric_label, ascending=False)
            order_source_titles = source_counts['Source title'].tolist()
            fig_source = px.bar(source_counts, y='Source title', x=metric_label, # Flipped axes
                                 title=f'{metric_label} by Source Title',
                                 labels={'Source title': 'Source Title', metric_label: metric_label},
                                 template='plotly_white',
                                 category_orders={'Source title': order_source_titles})
            st.plotly_chart(fig_source, use_container_width=True)
        else:
            st.info("No Source Title data available for charting based on current filter.")
    else:
        st.info("Upload Scopus Export CSV(s) and apply filters to see publication analysis charts.")


# ========================
#     MAIN APPLICATION
# ========================

def main():
    st.set_page_config(
        page_title="Scopus & QS Journal Filter & Export Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Scopus & QS Journal Filter & Export Analysis")
    st.markdown("Upload your Scopus data files to filter journals and analyze publication exports by ASJC codes or QS Narrow Subject Areas.")

    # --- Sidebar for File Uploads ---
    st.sidebar.header("Upload Files")

    # Initialize session state for file upload status
    if 'scopus_uploaded' not in st.session_state:
        st.session_state.scopus_uploaded = False
    if 'qs_uploaded' not in st.session_state:
        st.session_state.qs_uploaded = False
    if 'export_uploaded' not in st.session_state: # New state for export CSV
        st.session_state.export_uploaded = False

    scopus_excel_file = st.sidebar.file_uploader(
        "1. Upload Scopus Source Excel (.xlsx)", type=["xlsx"], key="scopus_excel_upload",
        help="Upload the 'Scopus Source List' Excel file to filter journals."
    )

    df_source, df_asjc = pd.DataFrame(), pd.DataFrame()
    df_qs_subject = pd.DataFrame()
    df_export = pd.DataFrame() # New DataFrame for export data
    df_export_tagged = pd.DataFrame() # New DataFrame for tagged export data

    if scopus_excel_file:
        df_source, df_asjc = read_scopus_excel(scopus_excel_file)
        if not df_source.empty and not df_asjc.empty:
            st.sidebar.success("Scopus Source Excel loaded successfully!")
            st.session_state.scopus_uploaded = True
        else:
            st.sidebar.error("Failed to parse Scopus Source Excel. Please ensure it's a valid Scopus Source List.")
            st.session_state.scopus_uploaded = False
    else:
        st.sidebar.info("Please upload the Scopus Source Excel first.")
        st.session_state.scopus_uploaded = False # Reset if file is cleared

    # QS Subject file uploader, disabled until Scopus file is uploaded
    qs_subject_file = st.sidebar.file_uploader(
        "2. Upload QS Subject File (.csv or .xlsx)",
        type=["csv", "xlsx"],
        key="qs_subject_file_upload",
        help="Upload the QS subject area mapping file (e.g., 'qs-subject.xlsx - Sheet1.csv' or an Excel version).",
        disabled=not st.session_state.scopus_uploaded # This makes it disabled initially
    )

    if qs_subject_file:
        df_qs_subject = read_qs_subject_file(qs_subject_file)
        if not df_qs_subject.empty:
            st.sidebar.success("QS Subject file loaded successfully!")
            st.session_state.qs_uploaded = True
        else:
            st.sidebar.error("Failed to parse QS Subject file. Please ensure it's a valid CSV or Excel file with the correct columns.")
            st.session_state.qs_uploaded = False
    else:
        st.sidebar.info("Please upload the QS Subject file.")
        st.session_state.qs_uploaded = False # Reset if file is cleared

    # Scopus Export CSV(s) uploader, disabled until BOTH Source and QS files are uploaded
    scopus_export_files = st.sidebar.file_uploader(
        "3. Upload Scopus Export CSV(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="scopus_export_files_upload",
        help="Upload one or more Scopus export CSV files (e.g., your author's publication list).",
        disabled=not (st.session_state.scopus_uploaded and st.session_state.qs_uploaded) # Disabled until both previous are uploaded
    )

    if scopus_export_files:
        df_export, export_error = read_scopus_export_csv(scopus_export_files)
        if export_error:
            st.sidebar.error(export_error)
            st.session_state.export_uploaded = False
        elif not df_export.empty:
            st.sidebar.success(f"Successfully merged {len(scopus_export_files)} export CSV files ({len(df_export)} unique rows).")
            st.session_state.export_uploaded = True
            # Tag the export data immediately after successful upload
            if not df_source.empty and not df_asjc.empty and not df_qs_subject.empty:
                df_export_tagged = tag_export_with_asjc_and_qs(df_export, df_source, df_asjc, df_qs_subject)
                if not df_export_tagged.empty:
                    st.sidebar.success("Scopus Export data tagged with ASJC and QS subjects.")
                else:
                    st.sidebar.warning("Failed to tag Scopus Export data. Check file contents.")
            else:
                st.sidebar.warning("Source or QS file missing, cannot tag Scopus Export.")
        else:
            st.sidebar.info("No valid data found in uploaded Scopus Export CSVs.")
            st.session_state.export_uploaded = False
    else:
        st.sidebar.info("Please upload Scopus Export CSV(s).")
        st.session_state.export_uploaded = False


    # --- Main Content Area with Tabs ---
    tabs = st.tabs(["Filter by Scopus ASJC", "Filter by QS Subject", "Scopus Export Analysis"])

    with tabs[0]:
        section_scopus_asjc_filter(df_source, df_asjc)

    with tabs[1]:
        section_qs_subject_filter(df_source, df_asjc, df_qs_subject, st.session_state.qs_uploaded)

    with tabs[2]:
        section_scopus_export_analysis(df_export_tagged, df_qs_subject, st.session_state.export_uploaded)


if __name__ == "__main__":
    main()
