import streamlit as st
import pandas as pd
import re
import plotly.express as px # Import Plotly Express
from wordcloud import WordCloud # For word cloud generation
import matplotlib.pyplot as plt # For displaying word cloud
from wordcloud import STOPWORDS
from pyvis.network import Network
import community as community_louvain  # python-louvain
import networkx as nx
import streamlit.components.v1 as components
from itertools import combinations
import math
import io

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

def calculate_h_index(df_export):
    """Calculates the h-index for a given DataFrame of publications."""
    if 'Cited by' not in df_export.columns or df_export.empty:
        return 0

    # Ensure 'Cited by' is numeric, coerce errors, and fill NaN with 0
    citations = pd.to_numeric(df_export['Cited by'], errors='coerce').fillna(0)

    # Filter out publications with no citations
    citations = citations[citations > 0].sort_values(ascending=False).reset_index(drop=True)

    h_index = 0
    for i, citation_count in enumerate(citations):
        if citation_count >= (i + 1):
            h_index = i + 1
        else:
            break
    return h_index

def parse_author_name_components(full_name):
    """
    Parses 'Lastname, F. M.' or 'Firstname Lastname' into components.
    Returns (last_name, first_initial, middle_initial).
    """
    full_name = str(full_name).strip()
    if not full_name:
        return '', '', ''

    # Handle "Lastname, F. M." format
    if ',' in full_name:
        parts = [p.strip() for p in full_name.split(',', 1)]
        last_name = parts[0]
        given_names_str = parts[1] if len(parts) > 1 else ''

        # Extract initials. This regex is more robust for variations like "A.", "A H", "A. H."
        initials_matches = re.findall(r'\b([A-Za-z])(?:\.|$|\s)', given_names_str)

        first_initial = initials_matches[0].upper() if initials_matches else ''
        middle_initial = initials_matches[1].upper() if len(initials_matches) > 1 else ''
        return last_name, first_initial, middle_initial
    else: # Assume "Firstname [MiddleName] Lastname" or just "Firstname Lastname"
        parts = full_name.split()
        if not parts:
            return '', '', ''

        last_name = parts[-1]
        first_initial = parts[0][0].upper() if parts else ''
        middle_initial = ''
        if len(parts) > 2: # Check for middle name/initial
            # If the second part is just an initial, consider it middle initial
            if len(parts[1]) == 1 and parts[1].isalpha():
                middle_initial = parts[1].upper()
            elif len(parts) > 2 and len(parts[-2]) == 1 and parts[-2].isalpha(): # Handle "First Middle. Last"
                middle_initial = parts[-2].upper()
        return last_name, first_initial, middle_initial

def parse_short_author_name(short_name_str):
    """
    Parses "Lastname Initials" or "Initials. Lastname" format from 'Authors' column.
    Returns (last_name, initials_string_cleaned).
    E.g., "Abdullah A.H." -> ("Abdullah", "AH")
    E.g., "A.H. Abdullah" -> ("Abdullah", "AH")
    """
    short_name_str = str(short_name_str).strip()
    if not short_name_str:
        return '', ''

    parts = short_name_str.split()
    if not parts:
        return '', ''

    last_name = ''
    initials_cleaned = ''

    # Pattern to detect initials (e.g., A., A.H., AH)
    initials_pattern = re.compile(r'^[A-Za-z]\.?([A-Za-z]\.?)*$')

    # Heuristic: Try to identify initials vs last name based on pattern and position
    # If the last part looks like initials: assume "Lastname Initials"
    if initials_pattern.match(parts[-1].replace('.', '')):
        last_name = " ".join(parts[:-1]) # Everything before last part is last name
        initials_cleaned = parts[-1].replace('.', '')
    # If the first part looks like initials: assume "Initials Lastname"
    elif initials_pattern.match(parts[0].replace('.', '')):
        initials_cleaned = parts[0].replace('.', '')
        last_name = " ".join(parts[1:]) # Everything after first part is last name
    else: # Fallback: more complex or single-part names
        if len(parts) == 1:
            # If single part, it could be either a last name or just initials
            if initials_pattern.match(parts[0].replace('.', '')):
                initials_cleaned = parts[0].replace('.', '')
            else:
                last_name = parts[0]
        elif len(parts) > 1:
            # Attempt to split "Lastname, Initials" if comma present
            if ',' in short_name_str:
                split_by_comma = [p.strip() for p in short_name_str.split(',', 1)]
                if len(split_by_comma) == 2:
                    last_name = split_by_comma[0]
                    initials_cleaned = split_by_comma[1].replace('.', '')
            else: # Assume "Firstname Lastname" or "Lastname Initials" without clear pattern
                last_name = parts[-1] # Assume last part is last name
                initials_parts = parts[:-1]
                initials_cleaned = "".join([p[0] for p in initials_parts if p]).upper() # Get initials from first parts

    return last_name.strip(), initials_cleaned.upper().strip()


def unnest_authors(df):
    """
    Un-nests the 'Author full names' and 'Author(s) ID' columns,
    creating a new row for each author for a given paper.
    Also extracts Author IDs and robustly matches corresponding short names from 'Authors' column.
    """
    required_unnest_cols = ['Author full names', 'Author(s) ID']
    if not all(col in df.columns for col in required_unnest_cols):
        return pd.DataFrame()

    expanded_data = []

    for _, row in df.iterrows():
        author_full_names_list = [name.strip() for name in str(row['Author full names']).split(';') if name.strip()]
        author_ids_list = [id.strip() for id in str(row['Author(s) ID']).split(';') if id.strip()]
        author_short_names_list = [name.strip() for name in str(row.get('Authors', '')).split(';') if name.strip()]
        # NEW by THChew - Reverse order of short names like 'M.S. Abdurrahman'
        author_short_names_list = author_short_names_list + [' '.join(i.split()[::-1]) for i in author_short_names_list]


        # Create a mapping for short names based on derived last name and initials
        # This will help in matching full names to their corresponding short names
        short_name_derived_map = {}
        for sn_str in author_short_names_list:
            sn_last, sn_initials = parse_short_author_name(sn_str)
            if sn_last and sn_initials:
                # Key: lowercase_lastname_initials (e.g., "abdullah_ah")
                short_name_derived_map[f"{sn_last.lower()}_{sn_initials.lower()}"] = sn_str
            elif sn_last: # Fallback if initials cannot be clearly derived from short name
                short_name_derived_map[sn_last.lower()] = sn_str # Key: lowercase_lastname (e.g., "abdullah")


        for i, full_name_entry in enumerate(author_full_names_list):
            name_match = re.match(r"^(.*?)\s*\((\d+)\)$", full_name_entry.strip())
            parsed_full_name = name_match.group(1).strip() if name_match else full_name_entry.strip()
            parsed_id = name_match.group(2).strip() if name_match else (author_ids_list[i] if i < len(author_ids_list) else None)

            # Extract components from the parsed full name to find a match in short_name_derived_map
            full_name_last, full_name_first_initial, full_name_middle_initial = parse_author_name_components(parsed_full_name)
            derived_full_name_initials = (full_name_first_initial + full_name_middle_initial).upper()

            matched_short_name = None

            # Attempt 1: Match by last name + initials
            if full_name_last and derived_full_name_initials:
                key_with_initials = f"{full_name_last.lower()}_{derived_full_name_initials.lower()}"
                matched_short_name = short_name_derived_map.get(key_with_initials)

            # Attempt 2: Fallback to match only by last name
            if not matched_short_name and full_name_last:
                key_only_lastname = full_name_last.lower()
                matched_short_name = short_name_derived_map.get(key_only_lastname)

            # If still no match and author_short_names_list has enough entries, use index as last resort
            if not matched_short_name and i < len(author_short_names_list):
                matched_short_name = author_short_names_list[i]

            new_row = row.copy()
            new_row['Individual Author'] = parsed_full_name
            new_row['Individual Author Scopus ID'] = parsed_id
            new_row['Individual Author Short Name'] = matched_short_name # Assign the matched short name
            new_row['Is First Author'] = (i == 0)
            expanded_data.append(new_row)

    if not expanded_data:
        return pd.DataFrame()

    return pd.DataFrame(expanded_data)

def detect_corresponding_author(df):
    """
    Detects if an 'Individual Author' is the corresponding author based on the
    'Correspondence Address' column, using a simplified string matching.
    """
    required_cols = ['Correspondence Address', 'Individual Author', 'Individual Author Short Name']

    # Check if required columns are present, otherwise return with 'Is Corresponding Author' as False
    if not all(col in df.columns for col in required_cols):
        df['Is Corresponding Author'] = False
        return df

    df['Is Corresponding Author'] = False

    # Normalization function
    def normalize(s):
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', str(s).lower())).strip()

    for index, row in df.iterrows():
        correspondence_address_raw = str(row['Correspondence Address'])

        # If correspondence address is empty, set matched to False and continue
        if pd.isna(correspondence_address_raw) or not correspondence_address_raw.strip():
            df.at[index, 'Is Corresponding Author'] = False
            continue

        normalized_correspondence_address = normalize(correspondence_address_raw)
        individual_author_name = str(row['Individual Author'])
        individual_author_short_name = str(row['Individual Author Short Name']).strip()

        normalized_full_name = normalize(individual_author_name)
        normalized_short_name = normalize(individual_author_short_name)

        matched = False

        # Simplified match logic
        # Prioritize matching the short name, then fallback to full name
        if normalized_short_name and normalized_short_name in normalized_correspondence_address:
            matched = True
        elif normalized_full_name and normalized_full_name in normalized_correspondence_address:
            matched = True

        df.at[index, 'Is Corresponding Author'] = matched

    return df

@st.cache_data
def prepare_author_analysis_data(df_export_tagged_input):
    """
    Prepares the author-level DataFrame for analysis, including unnesting
    and corresponding author detection. This function is cached.
    """
    if df_export_tagged_input.empty:
        return pd.DataFrame()

    df_for_author_analysis = df_export_tagged_input.copy()

    # Perform unnesting
    df_for_author_analysis = unnest_authors(df_for_author_analysis)

    # Perform corresponding author detection if unnesting was successful and relevant columns exist
    # Ensure 'Individual Author Short Name' exists before passing to detect_corresponding_author
    if not df_for_author_analysis.empty and 'Individual Author Short Name' in df_for_author_analysis.columns:
        df_for_author_analysis = detect_corresponding_author(df_for_author_analysis)

    return df_for_author_analysis


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
            st.error("The uploaded QS Subject file does not contain a 'QS Subject Area' column. Please check your file.")
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
    This function now returns the publication-level data without unnesting authors.
    """
    if not files:
        return pd.DataFrame(), "No files provided."

    all_dfs = []
    error_messages = []
    # Required columns for basic export processing
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
            st.warning("EID column not found in export CSVs. Deduplication might be less precise.")
            merged_df.drop_duplicates(subset=['Title', 'Year', 'Source title', 'DOI'], inplace=True)

        return merged_df, None # Return publication-level data
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

@st.cache_data
def tag_export_with_asjc_and_qs(df_export, df_source, df_asjc, df_qs_subject):
    """
    Tags the Scopus Export DataFrame with ASJC codes and QS Subject Areas
    by looking up ISSN/EISSN in the Scopus Source data,
    following the specific order: df_export['ISSN'] -> df_source['ISSN'],
    then for unmatched, df_export['ISSN'] -> df_source['EISSN'].
    This function is now cached.
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

    # Clean ISSN/EISSN in df_tagged (already done in read_scopus_export_csv but good to be safe)
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
        df_tagged.loc[df_tagged['__matched_asjc_codes_raw'].isna(), '__matched_asjc_codes_raw'] = \
            df_tagged.loc[df_tagged['__matched_asjc_codes_raw'].isna(), 'ISSN'].map(source_issn_map)

    # Step 2: For records still unmatched, lookup df_export['ISSN'] against df_source['EISSN']
    # Identify rows where ASJC code is still missing after the first lookup
    # Only proceed if '__matched_asjc_codes_raw' exists and there are NaNs
    if '__matched_asjc_codes_raw' in df_tagged.columns:
        unmatched_mask = df_tagged['__matched_asjc_codes_raw'].isna()
        # Apply the EISSN lookup ONLY to the unmatched rows
        # Ensure ISSN column exists in df_tagged before mapping
        if 'EISSN' in df_tagged.columns:
            df_tagged.loc[unmatched_mask, '__matched_asjc_codes_raw'] = \
                df_tagged.loc[unmatched_mask, 'EISSN'].map(source_eissn_map)

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
                # Note: For download, still convert to string to prevent issues with CSV export
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

def section_basic_information(df_export_tagged):
    """UI for displaying basic information about the Scopus Export data."""
    st.subheader("Basic Information")

    if df_export_tagged.empty:
        st.info("Upload Scopus Export CSV(s) to see basic information.")
        return

    # Ensure 'Cited by' column is numeric for calculations
    if 'Cited by' in df_export_tagged.columns:
        df_export_tagged['Cited by'] = pd.to_numeric(df_export_tagged['Cited by'], errors='coerce').fillna(0)
    else:
        st.warning("The 'Cited by' column is not available in the uploaded export data. Citation calculations will not be available.")
        df_export_tagged['Cited by'] = 0 # Default to 0 if column missing

    col1, col2 = st.columns(2)

    # Calculate and display h-Index
    h_index = calculate_h_index(df_export_tagged)
    col1.metric(label="H-Index", value=h_index)

    # Calculate and display Total Citations
    total_citations = int(df_export_tagged['Cited by'].sum()) # Ensure integer display
    col2.metric(label="Total Citations", value=total_citations)

    st.markdown("---")
    st.markdown("#### Publications per Document Type")

    if 'Document Type' in df_export_tagged.columns:
        document_type_counts = df_export_tagged['Document Type'].value_counts().reset_index()
        document_type_counts.columns = ['Document Type', 'Count']

        if not document_type_counts.empty:
            fig_pie = px.pie(document_type_counts,
                             values='Count',
                             names='Document Type',
                             title='Distribution of Publications by Document Type',
                             template='plotly_white')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No 'Document Type' data available for charting.")
    else:
        st.warning("The 'Document Type' column is not available in the uploaded export data for pie chart analysis.")

    st.markdown("---")

### --- generate word cloud image ---
def generate_word_cloud_image(text_data: str):
    if not text_data or not text_data.strip():
        return None, "No keywords provided to generate a word cloud."
    try:
        from wordcloud import WordCloud, STOPWORDS
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            collocations=False,
            min_font_size=10,
            stopwords=STOPWORDS,  # optional but helpful
            # font_path="DejaVuSans.ttf",  # uncomment if you have non-Latin text
        ).generate(text_data)
        # Return an array/PIL image instead of a Matplotlib Figure
        return wc.to_array(), None
    except ValueError as e:
        return None, f"Word cloud error: {e}"

### --- broken ---
# @st.cache_data
# def generate_word_cloud(text_data):
#     """Generates and returns a word cloud image from the provided text."""
#     if not text_data:
#         return None, "No keywords provided to generate a word cloud."
#
#     wordcloud = WordCloud(width=800, height=400,
#                           background_color='white',
#                           collocations=False, # Set to False to prevent combining words
#                           min_font_size=10).generate(text_data)
#
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     return fig, None


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
        default=all_qs_subjects if select_all_export_qs else [], # Corrected variable name
        key="export_qs_subject_multiselect_tab3"
    )

    # -- DISABLED TEMPORARY --
    ## filter_export_button = st.button("Filter Export by QS Subject", key="filter_export_qs_btn_tab3")

    # Define columns to display for the export analysis at a broader scope
    export_display_cols = [
        "Title", "Source title", "Year", "ISSN", "EISSN", "DOI", "Cited by",
        "Matched_ASJC", "Matched_ASJC_Description", "Matched_QS_Subject_Area"
    ]

    ## -- New 22-Aug-2025
    def _filter_by_qs(df, subjects):
        if not subjects:
            return df.copy()
        return df[df['Matched_QS_Subject_Area'].apply(
            lambda x: any(qs in subjects for qs in x) if isinstance(x, list) else False
        )].copy()

    # Always derive df_chart_data from the selection (no button)
    df_chart_data = _filter_by_qs(df_export_tagged, selected_export_qs_subjects)

    st.write(
        f"Displaying publications for current QS selection ({len(df_chart_data)})"
        if selected_export_qs_subjects else
        "Displaying all uploaded and tagged publications:"
    )

    existing_export_display_cols = [c for c in export_display_cols if c in df_chart_data.columns]
    st.dataframe(df_chart_data[existing_export_display_cols], use_container_width=True)

    # df_chart_data = pd.DataFrame() # Initialize an empty DataFrame
    # existing_export_display_cols = [] # Initialize here as well
    #
    # # Determine which DataFrame to use for display and charting
    # # If the filter button is clicked and subjects are selected, use df_filtered_export
    # # Otherwise, use the full df_export_tagged
    # if filter_export_button and selected_export_qs_subjects:
    #     # Filter the already tagged df_export_tagged DataFrame
    #     df_filtered_export = df_export_tagged[
    #         df_export_tagged['Matched_QS_Subject_Area'].apply(
    #             lambda x: any(qs in selected_export_qs_subjects for qs in x) if isinstance(x, list) else False
    #         )
    #     ].copy()
    #     df_chart_data = df_filtered_export # Use filtered data for charts and table
    #     st.write(f"Publications matching selected QS Subject Areas ({len(df_filtered_export)}):")
    #
    #     # Filter display_cols to only include columns that actually exist in df_filtered_export
    #     existing_export_display_cols = [col for col in export_display_cols if col in df_filtered_export.columns]
    #
    #     st.dataframe(df_filtered_export[existing_export_display_cols], use_container_width=True)
    #
    #     if not df_filtered_export.empty:
    #         df_to_download_export = df_filtered_export[existing_export_display_cols].copy()
    #         for col_list in ["Matched_ASJC", "Matched_ASJC_Description", "Matched_QS_Subject_Area"]:
    #             if col_list in df_to_download_export.columns:
    #                 df_to_download_export[col_list] = df_to_download_export[col_list].apply(
    #                     lambda x: "; ".join(map(str, x)) if isinstance(x, list) else (str(x) if pd.notna(x) else "")
    #                 )
    #         csv_export = df_to_download_export.to_csv(index=False).encode('utf-8')
    #         st.download_button(
    #             label="Download Filtered Scopus Export Data",
    #             data=csv_export,
    #             file_name="filtered_scopus_export_data.csv",
    #             mime="text/csv",
    #             key="download_export_qs_btn_tab3"
    #         )
    # elif df_export_tagged is not None and not df_export_tagged.empty:
    #     df_chart_data = df_export_tagged # Use full data for charts and table if no filter applied
    #     st.write("Displaying all uploaded and tagged publications:")
    #
    #     existing_export_display_cols = [col for col in export_display_cols if col in df_export_tagged.columns]
    #     st.dataframe(df_export_tagged[existing_export_display_cols], use_container_width=True)
    # else:
    #     st.info("Select one or more QS Narrow Subject Areas to filter your uploaded Scopus Export data.")

    # Call the new section for basic information
    section_basic_information(df_chart_data)

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
        st.markdown("#### Publications per QS Subject")
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

    # --- Authors in current QS selection ---
    st.markdown("---")
    st.subheader("Authors (based on current QS Subject selection)")

    if df_chart_data.empty:
        st.info("Upload Scopus Export CSV(s) and apply QS filters to list authors.")
    else:
        required_cols = ['Author full names', 'Author(s) ID']
        if not all(c in df_chart_data.columns for c in required_cols):
            st.warning("Author columns not found. Expected 'Author full names' and 'Author(s) ID'.")
        else:
            # 1) Unnest authors (your existing function)
            df_authors_long = unnest_authors(df_chart_data)

            if df_authors_long.empty:
                st.info("No author records found after applying the QS filter.")
            else:
                # Optional: normalize name whitespace
                df_authors_long['Individual Author'] = (
                    df_authors_long['Individual Author']
                    .astype(str)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )

                # 2) Deduplicate by (paper, author id) to avoid double-counting publications
                if 'EID' in df_authors_long.columns:
                    df_authors_long = df_authors_long.drop_duplicates(
                        subset=['EID', 'Individual Author Scopus ID']
                    )

                # Keep only rows with valid ID and some name
                df_authors_long = df_authors_long.dropna(
                    subset=['Individual Author Scopus ID']
                )
                # If name missing, fill blank to avoid NaNs in grouping
                df_authors_long['Individual Author'] = df_authors_long['Individual Author'].fillna("")

                # 3) Count publications per Author ID
                pub_counts = (
                    df_authors_long
                    .groupby('Individual Author Scopus ID', as_index=False)
                    .size()
                    .rename(columns={'size': 'Count of Publication'})
                )

                # 4) Name frequency per (Author ID, Name) – case-insensitive bucket for fairness
                #    We still keep the original-cased version for display preference later.
                df_names = df_authors_long.assign(
                    _name_key=df_authors_long['Individual Author'].str.casefold()
                )

                name_counts = (
                    df_names
                    .groupby(['Individual Author Scopus ID', '_name_key'], as_index=False)
                    .agg(
                        freq=('Individual Author', 'size'),
                        # Store the most common original-cased variant for display
                        # (longest among that _name_key, then lexicographic)
                        display_name=('Individual Author', lambda s: sorted(s, key=lambda x: (len(x), x), reverse=True)[0])
                    )
                )

                # 5) Pick best display name per Author ID:
                #    - highest freq
                #    - if tie, longest display_name
                #    - if still tie, lexicographic
                name_counts = name_counts.copy()
                name_counts['display_name'] = name_counts['display_name'].fillna("")
                name_counts['_name_len'] = name_counts['display_name'].str.len()

                best_names = (
                    name_counts
                    .sort_values(
                        by=['Individual Author Scopus ID', 'freq', '_name_len', 'display_name'],
                        ascending=[True, False, False, True]
                    )
                    .drop_duplicates(subset=['Individual Author Scopus ID'])
                    .loc[:, ['Individual Author Scopus ID', 'display_name']]
                    .rename(columns={'display_name': 'Name'})
                )

                # 6) Join counts + best name
                authors_summary = (
                    pub_counts
                    .merge(best_names, on='Individual Author Scopus ID', how='left')
                    .rename(columns={'Individual Author Scopus ID': 'Author ID'})
                    .sort_values('Count of Publication', ascending=False)
                    .reset_index(drop=True)
                )

                st.dataframe(authors_summary[['Author ID', 'Name', 'Count of Publication']], use_container_width=True)

    # ===================== Author Collaboration Network (with communities) =====================
    st.markdown("---")
    st.subheader("Author Collaboration Network (based on current QS selection)")

    # Controls
    colA, colB, colC, colD = st.columns(4)
    with colA:
        min_pubs = st.number_input("Min publications per author", min_value=1, value=2, step=1)
    with colB:
        min_collabs = st.number_input("Min collaborations per pair (unique EIDs)", min_value=1, value=1, step=1)
    with colC:
        max_authors = st.number_input("Max authors to display", min_value=20, value=300, step=10)
    with colD:
        chart_h = st.number_input("Chart height (px)", min_value=400, value=900, step=50)

    freeze_after_stabilize = st.checkbox("Freeze layout after stabilize", value=True)

    if df_chart_data.empty:
        st.info("Upload Scopus Export CSV(s) and apply QS filters to see the collaboration network.")
    else:
        required_cols = ['Author full names', 'Author(s) ID', 'EID']
        if not all(c in df_chart_data.columns for c in required_cols):
            st.warning("Expected columns not found: 'Author full names', 'Author(s) ID', 'EID'.")
        else:
            try:

                # ---------- Helper: pick best display name per Author ID ----------
                def pick_best_names(df_authors_long: pd.DataFrame) -> pd.DataFrame:
                    # Normalize
                    df = df_authors_long.copy()
                    df['Individual Author'] = (
                        df['Individual Author'].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
                    )
                    df['Individual Author Scopus ID'] = df['Individual Author Scopus ID'].astype(str).str.strip()

                    # Case-insensitive name bucket, keep best original-cased as display
                    df_names = df.assign(_name_key=df['Individual Author'].str.casefold())
                    name_counts = (
                        df_names
                        .groupby(['Individual Author Scopus ID', '_name_key'], as_index=False)
                        .agg(
                            freq=('Individual Author', 'size'),
                            display_name=('Individual Author',
                                          lambda s: sorted(s, key=lambda x: (len(x), x), reverse=True)[0])
                        )
                    )
                    name_counts['display_name'] = name_counts['display_name'].fillna("")
                    name_counts['_name_len'] = name_counts['display_name'].str.len()

                    best = (
                        name_counts
                        .sort_values(
                            by=['Individual Author Scopus ID', 'freq', '_name_len', 'display_name'],
                            ascending=[True, False, False, True]
                        )
                        .drop_duplicates(subset=['Individual Author Scopus ID'])
                        .loc[:, ['Individual Author Scopus ID', 'display_name']]
                        .rename(columns={'display_name': 'Name'})
                    )
                    return best

                # ---------- 1) Unnest authors (reuse your method) ----------
                df_authors_long = unnest_authors(df_chart_data)
                if df_authors_long.empty:
                    st.info("No author records after applying the QS filter.")
                else:
                    # Deduplicate by (EID, AuthorID) to avoid intra-paper duplicates
                    df_authors_long = df_authors_long.dropna(subset=['Individual Author Scopus ID', 'EID'])
                    df_authors_long['Individual Author Scopus ID'] = df_authors_long['Individual Author Scopus ID'].astype(str)
                    df_authors_long = df_authors_long.drop_duplicates(subset=['EID', 'Individual Author Scopus ID'])

                    # ---------- 2) Publications per author; filter & cap ----------
                    pub_counts = (
                        df_authors_long
                        .groupby('Individual Author Scopus ID', as_index=False)
                        .size()
                        .rename(columns={'size': 'Count of Publication'})
                    )
                    # Keep authors meeting min pubs
                    eligible_authors = pub_counts[pub_counts['Count of Publication'] >= int(min_pubs)].copy()

                    if eligible_authors.empty:
                        st.info("No authors meet the min publications filter.")
                    else:
                        # Name per ID
                        best_names = pick_best_names(df_authors_long)
                        eligible_authors = (
                            eligible_authors
                            .merge(best_names, on='Individual Author Scopus ID', how='left')
                            .sort_values('Count of Publication', ascending=False)
                            .head(int(max_authors))
                        )
                        keep_ids = set(eligible_authors['Individual Author Scopus ID'])

                        # ---------- 3) Build co-author pairs per EID (unique) ----------
                        pairs = []
                        for eid, grp in df_authors_long.groupby('EID'):
                            ids = sorted({a for a in grp['Individual Author Scopus ID'] if a in keep_ids})
                            if len(ids) > 1:
                                pairs.extend([tuple(sorted(p)) for p in combinations(ids, 2)])

                        if not pairs:
                            st.info("No co-authorship pairs under current filters.")
                        else:
                            edges_df = pd.Series(pairs, name='pair').to_frame()
                            edges_df[['AuthorID_A', 'AuthorID_B']] = pd.DataFrame(edges_df['pair'].tolist(), index=edges_df.index)
                            edges_df = (
                                edges_df
                                .drop(columns=['pair'])
                                .groupby(['AuthorID_A', 'AuthorID_B'], as_index=False)
                                .size()
                                .rename(columns={'size': 'Collab Count (unique EIDs)'})
                            )
                            # Edge filter
                            edges_df = edges_df[edges_df['Collab Count (unique EIDs)'] >= int(min_collabs)]

                            # ---------- 4) Build NetworkX graph ----------
                            G = nx.Graph()
                            # add nodes (even isolated, so they show up)
                            for _, r in eligible_authors.iterrows():
                                aid = r['Individual Author Scopus ID']
                                G.add_node(aid, pubs=int(r['Count of Publication']), name=r.get('Name') or aid)
                            # add edges with weights
                            for _, e in edges_df.iterrows():
                                a, b, w = e['AuthorID_A'], e['AuthorID_B'], int(e['Collab Count (unique EIDs)'])
                                if a in keep_ids and b in keep_ids:
                                    G.add_edge(a, b, weight=w)

                            # ---------- 5) Community detection ----------
                            partition = {}
                            try:
                                if G.number_of_edges() > 0:
                                    partition = community_louvain.best_partition(G)
                                else:
                                    partition = {n: 0 for n in G.nodes()}
                            except Exception:
                                # fallback: greedy modularity (communities as sets)
                                from networkx.algorithms.community import greedy_modularity_communities
                                if G.number_of_edges() > 0 and G.number_of_nodes() > 0:
                                    comms = list(greedy_modularity_communities(G))
                                    for cid, nodeset in enumerate(comms):
                                        for n in nodeset:
                                            partition[n] = cid
                                else:
                                    partition = {n: 0 for n in G.nodes()}

                            # ---------- 6) Color map by community ----------
                            # 20-color palette; cycle if needed
                            import matplotlib.cm as cm
                            palette = cm.tab20.colors
                            def comm_color(cid: int) -> str:
                                r, g, b = palette[cid % len(palette)]
                                return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

                            # ---------- 7) Build pyvis network ----------
                            net = Network(height=f"{int(chart_h)}px", width="100%", bgcolor="#ffffff", directed=False)

                            # Physics / overlap reduction
                            net.set_edge_smooth('false')
                            net.barnes_hut(
                                gravity=-5000,
                                central_gravity=0.1,
                                spring_length=200,
                                spring_strength=0.001
                            )
                            # net.set_options("""
                            # {
                            #   "physics": {
                            #     "solver": "barnesHut",
                            #     "stabilization": { "enabled": true, "iterations": 800, "fit": true },
                            #     "minVelocity": 1.2,
                            #     "centralGravity": 0.08,
                            #     "barnesHut": { "springLength": 240, "springConstant": 0.0015, "damping": 0.92, "avoidOverlap": 1 }
                            #   },
                            #   "interaction": {
                            #     "hover": true,
                            #     "selectConnectedEdges": true,
                            #     "hoverConnectedEdges": true
                            #   }
                            # }
                            # """)

                            # Optional freeze
                            if freeze_after_stabilize:
                                net.set_options('{"physics": {"enabled": false}}')

                            # Add nodes
                            meta = eligible_authors.set_index('Individual Author Scopus ID').to_dict('index')
                            for n in G.nodes():
                                pubs = meta.get(n, {}).get('Count of Publication', 1)
                                name = meta.get(n, {}).get('Name', n)
                                cid = partition.get(n, 0)
                                color = comm_color(cid)
                                size = 12 + 4 * math.sqrt(pubs)  # smooth scaling
                                title = f"{name}<br>Author ID: {n}<br>Publications: {pubs}<br>Community: {cid}"
                                net.add_node(n, label=name, title=title, size=size, color=color, group=cid, font={'size': 18})

                            # Add edges (width/value = unique EID count)
                            for u, v, data in G.edges(data=True):
                                w = int(data.get('weight', 1))
                                net.add_edge(u, v, value=w, width=w, title=f"Co-authored unique EIDs: {w}")

                            # Render (no disk I/O)
                            html_content = net.generate_html()
                            components.html(html_content, height=int(chart_h) + 50, width=int(chart_h) + 50, scrolling=True)

            except ModuleNotFoundError as m:
                missing = str(m).split("'")[-2] if "'" in str(m) else str(m)
                st.error(f"Missing dependency: `{missing}`. Please install it (e.g., `pip install pyvis networkx python-louvain`).")
            except Exception as ex:
                st.exception(ex)
    # =================== end Author Collaboration Network ====================


    # --- Keywords Analysis Section ---
    st.markdown("---")
    st.subheader("Keywords Analysis")

    if df_chart_data.empty:
        st.info("Upload Scopus Export CSV(s) and apply filters to analyze keywords.")
        return

    col_author_kw, col_index_kw = st.columns(2)
    use_author_keywords = col_author_kw.checkbox("Include Author Keywords", value=True, key="include_author_keywords")
    use_index_keywords = col_index_kw.checkbox("Include Index Keywords", value=True, key="include_index_keywords")

    generate_wordcloud_button = st.button("Generate Word Cloud", key="generate_wordcloud_btn")

    if generate_wordcloud_button:
        all_keywords_text = []

        if use_author_keywords and 'Author Keywords' in df_chart_data.columns:
            # Join all author keywords, handling NaN/empty strings and splitting by semicolons
            author_keywords = df_chart_data['Author Keywords'].dropna().astype(str).str.replace(";", " ").str.lower().tolist()
            all_keywords_text.extend(author_keywords)
        elif use_author_keywords:
            st.warning("Author Keywords column not found in your data.")

        if use_index_keywords and 'Index Keywords' in df_chart_data.columns:
            # Join all index keywords, handling NaN/empty strings and splitting by semicolons
            index_keywords = df_chart_data['Index Keywords'].dropna().astype(str).str.replace(";", " ").str.lower().tolist()
            all_keywords_text.extend(index_keywords)
        elif use_index_keywords:
            st.warning("Index Keywords column not found in your data.")

        combined_keywords_string = " ".join(all_keywords_text)

        if combined_keywords_string:

            wc_img, wc_error = generate_word_cloud_image(combined_keywords_string)

            if wc_img is not None:
                # <-- Use st.image (bypasses the pyplot->PIL pipeline that is erroring)
                st.image(
                    wc_img,
                    channels="RGB",          # wordcloud returns HxWx3
                    output_format="PNG",     # avoid the buggy JPEG save path
                    use_container_width=True
                )
            elif wc_error:
                st.error(wc_error)

        else:
            st.info("No keywords selected or found in the filtered data to generate a word cloud.")

    st.markdown("---")


def section_author_analysis(df_export_for_author_analysis, export_file_uploaded, df_qs_subject):
    """UI for author-specific analysis."""
    st.subheader("Author Analysis")

    if df_export_for_author_analysis.empty or not export_file_uploaded:
        st.warning("This functionality is not available until the Scopus Export CSV(s) have been uploaded and processed for authors.")
        return

    # Check if author data is un-nested
    if 'Individual Author' not in df_export_for_author_analysis.columns or 'Individual Author Scopus ID' not in df_export_for_author_analysis.columns:
        st.info("Author data needs to be un-nested from 'Author full names' to perform author analysis. Please ensure 'Author full names' and 'Author(s) ID' columns are present in your export file.")
        return

    # Create unique author entries for the selectbox (Name (Scopus ID))
    unique_authors = df_export_for_author_analysis[['Individual Author', 'Individual Author Scopus ID']].dropna().drop_duplicates()

    # Create display options and a mapping to their Scopus IDs
    author_display_options = [""] + sorted([
        f"{row['Individual Author']} ({row['Individual Author Scopus ID']})"
        for index, row in unique_authors.iterrows()
    ])

    display_to_id_map = {
        f"{row['Individual Author']} ({row['Individual Author Scopus ID']})": row['Individual Author Scopus ID']
        for index, row in unique_authors.iterrows()
    }

    selected_display_name = st.selectbox(
        "Select an Author:",
        options=author_display_options,
        key="author_select"
    )

    selected_author_id = None
    if selected_display_name:
        selected_author_id = display_to_id_map.get(selected_display_name)
        st.write(f"Analyzing data for: **{selected_display_name}**")

        # Filter the DataFrame for the selected author using Scopus ID
        df_author_filtered = df_export_for_author_analysis[
            df_export_for_author_analysis['Individual Author Scopus ID'] == selected_author_id
        ].copy()

        if df_author_filtered.empty:
            st.info(f"No publications found for {selected_display_name}.")
            return

        st.markdown("---")
        st.markdown("#### Filter by Author Role") # Changed title for clarity

        col_first, col_corr, col_co = st.columns(3)

        # Checkboxes for author roles
        is_first_author_selected = col_first.checkbox("First Author", value=True, key="filter_first_author")
        is_corresponding_author_selected = col_corr.checkbox("Corresponding Author", value=True, key="filter_corresponding_author")
        is_co_author_selected = col_co.checkbox("Co-Author", value=True, key="filter_co_author")

        role_filter_applied = False
        filtered_by_role = pd.DataFrame()

        # Apply role filters based on checkbox selection
        if is_first_author_selected and 'Is First Author' in df_author_filtered.columns:
            filtered_by_role = pd.concat([filtered_by_role, df_author_filtered[df_author_filtered['Is First Author'] == True]])
            role_filter_applied = True

        if is_corresponding_author_selected and 'Is Corresponding Author' in df_author_filtered.columns:
            filtered_by_role = pd.concat([filtered_by_role, df_author_filtered[df_author_filtered['Is Corresponding Author'] == True]])
            role_filter_applied = True

        if is_co_author_selected:
            # Co-author logic: not first and not corresponding
            if 'Is First Author' in df_author_filtered.columns and 'Is Corresponding Author' in df_author_filtered.columns:
                co_author_df = df_author_filtered[
                    (df_author_filtered['Is First Author'] == False) &
                    (df_author_filtered['Is Corresponding Author'] == False)
                ]
            elif 'Is First Author' in df_author_filtered.columns: # Only first author available
                co_author_df = df_author_filtered[df_author_filtered['Is First Author'] == False]
            elif 'Is Corresponding Author' in df_author_filtered.columns: # Only corresponding author available
                 co_author_df = df_author_filtered[df_author_filtered['Is Corresponding Author'] == False]
            else: # Neither available, all are treated as co-authors
                co_author_df = df_author_filtered.copy()

            filtered_by_role = pd.concat([filtered_by_role, co_author_df])
            role_filter_applied = True

        # Deduplicate publications based on EID after concatenation of roles
        if role_filter_applied and not filtered_by_role.empty:
            if 'EID' in filtered_by_role.columns:
                filtered_by_role.drop_duplicates(subset=['EID'], inplace=True)
            else:
                filtered_by_role.drop_duplicates(inplace=True) # Fallback deduplication

        if not role_filter_applied:
            st.info("Please select at least one author role to filter.")
            # If no role is selected, or if no data after role filter, ensure subsequent data is empty
            filtered_by_role = pd.DataFrame()
        elif filtered_by_role.empty:
            st.info(f"No publications found for {selected_display_name} matching the selected author roles.")

        # --- New QS Subject Filter for Author Analysis ---
        st.markdown("---")
        st.markdown("#### Filter by QS Subject Area")

        if df_qs_subject.empty:
            st.warning("QS Subject file is not loaded. Please upload it to enable this filter.")
            qs_subject_filter_df = filtered_by_role.copy() # No filtering possible
        elif filtered_by_role.empty:
            st.info("No publications left after role filter to apply QS Subject filter.")
            qs_subject_filter_df = pd.DataFrame()
        else:
            all_qs_subjects = sorted(df_qs_subject['QS Subject Area'].unique())
            select_all_author_qs = st.checkbox("Select All QS Subject Areas for Author Analysis", key="select_all_author_qs")

            selected_author_qs_subjects = st.multiselect(
                "Select QS Narrow Subject Areas for Author's Publications:",
                options=all_qs_subjects,
                default=all_qs_subjects if select_all_author_qs else [],
                key="author_qs_subject_multiselect"
            )

            filter_qs_button = st.button("Apply QS Subject Filter (Author)", key="filter_author_qs_btn")

            qs_subject_filter_df = filtered_by_role.copy() # Start with role-filtered data

            if filter_qs_button and selected_author_qs_subjects:
                qs_subject_filter_df = filtered_by_role[
                    filtered_by_role['Matched_QS_Subject_Area'].apply(
                        lambda x: any(qs in selected_author_qs_subjects for qs in x) if isinstance(x, list) else False
                    )
                ].copy()
                if qs_subject_filter_df.empty:
                    st.info(f"No publications found for {selected_display_name} matching the selected QS Subject Areas and roles.")
                else:
                    st.write(f"Publications also matching selected QS Subject Areas ({len(qs_subject_filter_df)}):")
            elif filter_qs_button and not selected_author_qs_subjects:
                 st.warning("Please select at least one QS Narrow Subject Area before applying QS filter.")
                 qs_subject_filter_df = pd.DataFrame() # No selection means no data after filter
            else:
                st.info("Select QS Narrow Subject Areas and click 'Apply QS Subject Filter' to filter.")

        # Determine the final DataFrame for display and metrics after all filters
        final_display_df = qs_subject_filter_df.copy()

        if final_display_df.empty:
            st.info("No publications to display after applying all filters.")
            return # Exit function if no data

        st.markdown("---")
        st.markdown("##### Summary for Selected Roles and QS Subjects")
        col_pub_count, col_cite_count, col_h_index = st.columns(3)

        # Use unique EID for publication count
        num_publications = len(final_display_df['EID'].unique()) if 'EID' in final_display_df.columns else len(final_display_df)
        col_pub_count.metric(label="Number of Publications", value=num_publications)

        # Ensure 'Cited by' column is numeric for calculations
        if 'Cited by' in final_display_df.columns:
            final_display_df['Cited by'] = pd.to_numeric(final_display_df['Cited by'], errors='coerce').fillna(0)
            total_citations = int(final_display_df['Cited by'].sum())
            h_index_author = calculate_h_index(final_display_df)
        else:
            total_citations = 0
            h_index_author = 0
            st.warning("The 'Cited by' column is not available for citation metrics.")

        col_cite_count.metric(label="Total Citations", value=total_citations)
        col_h_index.metric(label="H-Index", value=h_index_author)
        st.markdown("---")

        # Display columns for the filtered author data, including ASJC and QS
        author_display_cols = [
            "Title", "Source title", "Year", "Cited by", "Document Type",
            "Individual Author", "Is First Author", "Is Corresponding Author",
            "Matched_ASJC", "Matched_ASJC_Description", "Matched_QS_Subject_Area"
        ]

        # Filter to only include columns that actually exist in the dataframe
        existing_author_display_cols = [col for col in author_display_cols if col in final_display_df.columns]

        st.dataframe(final_display_df[existing_author_display_cols], use_container_width=True)

        # For download, still convert to string to prevent issues with CSV export
        df_to_download_export = final_display_df[existing_author_display_cols].copy()
        for col_list in ["Matched_ASJC", "Matched_ASJC_Description", "Matched_QS_Subject_Area"]:
            if col_list in df_to_download_export.columns:
                df_to_download_export[col_list] = df_to_download_export[col_list].apply(
                    lambda x: "; ".join(map(str, x)) if isinstance(x, list) else (str(x) if pd.notna(x) else "")
                )

        csv_author_filtered = df_to_download_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Filtered Data for {selected_display_name}",
            data=csv_author_filtered,
            file_name=f"{selected_display_name}_filtered_publications.csv",
            mime="text/csv",
            key="download_author_filtered_btn"
        )

    else:
        st.info("Select an author from the dropdown to perform analysis.")

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

    # Initialize df_export_tagged and df_export_for_author_analysis in session state
    if 'df_export_tagged' not in st.session_state:
        st.session_state.df_export_tagged = pd.DataFrame()
    if 'df_export_for_author_analysis' not in st.session_state:
        st.session_state.df_export_for_author_analysis = pd.DataFrame()


    scopus_excel_file = st.sidebar.file_uploader(
        "1. Upload Scopus Source Excel (.xlsx)", type=["xlsx"], key="scopus_excel_upload",
        help="Upload the 'Scopus Source List' Excel file to filter journals."
    )

    df_source, df_asjc = pd.DataFrame(), pd.DataFrame()
    df_qs_subject = pd.DataFrame()
    df_export = pd.DataFrame() # New DataFrame for export data

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
        # Read the export CSVs (publication-level)
        df_export, export_error = read_scopus_export_csv(scopus_export_files)
        if export_error:
            st.sidebar.error(export_error)
            st.session_state.export_uploaded = False
        elif not df_export.empty:
            st.sidebar.success(f"Successfully merged {len(scopus_export_files)} export CSV files ({len(df_export)} unique rows).")
            st.session_state.export_uploaded = True

            # Tag the export data (publication-level), uses cached tag_export_with_asjc_and_qs
            if not df_source.empty and not df_asjc.empty and not df_qs_subject.empty:
                st.session_state.df_export_tagged = tag_export_with_asjc_and_qs(df_export, df_source, df_asjc, df_qs_subject)
                if not st.session_state.df_export_tagged.empty:
                    st.sidebar.success("Scopus Export data tagged with ASJC and QS subjects.")

                    # Prepare the author-specific data using the cached prepare_author_analysis_data
                    st.session_state.df_export_for_author_analysis = prepare_author_analysis_data(st.session_state.df_export_tagged)
                    if not st.session_state.df_export_for_author_analysis.empty:
                        st.sidebar.success("Author-specific data prepared for analysis.")
                    else:
                        st.sidebar.warning("Could not prepare author-specific data. Check 'Author full names' and 'Author(s) ID' columns in export files.")
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

    # Button to clear Streamlit's cache for all data processing functions
    st.sidebar.markdown("---")
    st.sidebar.header("Cache Control")
    if st.sidebar.button("Clear Cache for All Data"):
        st.cache_data.clear()
        st.session_state.scopus_uploaded = False
        st.session_state.qs_uploaded = False
        st.session_state.export_uploaded = False
        st.session_state.df_export_tagged = pd.DataFrame()
        st.session_state.df_export_for_author_analysis = pd.DataFrame()
        st.sidebar.success("All cached data cleared. Please re-upload files.")
        st.rerun() # Rerun the app to reflect changes


    # --- Main Content Area with Tabs ---
    tabs = st.tabs(["Filter by Scopus ASJC", "Filter by QS Subject", "Scopus Export Analysis", "Author Analysis"])

    with tabs[0]:
        section_scopus_asjc_filter(df_source, df_asjc)

    with tabs[1]:
        section_qs_subject_filter(df_source, df_asjc, df_qs_subject, st.session_state.qs_uploaded)

    with tabs[2]:
        # Pass the publication-level tagged data for Scopus Export Analysis
        section_scopus_export_analysis(st.session_state.df_export_tagged, df_qs_subject, st.session_state.export_uploaded)

    with tabs[3]:
        # Pass the author-level unnested data and df_qs_subject for Author Analysis
        section_author_analysis(st.session_state.df_export_for_author_analysis, st.session_state.export_uploaded, df_qs_subject)


if __name__ == "__main__":
    main()
