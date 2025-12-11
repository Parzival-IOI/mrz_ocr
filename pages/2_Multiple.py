import streamlit as st
import json
import time
import io
import pandas as pd
from xlsxwriter.utility import xl_col_to_name
from mrz import mrz

def submit_action(file_data):
    start_time = time.time()

    # mrz() now returns preprocessed image, raw text, and a parsed dict
    mrz_preprocessed, mrzText, parsed = mrz(file_data)

    execution_time = time.time() - start_time

    return mrz_preprocessed, mrzText, parsed, execution_time


st.title("Streamlit Offical Travel Document")
st.write("Application for detecting and extracting MRZ data from travel documents.")

# Initialize session state for collected data
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = None
if 'excel_buffer' not in st.session_state:
    st.session_state.excel_buffer = None

uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

if st.button("Submit"):
    if not uploaded_files:
        st.error("Please upload at least one file first")
    else:
        st.success(f"Processing {len(uploaded_files)} file(s)...")
        collected_data = []
        
        for idx, file_data in enumerate(uploaded_files, 1):
            
            mrz_preprocessed, mrzText, parsed, execution_time = submit_action(file_data)

            if mrzText is None or parsed is None:
                st.error(f"Could not extract MRZ data from {file_data.name}")
            else:
                hashes = parsed.get("hashes", {})
                def hash_valid(key):
                    return hashes.get(key, {}).get("valid")

                data = {
                    "File Name": file_data.name,
                    "MRZ Text": mrzText,
                    "Execution Time (seconds)": f"{execution_time:.2f}",
                    "Document Type": parsed.get("type"),
                    "Document Number": parsed.get("document_number"),
                    "Document Number Valid": hash_valid("document_number"),
                    "Issuing Country": parsed.get("issuing_country"),
                    "Nationality": parsed.get("nationality"),
                    "Birth Date (YYMMDD)": parsed.get("birth_date"),
                    "Birth Date Valid": hash_valid("birth_date"),
                    "Expiry Date (YYMMDD)": parsed.get("expiry_date"),
                    "Expiry Date Valid": hash_valid("expiry_date"),
                    "Sex": parsed.get("sex"),
                    "Family Name": parsed.get("family_name"),
                    "Given Names": parsed.get("given_names"),
                }

                # Only TD1 has optional fields 1 and 2
                if parsed.get("type") == "TD1":
                    data["Optional Data 1"] = parsed.get("optional_data_1")
                    data["Optional Data 2"] = parsed.get("optional_data_2")
                else:
                    data["Optional Data"] = parsed.get("optional_data")
                    data["Optional Data Valid"] = hash_valid("optional_data")
                collected_data.append(data)

        if collected_data:
            df = pd.DataFrame(collected_data)
            summary_cols = [
                "File Name",
                "Document Type",
                "Document Number",
                "Family Name",
                "Given Names",
                "Sex",
                "Birth Date (YYMMDD)",
                "Nationality",
                "Expiry Date (YYMMDD)",
            ]
            summary_df = df.reindex(columns=summary_cols)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                # Include helper validity columns in summary sheet for styling (hidden later)
                summary_cols_with_valid = summary_cols + [
                    "Document Number Valid",
                    "Birth Date Valid",
                    "Expiry Date Valid",
                ]
                summary_df_full = df.reindex(columns=summary_cols_with_valid)
                
                # Separate clean rows from error rows
                # Error rows: any validity is False OR document type is UNKNOWN
                def is_error_row(row):
                    if row.get("Document Type") == "UNKNOWN":
                        return True
                    if row.get("Document Number Valid") == False:
                        return True
                    if row.get("Birth Date Valid") == False:
                        return True
                    if row.get("Expiry Date Valid") == False:
                        return True
                    if row.get("Optional Data Valid") == False:
                        return True
                    return False
                
                error_mask = summary_df_full.apply(is_error_row, axis=1)
                error_df = summary_df_full[error_mask].copy()
                clean_df = summary_df_full[~error_mask].copy()
                
                # Write Summary sheet (only clean rows, no validity columns)
                clean_summary = clean_df[summary_cols]
                clean_summary.to_excel(writer, index=False, sheet_name="Summary")
                
                # Write Error sheet with full data including validity columns
                if len(error_df) > 0:
                    error_df.to_excel(writer, index=False, sheet_name="Error")
                
                # Write All Data sheet
                df.to_excel(writer, index=False, sheet_name="All Data")

                workbook = writer.book
                invalid_row_fmt = workbook.add_format(
                    {"bg_color": "#FF0000", "font_color": "white"}
                )
                invalid_strike_fmt = workbook.add_format(
                    {"font_color": "white", "font_strikeout": True, "bg_color": "#FF0000"}
                )
                unknown_row_fmt = workbook.add_format({"bg_color": "#FFF200"})

                def apply_row_highlight(ws, cols_order, validity_cols):
                    present_valids = [c for c in validity_cols if c in cols_order]
                    if not present_valids:
                        return
                    # Build OR formula across validity columns, excluding UNKNOWN documents
                    terms = [f"${xl_col_to_name(cols_order.index(vc))}" + "2=FALSE" for vc in present_valids]
                    if "Document Type" in cols_order:
                        c_doc = cols_order.index("Document Type")
                        doc_col_letter = xl_col_to_name(c_doc)
                        formula = f"=AND(${doc_col_letter}2<>\"UNKNOWN\",OR({','.join(terms)}))"
                    else:
                        formula = f"=OR({','.join(terms)})"
                    last_row = len(error_df) if ws.name == "Error" else len(df)
                    ws.conditional_format(
                        1,
                        0,
                        last_row,
                        len(cols_order) - 1,
                        {"type": "formula", "criteria": formula, "format": invalid_row_fmt},
                    )

                def apply_unknown(ws, cols_order, last_row):
                    if "Document Type" not in cols_order:
                        return
                    c_doc = cols_order.index("Document Type")
                    doc_col_letter = xl_col_to_name(c_doc)
                    ws.conditional_format(
                        1,
                        0,
                        last_row,
                        len(cols_order) - 1,
                        {
                            "type": "formula",
                            "criteria": f"=${doc_col_letter}2=\"UNKNOWN\"",
                            "format": unknown_row_fmt,
                        },
                    )

                def apply_cell_strike(ws, display_col, valid_col, cols_order, last_row):
                    if display_col not in cols_order or valid_col not in cols_order:
                        return
                    c_disp = cols_order.index(display_col)
                    c_valid = cols_order.index(valid_col)
                    valid_col_letter = xl_col_to_name(c_valid)
                    # Only strike-through if hash is false AND document type is NOT UNKNOWN
                    if "Document Type" in cols_order:
                        c_doc = cols_order.index("Document Type")
                        doc_col_letter = xl_col_to_name(c_doc)
                        criteria = f"=AND({doc_col_letter}2<>\"UNKNOWN\",{valid_col_letter}2=FALSE)"
                    else:
                        criteria = f"={valid_col_letter}2=FALSE"
                    ws.conditional_format(
                        1,
                        c_disp,
                        last_row,
                        c_disp,
                        {
                            "type": "formula",
                            "criteria": criteria,
                            "format": invalid_strike_fmt,
                        },
                    )

                # Apply formatting to Error sheet
                if len(error_df) > 0:
                    error_ws = writer.sheets["Error"]
                    error_cols = list(error_df.columns)
                    last_row_error = len(error_df)
                    
                    apply_unknown(error_ws, error_cols, last_row_error)
                    apply_row_highlight(error_ws, error_cols, [
                        "Document Number Valid",
                        "Birth Date Valid",
                        "Expiry Date Valid",
                        "Optional Data Valid",
                    ])
                    apply_cell_strike(error_ws, "Document Number", "Document Number Valid", error_cols, last_row_error)
                    apply_cell_strike(error_ws, "Birth Date (YYMMDD)", "Birth Date Valid", error_cols, last_row_error)
                    apply_cell_strike(error_ws, "Expiry Date (YYMMDD)", "Expiry Date Valid", error_cols, last_row_error)
                    apply_cell_strike(error_ws, "Optional Data", "Optional Data Valid", error_cols, last_row_error)
                    
                    # Hide validity helper columns in Error
                    for vc in ["Document Number Valid", "Birth Date Valid", "Expiry Date Valid", "Optional Data Valid"]:
                        if vc in error_cols:
                            idx = error_cols.index(vc)
                            error_ws.set_column(idx, idx, None, None, {"hidden": True})

                # Apply formatting to All Data sheet
                all_ws = writer.sheets["All Data"]
                all_cols = list(df.columns)
                last_row_all = len(df)
                apply_unknown(all_ws, all_cols, last_row_all)
                apply_row_highlight(all_ws, all_cols, [
                    "Document Number Valid",
                    "Birth Date Valid",
                    "Expiry Date Valid",
                    "Optional Data Valid",
                ])
                apply_cell_strike(all_ws, "Document Number", "Document Number Valid", all_cols, last_row_all)
                apply_cell_strike(all_ws, "Birth Date (YYMMDD)", "Birth Date Valid", all_cols, last_row_all)
                apply_cell_strike(all_ws, "Expiry Date (YYMMDD)", "Expiry Date Valid", all_cols, last_row_all)
                apply_cell_strike(all_ws, "Optional Data", "Optional Data Valid", all_cols, last_row_all)
            buffer.seek(0)
            
            # Store in session state
            st.session_state.collected_data = collected_data
            st.session_state.excel_buffer = buffer.getvalue()
            
        else:
            st.error("No documents were successfully processed")

# Display results if data exists in session state
if st.session_state.collected_data:
    st.download_button(
        label="Download Excel",
        data=st.session_state.excel_buffer,
        file_name="mrz_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.code(json.dumps(st.session_state.collected_data, indent=2), language="json")
