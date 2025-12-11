import streamlit as st
import json
import time
from mrz import mrz

def submit_action(file_data):
    start_time = time.time()

    # mrz() now returns preprocessed image, raw text, and a parsed dict
    mrz_preprocessed, mrzText, parsed = mrz(file_data)

    execution_time = time.time() - start_time

    return mrz_preprocessed, mrzText, parsed, execution_time


st.title("Streamlit Offical Travel Document")
st.write("Application for detecting and extracting MRZ data from travel documents.")

file_data = st.file_uploader("Upload a file")

if st.button("Submit"):
    if file_data is None:
        st.error("Please upload a file first")
    else:
        mrz_preprocessed, mrzText, parsed, execution_time = submit_action(file_data)

        if mrzText is None or parsed is None:
            st.error("Could not extract MRZ data from the image")
        else:
            st.write("MRZ Text")
            st.markdown(f""">{mrzText}""")

            st.info(f"⏱️ Execution Time: {execution_time:.2f} seconds")

            data = {
                "Document Type": parsed.get("type"),
                "Document Number": parsed.get("document_number"),
                "Issuing Country": parsed.get("issuing_country"),
                "Nationality": parsed.get("nationality"),
                "Birth Date (YYMMDD)": parsed.get("birth_date"),
                "Expiry Date (YYMMDD)": parsed.get("expiry_date"),
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

            st.markdown("### Extracted Data")
            st.code(json.dumps(data, indent=2), language="json")

            hashes = parsed.get("hashes")
            if hashes:
                st.markdown("### Hash Checks")
                st.code(json.dumps(hashes, indent=2), language="json")

            st.image(mrz_preprocessed, caption="Preprocessed MRZ Image", width='stretch')

            # rewind file_data if mrz() consumed the stream
            file_data.seek(0)
            st.image(file_data, caption="Uploaded Passport Image", width='stretch')
