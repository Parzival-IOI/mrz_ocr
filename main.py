import streamlit as st
import json
import time
from mrz import mrz

def submit_action(file_data):
    start_time = time.time()

    # unpack directly from mrz()
    mrz_preprocessed, mrzText, PID, family_name, given_names = mrz(file_data)

    execution_time = time.time() - start_time
    return mrz_preprocessed, mrzText, PID, family_name, given_names, execution_time


st.title("Welcome to Streamlit Pass")
st.write("This is a sample Streamlit application.")

file_data = st.file_uploader("Upload a file")

if st.button("Submit"):
    if file_data is None:
        st.error("Please upload a file first")
    else:
        (
            mrz_preprocessed,
            mrzText,
            PID,
            family_name,
            given_names,
            execution_time,
        ) = submit_action(file_data)

        if mrzText is None:
            st.error("Could not extract MRZ data from the image")
        else:
            st.write("MRZ Text")
            st.markdown(f""">{mrzText}""")

            st.info(f"⏱️ Execution Time: {execution_time:.2f} seconds")

            data = {
                "Passport ID": PID,
                "Family Name": family_name,
                "Given Names": given_names,
            }
            st.markdown("### Extracted Data")
            st.code(json.dumps(data, indent=2), language="json")

            st.image(mrz_preprocessed, caption="Preprocessed MRZ Image", use_container_width=True)

            # if mrz() reads file_data.read(), you may need to rewind:
            file_data.seek(0)
            st.image(file_data, caption="Uploaded Passport Image", use_container_width=True)
