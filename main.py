import streamlit as st
import json
from mrz import main

def submit_action(file_data):
    return main(file_data)

st.title("Welcome to Streamlit Pass")
st.write("This is a sample Streamlit application.")

file_data = st.file_uploader("Upload a file")

if st.button("Submit"):
    if file_data is None:
        st.error("Please upload a file first")
    else:
        
        [mrz_preprocessed, mrzText, PID, family_name, given_names] = submit_action(file_data)

        if mrzText is None:
            st.error("Could not extract MRZ data from the image")
        else:
            st.write("MRZ Text")
            st.markdown(f""">{mrzText}""")
            
            data = {
                "Passport ID": PID,
                "Family Name": family_name,
                "Given Names": given_names
            }
            st.markdown("### Extracted Data")
            st.code(json.dumps(data, indent=2), language="json")

            st.image(mrz_preprocessed, caption="Preprocessed MRZ Image", width="stretch")

            st.image(file_data, caption="Uploaded Passport Image", width="stretch")

    
