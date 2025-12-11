import streamlit as st

st.set_page_config(
    page_title="MRZ Document Scanner",
    page_icon="🛂",
    layout="wide"
)

st.title("🛂 Travel Document MRZ Scanner")
st.markdown("### Welcome to the Machine Readable Zone (MRZ) Detection System")

st.markdown("""
---

## 📖 What is MRZ?

The **Machine Readable Zone (MRZ)** is a standardized area on travel documents such as passports, visas, 
and ID cards that contains encoded information about the document holder. This information is formatted 
according to international standards (ICAO Doc 9303) and can be read by optical character recognition (OCR) 
systems at border control and security checkpoints worldwide.

## 🎯 Purpose of This Application

This application provides an automated solution for:

- **Extracting MRZ data** from scanned travel documents
- **Parsing and validating** document information using OCR technology
- **Verifying check digits** to ensure data integrity
- **Converting raw MRZ text** into structured, readable JSON format

## 🚀 Features

- **Individual Document Processing**: Upload and process a single travel document at a time
- **Batch Processing**: Upload and process multiple documents simultaneously
- **Support for Multiple Document Types**:
  - **TD1**: ID cards (credit card-sized documents with 3-line MRZ)
  - **TD2**: Official travel documents (passport card-sized with 2-line MRZ)
  - **TD3**: Passports (standard passport with 2-line MRZ)
- **Automatic Data Validation**: Built-in check digit verification for document authenticity
- **Fast Processing**: Optimized OCR engine with custom-trained models

## 📄 Extracted Information

The application extracts the following information from travel documents:

- Document type and number
- Issuing country
- Holder's full name (family name and given names)
- Nationality
- Date of birth
- Document expiry date
- Sex/Gender
- Optional data fields
- Hash verification results

## 🔧 How to Use

Navigate to one of the pages from the sidebar:

1. **Individual**: Process a single document - ideal for quick checks and detailed viewing
2. **Multiple**: Batch process multiple documents - perfect for processing several documents at once

## 🔒 Privacy & Security

- All processing is done locally on the server
- No data is stored permanently
- Uploaded files are processed in memory and discarded after extraction

---

**Note**: This application is designed for demonstration and educational purposes. 
For production use in sensitive environments, ensure proper security measures and compliance 
with data protection regulations.
""")

st.info("👈 Select a processing mode from the sidebar to get started!")
