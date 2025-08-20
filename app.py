import os
import tempfile
import streamlit as st
import json

# Try to import from your existing ocr_utils or fall back to compatible version
try:
    from ocr_utils import analyze_prescription
except ImportError:
    st.error("❌ Could not import analyze_prescription from ocr_utils.py")
    st.info("Please ensure ocr_utils.py is in the same directory and contains the analyze_prescription function.")
    st.stop()

# Set up the Streamlit page configuration
st.set_page_config(page_title="AI Prescription Scanner", page_icon="💊", layout="wide")

# Title and description
st.title("💊 AI Prescription Scanner")
st.caption(
    "Extract medicines & lab tests from prescriptions. Fetch FDA warnings/interactions with offline fallback. "
    "⚠️ **For demonstration only — not medical advice.**"
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    med_threshold = st.slider("Medicine fuzzy match threshold", 70, 95, 82, 1, 
                             help="Higher values = stricter matching")
    lab_threshold = st.slider("Lab test fuzzy match threshold", 65, 95, 75, 1,
                             help="Higher values = stricter matching")
    top_k_meds = st.slider("Max medicines to detect", 1, 10, 6, 1)
    top_k_labs = st.slider("Max lab tests to detect", 1, 15, 10, 1)
    
    st.markdown("---")
    st.subheader("🔗 openFDA Integration")
    st.caption("OpenFDA provides official drug information. API key is optional but recommended for higher rate limits.")
    api_key = st.text_input("OpenFDA API Key (optional)", type="password", 
                           help="Get a free API key from https://open.fda.gov/apis/authentication/")

# File uploader for prescription images
uploaded = st.file_uploader(
    "📁 Upload a prescription image", 
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
)

if uploaded:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded, caption="Uploaded Prescription", use_column_width=True)
    
    # Create a temporary file to store the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner("🔍 Analyzing prescription..."):
            # Try different function call patterns based on your original code
            try:
                # Try with all parameters (enhanced version)
                result = analyze_prescription(
                    tmp_path,
                    med_threshold=med_threshold,
                    lab_threshold=lab_threshold,
                    top_k_meds=top_k_meds,
                    top_k_labs=top_k_labs,
                    api_key=api_key if api_key else None
                )
            except TypeError:
                # Fallback to original function signature
                st.warning("Using basic analysis mode (some parameters not supported)")
                result = analyze_prescription(tmp_path, api_key=api_key if api_key else None)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Check if result is structured correctly
        if not isinstance(result, dict):
            st.error("❌ Unexpected result format from analysis function")
            st.stop()
        
        # Display confidence score if available
        confidence = result.get("confidence_score")
        if confidence is not None:
            st.metric("📊 Analysis Confidence", f"{confidence:.1%}")
        
        # -------- Extracted text --------
        st.subheader("📄 Extracted Text")
        raw_text = result.get("raw_text", "")
        if raw_text:
            st.text_area("Raw OCR Output", raw_text, height=150)
        else:
            st.info("ℹ️ No text was extracted. Try using a clearer image.")

        # Create two columns for medicines and lab tests
        col1, col2 = st.columns(2)

        # -------- Medicines --------
        with col1:
            st.subheader("💊 Detected Medicines")
            meds = result.get("medicines", [])
            
            if not meds:
                st.info("ℹ️ No medicines detected. Try adjusting the threshold or using a clearer image.")
            else:
                for i, med in enumerate(meds):
                    # Create expandable section for each medicine
                    with st.expander(f"**{med.get('canonical', 'Unknown')}** (Score: {med.get('score', 0)})", expanded=i==0):
                        
                        # Basic info
                        matched_as = med.get('matched_as', '')
                        if matched_as:
                            st.write(f"**Matched as:** `{matched_as}`")
                        
                        generic = med.get('generic', '')
                        if generic:
                            st.write(f"**Generic name:** {generic}")
                        
                        drug_class = med.get('drug_class', '')
                        if drug_class:
                            st.write(f"**Drug class:** {drug_class}")
                        
                        # Label information
                        label = med.get("label", {})
                        source = label.get("source", "offline")
                        
                        if source == "openFDA":
                            st.success("🟢 **Data from openFDA**")
                        else:
                            st.warning("🟡 **Offline database**")
                        
                        # Display warnings and other information
                        boxed = label.get("boxed_warning", [])
                        if boxed:
                            st.error("⚠️ **BLACK BOX WARNING**")
                            for warning in boxed[:2]:  # Limit to 2 warnings
                                st.write(f"• {warning}")
                        
                        warnings = label.get("warnings", [])
                        if warnings:
                            st.warning("⚠️ **Warnings**")
                            for warning in warnings[:3]:  # Limit to 3 warnings
                                st.write(f"• {warning}")
                        
                        interactions = label.get("drug_interactions", [])
                        if interactions:
                            st.info("🔗 **Drug Interactions**")
                            for interaction in interactions[:2]:  # Limit to 2 interactions
                                st.write(f"• {interaction}")
                        
                        contraindications = label.get("contraindications", [])
                        if contraindications:
                            st.error("🚫 **Contraindications**")
                            for contra in contraindications[:2]:
                                st.write(f"• {contra}")
                        
                        dosage = label.get("dosage", [])
                        if dosage:
                            st.success("💊 **Dosage & Administration**")
                            for dose in dosage[:1]:  # Limit to 1 dosage instruction
                                st.write(f"• {dose}")
                        
                        indications = label.get("indications", [])
                        if indications:
                            st.info("🎯 **Indications**")
                            for indication in indications[:1]:
                                st.write(f"• {indication}")
                        
                        adverse_reactions = label.get("adverse_reactions", [])
                        if adverse_reactions:
                            st.warning("😷 **Adverse Reactions**")
                            for reaction in adverse_reactions[:2]:
                                st.write(f"• {reaction}")

        # -------- Lab tests --------
        with col2:
            st.subheader("🧪 Detected Lab Tests")
            labs = result.get("lab_tests", [])
            
            if not labs:
                st.info("ℹ️ No lab tests detected. Try adjusting the threshold.")
            else:
                for lab in labs:
                    with st.expander(f"**{lab.get('canonical', 'Unknown')}** (Score: {lab.get('score', 0)})"):
                        matched_as = lab.get('matched_as', '')
                        if matched_as:
                            st.write(f"**Matched as:** `{matched_as}`")
                        
                        description = lab.get('description', '')
                        if description:
                            st.write(f"**Description:** {description}")
                        else:
                            st.write("*No description available*")

        # -------- Interaction warnings --------
        pairwise_notes = result.get("pairwise_notes", [])
        if pairwise_notes:
            st.subheader("⚠️ Drug Interaction Warnings")
            st.warning("**These are automated checks - always consult a healthcare professional**")
            for note in pairwise_notes:
                st.error(f"🚨 {note}")

        # -------- Analysis summary --------
        st.subheader("📈 Analysis Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Medicines Found", len(meds))
        with col2:
            st.metric("Lab Tests Found", len(labs))
        with col3:
            st.metric("Warnings", len(pairwise_notes))

        # -------- Export functionality --------
        if st.button("📥 Export Results as JSON"):
            export_data = {
                "timestamp": str(st.session_state.get('analysis_time', 'unknown')),
                "image_name": uploaded.name,
                "analysis_results": result
            }
            
            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"prescription_analysis_{uploaded.name}.json",
                mime="application/json"
            )

    except Exception as e:
        st.error(f"❌ **Analysis failed:** {str(e)}")
        st.info("**Troubleshooting tips:**")
        st.write("• Ensure the image is clear and well-lit")
        st.write("• Check that Tesseract OCR is properly installed")
        st.write("• Verify that drug_db.json and lab_tests_db.json files exist in the data/ folder")
        
        # Clean up temporary file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        if st.checkbox("Show detailed error for debugging"):
            st.exception(e)

else:
    # Show instructions when no file is uploaded
    st.info("👆 **Upload a prescription image to get started**")
    
    st.markdown("### 📋 **What this app can do:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Medicine Detection:**
        • Extract drug names from prescriptions
        • Find generic and brand names
        • Get FDA warnings and interactions
        • Identify drug classes and contraindications
        """)
    
    with col2:
        st.markdown("""
        **Lab Test Detection:**
        • Identify laboratory tests
        • Extract test abbreviations
        • Provide test descriptions
        • Support common medical acronyms
        """)
    
    st.markdown("### ⚠️ **Important Disclaimers:**")
    st.error("""
    • **This is a demonstration tool only**
    • **Not intended for medical diagnosis or treatment**
    • **Always consult healthcare professionals**
    • **Verify all extracted information manually**
    • **OCR accuracy depends on image quality**
    """)
    
    st.markdown("### 🔧 **Setup Requirements:**")
    with st.expander("Click to see setup instructions"):
        st.code("""
# Required files structure:
prescription/
├── app.py (this file)
├── ocr_utils.py (analysis functions)
└── data/
    ├── drug_db.json
    └── lab_tests_db.json

# Install required packages:
pip install streamlit opencv-python pytesseract rapidfuzz requests numpy
        """)

# Add footer
st.markdown("---")
st.caption("Built with Streamlit • OpenCV • Tesseract OCR • OpenFDA API")
