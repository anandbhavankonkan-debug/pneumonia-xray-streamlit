"Streamlit frontend for three-class chest X-ray classification."

import pandas as pd
import streamlit as st

from inference_backend import (
    load_manifest,
    load_prediction_model,
    predict_xray,
)


st.set_page_config(
    page_title="Pneumonia X-ray Classifier",
    page_icon="🫁",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading Fine-tuned VGG16 model...")
def get_application_resources():
    return load_prediction_model(), load_manifest()


st.title("Pneumonia X-ray Classification")
st.caption("Fine-tuned VGG16 · Three-class academic demonstration")

st.warning(
    "This application is an academic proof of concept. It is not a medical "
    "device, does not provide a diagnosis and must not replace assessment by "
    "a qualified clinician."
)

st.markdown(
    "Upload one chest X-ray in **DICOM, PNG, JPG or JPEG** format. "
    "The app will show the most likely class and all model probabilities."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["dcm", "png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="Maximum supported file size: 20 MB.",
)

if uploaded_file is None:
    st.info("Upload an image to begin inference.")
else:
    try:
        model, manifest = get_application_resources()
        file_bytes = uploaded_file.getvalue()
        with st.spinner("Preprocessing image and generating prediction..."):
            result = predict_xray(
                model,
                manifest,
                uploaded_file.name,
                file_bytes,
            )

        st.image(
            result["display_image"],
            caption=f"Uploaded image: {uploaded_file.name}",
            clamp=True,
            use_container_width=True,
        )

        st.subheader("Prediction result")
        result_col, confidence_col = st.columns(2)
        result_col.metric("Predicted class", result["predicted_class"])
        confidence_col.metric("Confidence", f"{result['confidence']:.2%}")

        probability_df = pd.DataFrame(
            {
                "Class": list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values()),
            }
        ).set_index("Class")

        st.markdown("#### Probabilities for all classes")
        st.bar_chart(probability_df, y="Probability")
        st.dataframe(
            probability_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True,
        )

        st.caption(
            "Confidence is the model's softmax probability, not a measure of "
            "clinical certainty."
        )
    except Exception as error:
        st.error(f"The image could not be processed: {error}")
