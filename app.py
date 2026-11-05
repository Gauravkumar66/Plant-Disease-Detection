import streamlit as st
from PIL import Image

from utils import (
    load_plant_model,
    load_class_names,
    predict_image,
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 80% 0%, #e7f3e8 0%, transparent 30%),
            #f6f8f5;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #123b2a;
    }

    [data-testid="stSidebar"] * {
        color: #f2f8f1;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d4e6d8;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .brand-subtitle {
        color: #b9d4c0;
        font-size: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .sidebar-label {
        font-size: 11px;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #9fbea8;
        font-weight: 700;
        margin-top: 28px;
        margin-bottom: 10px;
    }

    .sidebar-info {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px;
        margin-top: 8px;
    }

    .hero {
        padding: 24px 0 18px 0;
    }

    .eyebrow {
        color: #398052;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        color: #183d2b;
        font-size: clamp(32px, 4vw, 48px);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .hero-title span {
        color: #4b9460;
    }

    .hero-description {
        color: #68796d;
        font-size: 16px;
        line-height: 1.7;
        margin-top: 14px;
        max-width: 700px;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: #203d2c;
        margin-bottom: 4px;
    }

    .section-description {
        font-size: 13px;
        color: #7b897e;
        margin-bottom: 16px;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e4ebe3;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 8px 30px rgba(30, 65, 40, 0.04);
    }

    .panel-heading {
        color: #284a34;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .panel-caption {
        color: #829087;
        font-size: 12px;
        margin-bottom: 14px;
    }

    .status-pill {
        display: inline-block;
        background: #e8f5eb;
        color: #347b48;
        border-radius: 100px;
        padding: 6px 12px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .prediction-name {
        color: #1e452d;
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.3;
        margin: 10px 0;
        overflow-wrap: anywhere;
    }

    .confidence-label {
        color: #77867a;
        font-size: 12px;
        font-weight: 600;
    }

    .confidence-value {
        color: #3c8652;
        font-size: 27px;
        font-weight: 800;
        font-family: 'Manrope', sans-serif;
    }

    .prediction-row {
        background: #f7faf6;
        border: 1px solid #e9efe7;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 8px 0;
    }

    .prediction-row-title {
        color: #36513d;
        font-size: 13px;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .prediction-row-value {
        color: #4c8c5b;
        font-size: 13px;
        font-weight: 700;
    }

    .footer {
        color: #8b978d;
        font-size: 12px;
        text-align: center;
        padding: 28px 0 10px 0;
    }

    div[data-testid="stFileUploader"] {
        background: #f9fcf8;
        border: 1.5px dashed #b7d1bb;
        border-radius: 16px;
        padding: 14px;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent;
    }

    .stButton > button {
        background: #2e7544;
        color: white;
        border: 1px solid #2e7544;
        border-radius: 12px;
        padding: 12px 18px;
        font-weight: 700;
        min-height: 48px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #205b34;
        border-color: #205b34;
        color: white;
        transform: translateY(-1px);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px #cce5d0;
    }

    [data-testid="stMetric"] {
        background: #f6faf5;
        border: 1px solid #e6eee4;
        padding: 14px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #738275;
    }

    [data-testid="stMetricValue"] {
        color: #2e7544;
    }

    hr {
        border-color: #e5ece4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def get_model():
    return load_plant_model()


@st.cache_data
def get_classes():
    return load_class_names()


try:
    model = get_model()
    class_names = get_classes()

except Exception as error:
    st.error(f"Could not load the model or class mapping: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">🌿 PlantCare AI</div>
        <div class="brand-subtitle">Intelligent plant health</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
    st.markdown("🌱  Disease detection")
    st.markdown("📊  Prediction overview")

    st.markdown('<div class="sidebar-label">Model information</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-info">
            <div style="font-size:12px;color:#b9d4c0;">MODEL INPUT</div>
            <div style="font-size:18px;font-weight:700;margin:4px 0;">224 × 224 RGB</div>
            <div style="font-size:12px;color:#d4e6d8;">Normalized image pixels</div>
        </div>
        <div class="sidebar-info">
            <div style="font-size:12px;color:#b9d4c0;">SUPPORTED CLASSES</div>
            <div style="font-size:18px;font-weight:700;margin:4px 0;">38 classes</div>
            <div style="font-size:12px;color:#d4e6d8;">Based on your trained model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">How to use</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Upload a clear leaf image.
        2. Select **Analyze leaf**.
        3. Review the predicted class and confidence.
        """
    )

    st.markdown("---")
    st.caption("PlantCare AI · Deep Learning Demo")


# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered plant health assistant</div>
        <h1 class="hero-title">
            Understand your plant.<br>
            <span>Protect its health.</span>
        </h1>
        <div class="hero-description">
            Upload a leaf image and let your trained deep learning model
            identify the most likely class from its 38 supported categories.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# UPLOAD PANEL
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">Leaf analysis</div>
    <div class="section-description">
        Start with a clear, well-lit image of a single plant leaf.
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, tip_col = st.columns([1.7, 1], gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG.",
        label_visibility="collapsed",
    )

with tip_col:
    st.markdown(
        """
        <div class="panel" style="padding:16px;">
            <div class="panel-heading">📷 Image quality tips</div>
            <div class="panel-caption" style="margin:8px 0 0 0;">
                Use a focused image, natural lighting, and keep the leaf
                clearly visible. Avoid blurry or heavily shadowed images.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# IMAGE PREVIEW + PREDICTION
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)

    image_col, result_col = st.columns([1, 1], gap="large")

    with image_col:
        st.markdown(
            """
            <div class="section-title">Your uploaded image</div>
            <div class="section-description">
                Preview the leaf before running the analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )

        st.caption(
            f"Image size: {image.width} × {image.height} pixels"
        )

    with result_col:
        st.markdown(
            """
            <div class="section-title">AI analysis</div>
            <div class="section-description">
                Run the model to see its prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="panel">
                <div class="panel-heading">Ready to analyze</div>
                <div class="panel-caption">
                    Your image will be resized and normalized before prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "🌿  Analyze leaf",
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("Analyzing image..."):
                try:
                    results = predict_image(
                        model,
                        image,
                        class_names,
                    )

                    st.session_state["plant_results"] = results

                except Exception as error:
                    st.error(f"Prediction failed: {error}")
                    st.session_state.pop("plant_results", None)

        results = st.session_state.get("plant_results")

        if results:
            best_result = results[0]

            predicted_name = best_result["class_name"].replace(
                "___", " — "
            )
            confidence = best_result["confidence"]

            st.markdown(
                f"""
                <div class="panel" style="margin-top:16px;">
                    <div class="status-pill">✓ ANALYSIS COMPLETE</div>
                    <div class="confidence-label" style="margin-top:16px;">
                        MOST LIKELY CLASS
                    </div>
                    <div class="prediction-name">
                        {predicted_name}
                    </div>
                    <div class="confidence-label">
                        Model confidence
                    </div>
                    <div class="confidence-value">
                        {confidence * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                max(0.0, min(1.0, confidence))
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="section-title">Top predictions</div>
                <div class="section-description">
                    The three highest model output scores.
                </div>
                """,
                unsafe_allow_html=True,
            )

            for rank, result in enumerate(results, start=1):
                name = result["class_name"].replace("___", " — ")
                score = result["confidence"] * 100

                st.markdown(
                    f"""
                    <div class="prediction-row">
                        <div style="display:flex;justify-content:space-between;gap:12px;">
                            <div class="prediction-row-title">
                                {rank}. {name}
                            </div>
                            <div class="prediction-row-value">
                                {score:.2f}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(
                    max(0.0, min(1.0, result["confidence"]))
                )

            st.caption(
                "Prediction confidence is a model score, not a confirmed "
                "agricultural diagnosis."
            )

else:
    st.markdown(
        """
        <div class="panel" style="text-align:center;padding:38px 20px;">
            <div style="font-size:42px;">🌿</div>
            <div class="section-title" style="margin-top:10px;">
                Your plant analysis starts here
            </div>
            <div class="section-description" style="margin:8px 0 0 0;">
                Upload a leaf image above to view its preview and prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        PlantCare AI · Plant disease classification powered by deep learning
    </div>
    """,
    unsafe_allow_html=True,
)