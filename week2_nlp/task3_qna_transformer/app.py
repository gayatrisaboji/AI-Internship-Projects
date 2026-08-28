
import streamlit as st

from qna_model import load_qa_model, get_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Question Answering",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .hero {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(
            135deg,
            #f5f7ff,
            #eef2ff
        );
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
        font-size: 22px;
        font-weight: 600;
    }

    .pipeline-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div style="font-size:15px; font-weight:700;">
            HUGGINGFACE - TRANSFORMERS - NLP
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Ask your context.<br>Get your answer.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An intelligent extractive Question Answering system powered by a pre-trained Transformer model.</div>',
    unsafe_allow_html=True
)

   

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return load_qa_model()


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 AI QnA")

    st.write("Week 2 • NLP • Task 03")

    st.divider()

    st.subheader("Application")

    st.success("✓ Model Loaded")
    st.success("✓ Transformer NLP")
    st.success("✓ Extractive QnA")

    st.divider()

    st.subheader("NLP Pipeline")

    st.write("✓ Context Processing")
    st.write("✓ Question Encoding")
    st.write("✓ Transformer Inference")
    st.write("✓ Answer Extraction")

    st.divider()

    st.info(
        "Powered by HuggingFace Transformers "
        "and Streamlit."
    )


# ============================================================
# CONTEXT SECTION
# ============================================================

st.header("📚 Context")

default_context = """
Artificial intelligence is a branch of computer science
that focuses on creating machines capable of performing
tasks that normally require human intelligence. These tasks
include learning, reasoning, problem solving, understanding
natural language, and recognizing images.

Machine learning is a major part of artificial intelligence
that allows computers to learn patterns from data without
being explicitly programmed for every task.

Artificial intelligence is widely used in healthcare,
education, finance, transportation, robotics, and
customer service.
"""

context = st.text_area(
    "Enter your context:",
    value=default_context.strip(),
    height=260
)


# ============================================================
# QUESTION SECTION
# ============================================================

st.header("❓ Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is artificial intelligence?"
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💡 Example Questions")

col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "What is AI?",
        use_container_width=True
    ):

        st.session_state["question"] = (
            "What is artificial intelligence?"
        )

        st.rerun()


with col2:

    if st.button(
        "What is machine learning?",
        use_container_width=True
    ):

        st.session_state["question"] = (
            "What is machine learning?"
        )

        st.rerun()


with col3:

    if st.button(
        "Where is AI used?",
        use_container_width=True
    ):

        st.session_state["question"] = (
            "Where is artificial intelligence used?"
        )

        st.rerun()


# ============================================================
# USE SELECTED EXAMPLE QUESTION
# ============================================================

if "question" in st.session_state:

    question = st.session_state["question"]

    st.info(
        f"Selected question: {question}"
    )


# ============================================================
# FIND ANSWER
# ============================================================

st.divider()

if st.button(
    "🔍 Find Answer",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not context.strip():

        st.warning(
            "Please enter a context."
        )

        st.stop()


    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Transformer is finding the answer..."
    ):

        try:

            result = get_answer(
                model,
                context,
                question
            )

        except Exception as e:

            st.error(
                "❌ Error while generating the answer."
            )

            st.code(str(e))

            st.stop()


    # ========================================================
    # EXTRACT RESULT
    # ========================================================

    answer = result["answer"]
    confidence = result["confidence"]
    start = result["start"]
    end = result["end"]


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    st.header("🎯 Answer")

    st.markdown(
        f"""
        <div class="answer-box">
            {answer}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    st.write("")

    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    with metric2:

        st.metric(
            "Answer Start",
            start
        )


    with metric3:

        st.metric(
            "Answer End",
            end
        )


    # ========================================================
    # QUESTION
    # ========================================================

    st.subheader("❓ Question")

    st.write(question)


    # ========================================================
    # CONTEXT
    # ========================================================

    with st.expander(
        "📖 View Context"
    ):

        st.write(context)


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "🤖 Model Information"
    ):

        st.write(
            "Model: distilbert-base-cased-distilled-squad"
        )

        st.write(
            "Task: Extractive Question Answering"
        )

        st.write(
            "Framework: HuggingFace Transformers"
        )


# ============================================================
# FEATURES
# ============================================================

st.divider()

st.subheader("✨ Key Features")

feature1, feature2, feature3 = st.columns(3)


with feature1:

    st.markdown(
        """
        ### 🧠 Transformer Model

        Uses a pre-trained DistilBERT Transformer
        model for intelligent question answering.
        """
    )


with feature2:

    st.markdown(
        """
        ### 📖 Context Based

        Searches the supplied context and extracts
        the most relevant answer span.
        """
    )


with feature3:

    st.markdown(
        """
        ### 🎯 Confidence Score

        Displays the model's confidence score for
        the extracted answer.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI QnA • Week 2 • NLP • Task 03 • "
    "Powered by Transformers & Streamlit"
)

