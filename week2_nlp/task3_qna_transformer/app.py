import streamlit as st
from qna_model import load_qa_model, get_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Question Answering",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Question Answering System")

st.write(
    "Ask questions from a given context using a "
    "Transformer-based Question Answering model."
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return load_qa_model()


try:
    model = load_model()

    st.success("✅ Transformer model loaded successfully.")

except Exception as e:

    st.error("❌ Model loading failed.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 AI QnA")

    st.write("Week 2 • NLP • Task 03")

    st.divider()

    st.subheader("NLP Pipeline")

    st.write("✅ Context Processing")
    st.write("✅ Question Encoding")
    st.write("✅ Transformer Inference")
    st.write("✅ Answer Extraction")

    st.divider()

    st.info(
        "Powered by HuggingFace Transformers "
        "and Streamlit."
    )


# ============================================================
# CONTEXT
# ============================================================

st.header("📚 Context")

default_context = (
    "Python is a high-level, general-purpose programming "
    "language created by Guido van Rossum. It was first "
    "released in 1991. Python is widely used for web "
    "development, data science, artificial intelligence, "
    "machine learning, automation, and scientific computing. "
    "It is known for its simple syntax and readability."
)

context = st.text_area(
    "Enter your context:",
    value=default_context,
    height=250
)


# ============================================================
# QUESTION
# ============================================================

st.header("❓ Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: Who created Python?"
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💡 Example Questions")

col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "Who created Python?",
        use_container_width=True
    ):

        question = "Who created Python?"

        st.session_state["question"] = question


with col2:

    if st.button(
        "When was Python released?",
        use_container_width=True
    ):

        question = "When was Python first released?"

        st.session_state["question"] = question


with col3:

    if st.button(
        "What is Python used for?",
        use_container_width=True
    ):

        question = "What is Python used for?"

        st.session_state["question"] = question


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
    # MODEL PROCESSING
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Finding the answer..."
    ):

        try:

            # Try:
            # get_answer(model, question, context)

            result = get_answer(
                model,
                question,
                context
            )


        except TypeError:

            # Try alternate function order
            try:

                result = get_answer(
                    question,
                    context,
                    model
                )

            except Exception as e:

                st.error(
                    "❌ Error while getting answer."
                )

                st.code(str(e))

                st.stop()


        except Exception as e:

            st.error(
                "❌ Error while getting answer."
            )

            st.code(str(e))

            st.stop()


    # ========================================================
    # PROCESS RESULT
    # ========================================================

    answer = ""
    confidence = None


    if isinstance(result, dict):

        answer = result.get(
            "answer",
            result.get("text", "")
        )

        confidence = result.get(
            "confidence",
            result.get("score", None)
        )


    elif isinstance(result, tuple):

        if len(result) >= 1:

            answer = result[0]

        if len(result) >= 2:

            confidence = result[1]


    else:

        answer = str(result)


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.header("🎯 Answer")

    st.success(
        answer
    )


    if confidence is not None:

        try:

            confidence_value = float(
                confidence
            )

            if confidence_value <= 1:

                confidence_value *= 100

            st.metric(
                "Confidence",
                f"{confidence_value:.2f}%"
            )

        except Exception:

            st.write(
                f"Confidence: {confidence}"
            )


    # ========================================================
    # SHOW QUESTION
    # ========================================================

    st.subheader("Question")

    st.write(question)


    # ========================================================
    # SHOW CONTEXT
    # ========================================================

    with st.expander(
        "📖 View Context"
    ):

        st.write(context)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI QnA • Week 2 • NLP • Task 03 • "
    "Transformer Question Answering"
)
