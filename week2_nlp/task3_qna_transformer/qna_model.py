from transformers import pipeline


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-cased-distilled-squad"


# ============================================================
# LOAD QUESTION ANSWERING MODEL
# ============================================================

def load_qa_model():
    """
    Load the pre-trained HuggingFace
    extractive Question Answering model.
    """

    qa_pipeline = pipeline(
        "question-answering",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME
    )

    return qa_pipeline


# ============================================================
# GENERATE ANSWER
# ============================================================

def get_answer(qa_pipeline, context, question):
    """
    Extract the most relevant answer from the context.
    """

    result = qa_pipeline(
        question=question,
        context=context
    )

    return {
        "answer": result["answer"],
        "confidence": result["score"],
        "start": result["start"],
        "end": result["end"]
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI QnA - HUGGINGFACE TRANSFORMER TEST")
    print("=" * 60)

    print("\nLoading model...")

    qa = load_qa_model()

    print("Model loaded successfully!")

    context = """
    Python is a high-level programming language created by
    Guido van Rossum. It was first released in 1991.
    Python is widely used for artificial intelligence,
    machine learning, data science, web development,
    automation, and software development.
    Python is known for its simple and readable syntax.
    """

    question = "Who created Python?"

    result = get_answer(
        qa,
        context,
        question
    )

    print("\nQUESTION")
    print("-" * 60)
    print(question)

    print("\nANSWER")
    print("-" * 60)
    print(result["answer"])

    print("\nCONFIDENCE")
    print("-" * 60)
    print(f"{result['confidence'] * 100:.2f}%")

    print("\nANSWER POSITION")
    print("-" * 60)
    print(f"Start: {result['start']}")
    print(f"End:   {result['end']}")

    print("\n" + "=" * 60)