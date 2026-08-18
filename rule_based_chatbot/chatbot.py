import re
import random

from intents import INTENTS
from responses import RESPONSES
from preprocessor import preprocess_text


print("===================================")
print("       RULE-BASED CHATBOT")
print("===================================")
print("Chatbot: Hello! I am your chatbot.")
print("Chatbot: Ask me about Python, AI, or careers.")
print("Chatbot: Type 'bye' to exit.")
print()


def contains_keyword(text, keywords):
    """
    Check whether any keyword appears as a complete word
    in the user's input.
    """
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"

        if re.search(pattern, text):
            return True

    return False


def detect_intent(user_input):
    """
    Identify the user's intent using keyword mapping.
    """
    for intent, keywords in INTENTS.items():

        if contains_keyword(user_input, keywords):
            return intent

    return "fallback"


def get_response(intent):
    """
    Select a random response for the detected intent.
    """
    return random.choice(RESPONSES[intent])


# Store conversation history
conversation_history = []


# Main chatbot loop
while True:

    # Get user input
    user_input = input("You: ")

    # Preprocess user input
    user_input = preprocess_text(user_input)

    # Detect intent
    intent = detect_intent(user_input)

    # Store user message and detected intent
    conversation_history.append({
        "user_input": user_input,
        "intent": intent
    })

    # Generate response
    response = get_response(intent)

    # Display response
    print("Chatbot:", response)

    # Exit chatbot
    if intent == "goodbye":
        print()
        print("===================================")
        print("       CONVERSATION SUMMARY")
        print("===================================")

        print("Total messages:", len(conversation_history))

        print()
        print("Topics discussed:")

        for message in conversation_history:
            if message["intent"] != "goodbye":
                print(
                    "-",
                    message["user_input"],
                    "→",
                    message["intent"]
                )

        print()
        print("Chatbot: Thank you for chatting with me!")
        break