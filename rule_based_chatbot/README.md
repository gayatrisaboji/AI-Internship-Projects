# Rule-Based Chatbot

## Project Overview

This project implements a simple conversational chatbot using Python and rule-based Natural Language Processing (NLP) techniques.

The chatbot identifies the user's intent by matching keywords and regular-expression patterns against the user's input and then generates an appropriate response.

## Features

- Rule-based intent recognition
- Keyword mapping
- Regular-expression based word matching
- Text preprocessing
- Multiple conversation topics
- Multiple responses for each intent
- Random response selection
- Fallback responses
- Conversation history
- Conversation summary

## Supported Topics

The chatbot currently supports:

1. Greetings
2. Python and programming
3. Careers and internships
4. Artificial Intelligence
5. Chatbot identity
6. Help
7. Thanks
8. Goodbye

## Project Structure

```text
rule_based_chatbot/
│
├── chatbot.py
├── intents.py
├── responses.py
├── preprocessor.py
└── README.md