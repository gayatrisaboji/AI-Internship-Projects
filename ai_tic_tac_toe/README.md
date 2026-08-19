# AI-Based Tic-Tac-Toe Game

## Description

A command-line Tic-Tac-Toe game where a human player competes against an AI opponent.

The AI uses the Minimax algorithm to evaluate possible future moves and select the best available move.

## Features

- Human vs AI gameplay
- Minimax algorithm
- Automatic AI move selection
- Win detection
- Draw detection
- Invalid move handling
- Simple command-line interface

## How It Works

The game uses a 3x3 board.

The human player uses:

X

The AI uses:

O

The AI examines possible moves using the Minimax algorithm.

The algorithm:

1. Generates possible moves.
2. Simulates each move.
3. Evaluates the resulting game state.
4. Recursively evaluates future moves.
5. Selects the move with the highest score.

## AI Scoring

The AI uses the following scoring system:

- AI win → positive score
- Human win → negative score
- Draw → 0

The AI also considers the depth of the game so that it prefers winning quickly and delays losing situations.

## Technologies Used

- Python
- Minimax Algorithm
- Recursion
- Command Line Interface

## Project Structure

```text
ai_tic_tac_toe/
│
├── tic_tac_toe.py
├── README.md
└── requirements.txt