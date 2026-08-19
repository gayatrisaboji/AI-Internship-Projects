import math


# ============================================================
# AI-BASED TIC-TAC-TOE
# ============================================================

HUMAN = "X"
AI = "O"
EMPTY = " "


# ============================================================
# 1. DISPLAY BOARD
# ============================================================

def display_board(board):
    print()
    print("     |     |")
    print(f"  {board[0]}  |  {board[1]}  |  {board[2]}")
    print("_____|_____|_____")
    print("     |     |")
    print(f"  {board[3]}  |  {board[4]}  |  {board[5]}")
    print("_____|_____|_____")
    print("     |     |")
    print(f"  {board[6]}  |  {board[7]}  |  {board[8]}")
    print("     |     |")
    print()


# ============================================================
# 2. CHECK WINNER
# ============================================================

def check_winner(board):
    winning_combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    for a, b, c in winning_combinations:
        if (
            board[a] != EMPTY
            and board[a] == board[b]
            and board[b] == board[c]
        ):
            return board[a]

    if EMPTY not in board:
        return "DRAW"

    return None


# ============================================================
# 3. GET AVAILABLE MOVES
# ============================================================

def get_available_moves(board):
    return [
        i for i in range(9)
        if board[i] == EMPTY
    ]


# ============================================================
# 4. MINIMAX ALGORITHM
# ============================================================

def minimax(board, depth, is_maximizing):
    result = check_winner(board)

    # AI wins
    if result == AI:
        return 10 - depth

    # Human wins
    if result == HUMAN:
        return depth - 10

    # Draw
    if result == "DRAW":
        return 0

    # AI's turn
    if is_maximizing:

        best_score = -math.inf

        for move in get_available_moves(board):

            board[move] = AI

            score = minimax(
                board,
                depth + 1,
                False
            )

            board[move] = EMPTY

            best_score = max(
                best_score,
                score
            )

        return best_score

    # Human's turn
    else:

        best_score = math.inf

        for move in get_available_moves(board):

            board[move] = HUMAN

            score = minimax(
                board,
                depth + 1,
                True
            )

            board[move] = EMPTY

            best_score = min(
                best_score,
                score
            )

        return best_score


# ============================================================
# 5. FIND BEST AI MOVE
# ============================================================

def find_best_move(board):

    best_score = -math.inf
    best_move = None

    for move in get_available_moves(board):

        board[move] = AI

        score = minimax(
            board,
            0,
            False
        )

        board[move] = EMPTY

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# ============================================================
# 6. HUMAN MOVE
# ============================================================

def human_move(board):

    while True:

        try:

            position = int(
                input(
                    "Enter your move (1-9): "
                )
            )

            position -= 1

            if position < 0 or position > 8:
                print(
                    "Please enter a number "
                    "between 1 and 9."
                )
                continue

            if board[position] != EMPTY:
                print(
                    "That position is already occupied."
                )
                continue

            board[position] = HUMAN
            break

        except ValueError:

            print(
                "Invalid input. "
                "Please enter a number from 1 to 9."
            )


# ============================================================
# 7. AI MOVE
# ============================================================

def ai_move(board):

    print("AI is thinking...")

    move = find_best_move(board)

    if move is not None:
        board[move] = AI

    print(
        f"AI selected position {move + 1}."
    )


# ============================================================
# 8. MAIN GAME
# ============================================================

def play_game():

    board = [EMPTY] * 9

    print("=" * 40)
    print("       AI-BASED TIC-TAC-TOE")
    print("=" * 40)

    print("\nYou are X.")
    print("AI is O.")

    print("\nBoard positions:")

    position_board = [
        "1", "2", "3",
        "4", "5", "6",
        "7", "8", "9"
    ]

    display_board(position_board)

    print("You will play first.")

    while True:

        # ----------------------------------------------------
        # HUMAN TURN
        # ----------------------------------------------------

        display_board(board)

        human_move(board)

        result = check_winner(board)

        if result is not None:

            display_board(board)

            if result == HUMAN:
                print("🎉 You win!")

            elif result == "DRAW":
                print("It's a draw!")

            return

        # ----------------------------------------------------
        # AI TURN
        # ----------------------------------------------------

        ai_move(board)

        result = check_winner(board)

        if result is not None:

            display_board(board)

            if result == AI:
                print("🤖 AI wins!")

            elif result == "DRAW":
                print("It's a draw!")

            return


# ============================================================
# 9. START GAME
# ============================================================

if __name__ == "__main__":
    play_game()