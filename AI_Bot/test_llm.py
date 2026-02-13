from llm import ask_llm, SYSTEM_PROMPT

if __name__ == "__main__":
    print("********Welcome to Romit's AI Verse, My name is TONY, AN AI BOT********\n Type /bye to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"/bye", "exit", "quit"}:
            break
        reply = ask_llm(user_input, system=SYSTEM_PROMPT)
        print(f"TONY: {reply}\n")