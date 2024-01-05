from functions import agent_executor
control = True
while True:
  if control:
    print("Bot: ", "Hello, I am a bot that can help you with stock price information.\nI can tell you the price of a stock, the percentage change of a stock over \na period of time, and the best performing stock over a period of time.\nWhat would you like to know?")
    print("\nRemember to type 'exit' to exit\n")
    control = False

  content: str = input("You: ")
  if content == "exit":
    break

  try:
    result = agent_executor(content)
    print("Bot: ", result['output'])
  except Exception as e:
    print(e)
    break
