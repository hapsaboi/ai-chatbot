# inference.py
from chatbot import UniversityChatbot
import time

if __name__ == "__main__":
    # Initialize chatbot
    print("Initializing chatbot...")
    chatbot = UniversityChatbot()
    print("Chatbot ready!")
    
    # Interactive loop
    while True:
        try:
            query = input("\nAsk a question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            start_time = time.time()
            response = chatbot.get_response(query)
            end_time = time.time()
            
            print(f"\nResponse: {response['response']}")
            print(f"Confidence: {response['confidence']}")
            print(f"\nResponse: {response['response']}")
            print(f"Confidence: {response['confidence']:.2f}")
            print(f"Response time: {(end_time - start_time):.2f} seconds")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    print("\nThank you for using the University Chatbot!")