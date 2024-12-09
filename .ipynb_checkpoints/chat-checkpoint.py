import streamlit as st

# Define the chatbot function
def chatbot(query):
    response = get_most_relevant_sentence(query, processed_sentences)
    return response

# Streamlit interface
def main():
    st.title("Shakespeare Chatbot")
    user_input = st.text_input("Ask me anything about Shakespeare:")
    
    if user_input:
        response = chatbot(user_input)
        st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    main()