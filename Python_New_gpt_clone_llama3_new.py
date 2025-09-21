import streamlit as st
import time
from ollama import Client

# Initialize Ollama client
client = Client()


st.set_page_config(page_title=" Local ChatGPT Clone", page_icon="🤖", layout="wide")
st.title("🤖 Local ChatGPT Clone")


st.sidebar.title(" Settings")

# Model Selector
available_models = ["llama3:latest", "mistral:latest", "codellama:latest"]
model = st.sidebar.selectbox("Select Model", available_models)

system_prompt = st.sidebar.text_area(
    "System Prompt (optional)",
    "You are a helpful assistant.",
    help="Define the behavior of the assistant"
)


if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []


st.sidebar.markdown("---")
st.sidebar.info("Switch your browser or Streamlit settings for dark/light mode.")


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]


for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        token_count = 0
        try:
            for chunk in client.chat(
                model=model,
                messages=st.session_state.messages,
                stream=True
            ):
                content = chunk["message"]["content"]
                token_count += len(content.split())
                full_response += content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            message_placeholder.error(f"⚠️ Error: {e}")
            full_response = "⚠️ An error occurred during generation."

        end_time = time.time()

    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

 
    with st.expander("📊Stats", expanded=False):
        st.markdown(f"**Tokens (estimated):** {token_count}")
        st.markdown(f"**Response time:** {end_time - start_time:.2f} seconds")


st.sidebar.markdown("---")
st.sidebar.markdown("🛠️ Built with [Streamlit](https://streamlit.io/) and [Ollama](https://ollama.com)")
st.sidebar.markdown("👨‍💻 By *Your Name* — AI Sandipan")
