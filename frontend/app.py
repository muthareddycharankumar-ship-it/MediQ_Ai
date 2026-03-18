import streamlit as st
import requests

API = "http://192.168.0.130:8000/ask"

st.set_page_config(page_title="MedIQ", page_icon="🩺", layout="centered")

st.markdown("""
<style>
.stChatMessage {
    opacity: 1 !important;
    filter: none !important;
}
.stChatMessageContent {
    opacity: 1 !important;
    filter: none !important;
}
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    filter: none !important;
}

/* Auto scroll to bottom */
.main .block-container {
    padding-bottom: 100px;
}
</style>

<script>
function scrollToBottom() {
    const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
    if (messages.length > 0) {
        messages[messages.length - 1].scrollIntoView({ behavior: "smooth", block: "start" });
    }
}
</script>
""", unsafe_allow_html=True)

st.title("🩺 MedIQ")
st.caption("Rheumatology AI Assistant powered by Okulr Techminds")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Auto scroll anchor
scroll_anchor = st.empty()

# Chat input
question = st.chat_input("Ask your rheumatology question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🩺"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("_Thinking..._")
        response_placeholder = st.empty()
        full_response = ""
        first_chunk = True

        try:
            with requests.post(
                API,
                json={"question": question},
                stream=True,
                timeout=120
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        if first_chunk:
                            thinking_placeholder.empty()
                            first_chunk = False
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except requests.exceptions.ConnectionError:
            thinking_placeholder.empty()
            full_response = "❌ Cannot connect to backend. Make sure the server is running on port 8000."
            response_placeholder.markdown(full_response)
        except requests.exceptions.Timeout:
            thinking_placeholder.empty()
            full_response = "⏱️ Request timed out. The model is taking too long to respond."
            response_placeholder.markdown(full_response)
        except Exception as e:
            thinking_placeholder.empty()
            full_response = f"❌ Error: {str(e)}"
            response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Scroll to latest message
    scroll_anchor.markdown("""
    <script>
        window.parent.document.querySelector('[data-testid="stChatFloatingInputContainer"]')
            .scrollIntoView({ behavior: "smooth" });
    </script>
    """, unsafe_allow_html=True)

    st.rerun()
