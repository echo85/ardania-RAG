import streamlit as st
import agent
import os

# --- Page Configuration ---
st.set_page_config(page_title="Ardania Agent Chat", page_icon="🤖")
st.title("🤖 Ardania Agent Chat")
st.caption("Ask me anything about Ardania GDR Ultima Online!")

# --- Environment Variable Check (Optional but Recommended) ---
# Check if necessary environment variables are set, especially OLAMA_API_URL
if not os.path.exists(agent.CHROMA_DB_PATH):
    st.warning(
        f"ChromaDB path '{agent.CHROMA_DB_PATH}' not found. Ensure the vector database exists.",
        icon="⚠️",
    )


# --- Agent Initialization (Cached) ---
@st.cache_resource  # Cache the agent components for efficiency
def load_agent_graph():
    """Loads the agent components and builds the graph."""
    try:
        llm, embeddings, vector_store = agent.initialize_components()
        graph = agent.build_graph(llm, embeddings, vector_store)
        return graph
    except ConnectionError as ce:
        st.error(f"Failed to initialize agent components: {ce}", icon="🚨")
        st.stop()  # Stop execution if components fail to load
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}", icon="🚨")
        st.stop()


# Load the graph (will be cached after first run)
agent_graph = load_agent_graph()

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Interaction ---
if prompt := st.chat_input("What is your question about Ardania?"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Use a placeholder for streaming-like effect
        full_response = ""
        try:
            final_state = agent_graph.invoke({"question": prompt})
            full_response = final_state.get(
                "answer", "Sorry, I couldn't generate a response."
            )

            # Display the full response
            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"An error occurred: {e}"
            st.error(full_response, icon="🚨")

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
