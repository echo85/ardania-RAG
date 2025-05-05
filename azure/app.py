import streamlit as st
from agent import get_response

# --- Configurazione Pagina Streamlit ---
st.set_page_config(page_title="Ardania Agent Chat", page_icon="🤖")
st.title("🤖 Ardania Agent Chat")
st.caption("Questa app ti permette di chattare con un agente AI specializzato nel mondo di Ardania.")

# --- Gestione Cronologia Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Ciao! Sono il tuo agente. Come posso aiutarti?"})

# --- Visualizzazione Messaggi Esistenti ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # Usa il ruolo (user/assistant) per l'avatar
        st.markdown(message["content"]) # markdown permette di formattare il testo

# --- Input Utente e Logica di Chat ---
if prompt := st.chat_input("Scrivi qui il tuo messaggio..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("L'agente sta pensando..."):
            try:
                response = get_response(prompt)
                st.markdown(response) # Mostra la risposta dell'agente
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Errore durante la chiamata all'agente: {e}")
                error_message = f"Mi dispiace, si è verificato un errore: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})


if len(st.session_state.messages) > 1: 
    if st.button("Pulisci Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Ciao! Sono il tuo agente. Come posso aiutarti?"}] # Resetta con messaggio iniziale
        st.rerun() 