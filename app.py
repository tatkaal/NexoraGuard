# ## 6. Interactive Chatbot Demonstration (Streamlit Web App)

def run_streamlit_app():
    import streamlit as st
    from nexora_rag import init_nexora_rag

    st.set_page_config(page_title="NexoraGuard AI Chatbot", layout="wide")

    # --- Sidebar ---
    st.sidebar.title("NexoraGuard Controls")
    st.sidebar.info(
        "This is a demo of the Nexora RAG Customer Service Chatbot. "
        "It uses a local LLM (`qwen3:1.7b` via Ollama) and a knowledge base "
        "built from Nexora's product and FAQ data."
    )
    show_debug_info = st.sidebar.checkbox("Show Debug Info (Intent, Entities, Sources)", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About NexoraGuard")
    st.sidebar.markdown(
        "NexoraGuard is designed to assist with queries regarding Nexora's insurance products, "
        "coverage, claims, and account management."
    )

    # --- Header ---
    st.title("🤖 NexoraGuard Insurance Assistant")
    st.caption("Your AI-powered guide to Nexora's insurance solutions.")

    # --- Load or build RAG backend (cached) ---
    ask_nexora_guard, rag_chain, vector_store = init_nexora_rag()

    # --- Initialize chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm NexoraGuard. How can I help you with your Nexora insurance needs today?",
                "sources": []
            }
        ]

    # --- Render chat messages ---
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages when debugging
            if message["role"] == "assistant" and show_debug_info and message.get("sources"):
                with st.expander("View Sources Used", expanded=False):
                    for src_idx, source in enumerate(message["sources"]):
                        st.markdown(
                            f"**Source {src_idx+1}: Type – {source.metadata.get('source_type','N/A')}**"
                        )
                        if source.metadata.get('source_type') == 'product_data':
                            st.caption(f"Product: {source.metadata.get('product_name','N/A')}")
                        elif source.metadata.get('source_type') == 'faq':
                            st.caption(f"FAQ Category: {source.metadata.get('category','N/A')}")
                            st.caption(f"Q: {source.metadata.get('question','N/A')}")

                        key = f"snip_{msg_idx}_{src_idx}_{source.metadata.get('product_id', source.metadata.get('question',''))}"
                        st.text_area(
                            label=f"Content Snippet {src_idx+1}",
                            value=(source.page_content[:500] + "...") if len(source.page_content) > 500 else source.page_content,
                            height=100,
                            disabled=True,
                            key=key
                        )

    # --- User input ---
    if user_input := st.chat_input("Ask about Nexora products, coverage, claims..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant thinking
        with st.spinner("NexoraGuard is thinking..."):
            if not rag_chain or not vector_store:
                response = {
                    "answer": "I'm sorry, the AI system isn't fully initialized. Please try again later.",
                    "sources": [], "intent": "error", "entities": {}
                }
            else:
                response = ask_nexora_guard(user_input)

        # Append and display assistant response
        assistant_msg = {
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", [])
        }
        st.session_state.messages.append(assistant_msg)

        with st.chat_message("assistant"):
            st.markdown(assistant_msg["content"])
            if show_debug_info:
                st.caption(f"Detected Intent: {response['intent']}")
                st.caption(f"Extracted Entities: {response['entities']}")

                if assistant_msg["sources"]:
                    with st.expander("View Sources Used", expanded=False):
                        new_idx = len(st.session_state.messages) - 1
                        for src_idx, source in enumerate(assistant_msg["sources"]):
                            st.markdown(
                                f"**Source {src_idx+1}: Type – {source.metadata.get('source_type','N/A')}**"
                            )
                            if source.metadata.get('source_type') == 'product_data':
                                st.caption(f"Product: {source.metadata.get('product_name','N/A')}")
                            elif source.metadata.get('source_type') == 'faq':
                                st.caption(f"FAQ Category: {source.metadata.get('category','N/A')}")
                                st.caption(f"Q: {source.metadata.get('question','N/A')}")

                            key2 = f"streamlit_{new_idx}_{src_idx}_{source.metadata.get('product_id', source.metadata.get('question',''))}"
                            st.text_area(
                                label=f"Content Snippet {src_idx+1}",
                                value=(source.page_content[:500] + "...") if len(source.page_content) > 500 else source.page_content,
                                height=100,
                                disabled=True,
                                key=key2
                            )
                else:
                    st.caption("No specific source documents were heavily relied upon for this response.")

# # To signal where the "main" execution for streamlit would start if this notebook were a script:
if __name__ == '__main__':
    run_streamlit_app()