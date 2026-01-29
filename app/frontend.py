"""
Streamlit Frontend for RAG Knowledge Assistant

This module provides a ChatGPT-style chat interface for the RAG system with:
- Modern chat UI using st.chat_message
- Conversation history with session state
- Source citations with filename and page numbers
- Connection status and settings sidebar
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from app.engine import get_chat_engine
import config

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_chat_engine():
    """Initialize chat engine once and cache it."""
    try:
        engine = get_chat_engine()
        return engine, True  # engine, success
    except Exception as e:
        st.error(f"Failed to initialize chat engine: {e}")
        return None, False


# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_engine" not in st.session_state:
    engine, success = initialize_chat_engine()
    st.session_state.chat_engine = engine
    st.session_state.engine_loaded = success


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    # Connection status
    if st.session_state.engine_loaded:
        st.success("✅ Connected to Qdrant")
        st.caption(f"Collection: {config.COLLECTION_NAME}")
        st.caption(f"Model: {config.LLM_MODEL}")
    else:
        st.error("❌ Not connected to Qdrant")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Info section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **RAG Knowledge Assistant**

        This app uses:
        - 🤖 GLM-4-Flash LLM
        - 🔢 HuggingFace Embeddings
        - 🔍 Hybrid Search (BM25 + Dense)
        - 📊 Phoenix Observability
        """)

    # Phoenix link
    st.divider()
    st.markdown(f"[📊 View Phoenix Traces](http://localhost:{config.PHOENIX_PORT})")


# Main title
st.title("💬 RAG Knowledge Assistant")
st.caption("Ask questions about your documents")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available (only for assistant messages)
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**[{i}]** {source['file']} - Page {source['page']}")


# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if engine is loaded
    if not st.session_state.engine_loaded:
        st.error("Chat engine not loaded. Please check your configuration.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": None
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response with glass-box observability
    with st.chat_message("assistant"):
        try:
            # Create status container for real-time observability
            with st.status("🚀 Processing Query...", expanded=True) as status:
                # Step 1: Show retrieval phase
                st.write("🔍 Retrieving Context...")

                # Step 2: Call chat engine (blocking, non-streaming)
                response = st.session_state.chat_engine.chat(prompt)

                # Step 3: Extract retrieval metrics
                num_chunks = len(response.source_nodes) if response.source_nodes else 0
                st.write(f"✅ Retrieved {num_chunks} relevant chunks")

                # Step 4: Extract sources from source_nodes
                sources = []
                if response.source_nodes:
                    for node in response.source_nodes:
                        sources.append({
                            "file": node.metadata.get('file_name', 'Unknown'),
                            "page": node.metadata.get('page_label', 'N/A')
                        })

                # Step 5: Extract token usage and calculate cost
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0
                cost = 0.0

                try:
                    # Try to extract token usage from response.raw (GLM API response)
                    if hasattr(response, 'raw') and response.raw:
                        # GLM API returns OpenAI-compatible format with usage object
                        if isinstance(response.raw, dict) and 'usage' in response.raw:
                            usage = response.raw['usage']
                            input_tokens = usage.get('prompt_tokens', 0)
                            output_tokens = usage.get('completion_tokens', 0)
                            total_tokens = usage.get('total_tokens', 0)
                        # Handle case where raw might be an object with usage attribute
                        elif hasattr(response.raw, 'usage'):
                            usage = response.raw.usage
                            input_tokens = getattr(usage, 'prompt_tokens', 0)
                            output_tokens = getattr(usage, 'completion_tokens', 0)
                            total_tokens = getattr(usage, 'total_tokens', 0)

                    # Fallback: Estimate from word count if no usage data available
                    if total_tokens == 0:
                        response_text = str(response.response)
                        # Rough estimation: 1.3 tokens per word
                        word_count = len(response_text.split())
                        prompt_words = len(prompt.split())
                        output_tokens = int(word_count * 1.3)
                        input_tokens = int(prompt_words * 1.3)
                        total_tokens = input_tokens + output_tokens

                    # Calculate cost: GLM-4-Flash pricing $0.01 / 1M tokens
                    cost_per_token = 0.00000001  # $0.01 / 1,000,000 tokens
                    cost = total_tokens * cost_per_token

                    # Store cost info for display outside status container
                    cost_info = f"💰 Cost: ${cost:.4f} | 📥 Input: {input_tokens} tok | 📤 Output: {output_tokens} tok"

                except Exception as token_error:
                    # If token extraction fails, set fallback message
                    cost_info = "⚠️ Cost tracking unavailable"

                # Step 6: Update status to show generation complete
                status.update(label="✅ Answer Generated", state="complete", expanded=False)

            # Step 7: Display complete response (non-streaming)
            response_text = str(response.response)
            st.markdown(response_text)

            # Step 8: Display sources in expander (keep existing UI)
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**[{i}]** {source['file']} - Page {source['page']}")

            # Display cost information at bottom of message
            if 'cost_info' in locals():
                st.caption(cost_info)

            # Step 9: Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources
            })

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(f"Error generating response: {e}")

            # Add error message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": None
            })
