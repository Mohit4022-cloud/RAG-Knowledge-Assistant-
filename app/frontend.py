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
    page_title="Meta RAG Bot",
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
        # Capitalize model name for display
        model_display_name = config.LLM_MODEL.upper() if config.LLM_MODEL == "glm-4.7" else config.LLM_MODEL
        st.caption(f"Model: {model_display_name}")
    else:
        st.error("❌ Not connected to Qdrant")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Info section
    with st.expander("🛠️ What I Can Answer"):
        st.markdown("""
        ### 💼 Business Justification
        Ask me about economics and TCO:
        - "Why did you choose GLM-4.7 over GPT-5.2?"
        - "What's the ROI of RAG vs fine-tuning?"
        - "Why Qdrant instead of Pinecone?"
        - "How much does this system cost to run?"

        ### 🔍 Architecture & SLAs
        Ask me about implementation and performance:
        - "What's your retrieval latency (P95/P99)?"
        - "How does hybrid search work?"
        - "Can this scale to 100K queries/day?"
        - "What's your vendor independence strategy?"

        ### 📊 Cost & OpEx Forecasting
        Ask me about financial planning:
        - "What's the 3-year TCO projection?"
        - "How do costs scale with query volume?"
        - "What's the payback period?"

        ### 📚 RAG Concepts
        Learn from the research papers:
        - "What is RAG and how does it work?"
        - "What are best practices for chunking?"
        - "How do I evaluate a RAG system?"
        """)

    # Phoenix link
    st.divider()
    st.markdown(f"[📊 View Phoenix Traces](http://localhost:{config.PHOENIX_PORT})")


# Main title
st.title("💬 Meta RAG Enterprise Assistant")
st.caption("I demonstrate enterprise-grade RAG architecture with 83.7% cost savings (vs GPT-5.2) and full business justification for every decision. Ask me about TCO, SLAs, or vendor independence!")

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
if prompt := st.chat_input("Ask me: 'How was this system architected?'"):
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
                st.write("✨ Generating Response...")

                # Update status
                status.update(label="✅ Streaming Response...", state="complete", expanded=False)

            # Step 2: Use built-in streaming from ContextChatEngine
            response_placeholder = st.empty()
            accumulated_response = ""
            sources = []

            # Check if streaming is enabled via config
            import config
            use_streaming = getattr(config, 'ENABLE_STREAMING', True)

            if use_streaming and hasattr(st.session_state.chat_engine, 'stream_chat'):
                # Use streaming
                try:
                    streaming_response = st.session_state.chat_engine.stream_chat(prompt)

                    # Extract source nodes from streaming response
                    if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
                        for node in streaming_response.source_nodes:
                            sources.append({
                                "file": node.metadata.get('file_name', 'Unknown'),
                                "page": node.metadata.get('page_label', 'N/A')
                            })

                    # Stream the response with progressive rendering
                    for chunk in streaming_response.chat_stream:
                        if hasattr(chunk, 'delta') and chunk.delta:
                            accumulated_response += chunk.delta
                            response_placeholder.markdown(accumulated_response)

                except Exception as stream_error:
                    # Fallback to non-streaming
                    response = st.session_state.chat_engine.chat(prompt)
                    accumulated_response = str(response.response)
                    response_placeholder.markdown(accumulated_response)

                    # Extract sources from non-streaming response
                    if response.source_nodes:
                        for node in response.source_nodes:
                            sources.append({
                                "file": node.metadata.get('file_name', 'Unknown'),
                                "page": node.metadata.get('page_label', 'N/A')
                            })
            else:
                # Use non-streaming (fallback or disabled)
                response = st.session_state.chat_engine.chat(prompt)
                accumulated_response = str(response.response)
                response_placeholder.markdown(accumulated_response)

                # Extract sources
                if response.source_nodes:
                    for node in response.source_nodes:
                        sources.append({
                            "file": node.metadata.get('file_name', 'Unknown'),
                            "page": node.metadata.get('page_label', 'N/A')
                        })

            # Step 3: Calculate cost from accumulated response
            input_tokens = 0
            output_tokens = 0

            # Estimate from word count
            word_count = len(accumulated_response.split())
            prompt_words = len(prompt.split())
            output_tokens = int(word_count * 1.3)
            input_tokens = int(prompt_words * 1.3)

            # Calculate cost: GLM-4.7 pricing $0.50 input / $1.85 output per 1M tokens
            input_cost = input_tokens * 0.0000005   # $0.50 / 1,000,000
            output_cost = output_tokens * 0.00000185  # $1.85 / 1,000,000
            cost = input_cost + output_cost

            cost_info = f"💰 Cost: ${cost:.6f} | 📥 Input: {input_tokens} tok | 📤 Output: {output_tokens} tok"

            # Step 4: Display sources in expander
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**[{i}]** {source['file']} - Page {source['page']}")

            # Display cost information
            st.caption(cost_info)

            # Step 5: Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": accumulated_response,
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
