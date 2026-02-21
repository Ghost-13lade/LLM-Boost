"""
LLM Boost - Streamlit UI
The "Viral" Interface for the LLM Boost framework.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LLM Boost components
from src.providers import get_provider, NetworkProvider
from src.memory import MemoryManager
from src.core import run_llm_boost
from src.utils import OutputParser

# Try to import mic recorder (optional)
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except ImportError:
    MIC_RECORDER_AVAILABLE = False


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="LLM Boost",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# LOAD CUSTOM CSS
# ============================================
def load_css():
    """Load custom CSS styling."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()


# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = None
    
    if "provider" not in st.session_state:
        st.session_state.provider = None
    
    # Determine default model from env
    env_backend = os.getenv("LLM_BACKEND", "network")
    if env_backend == "mlx":
        default_model = "Local (MLX)"
    else:
        default_model = "OpenAI"
    
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "model": default_model,
            "api_key": "",
            "incognito": False,
            "use_search": os.getenv("SEARCH_ENABLED", "true").lower() == "true",
            "temperature": 0.7,
            "mlx_model": os.getenv("MODEL_PATH", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        }


init_session_state()


# ============================================
# SIDEBAR - SETTINGS
# ============================================
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Model Selection
    model_options = ["Local (MLX)", "OpenAI", "LM Studio", "OpenRouter"]
    selected_model = st.selectbox(
        "Model Provider",
        model_options,
        index=model_options.index(st.session_state.settings.get("model", "OpenAI"))
        if st.session_state.settings.get("model") in model_options else 1
    )
    st.session_state.settings["model"] = selected_model
    
    # API Key Input (conditional - only for cloud providers)
    cloud_providers = ["OpenAI", "OpenRouter"]
    if selected_model in cloud_providers:
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.settings.get("api_key", ""),
            help=f"Enter your {selected_model} API key"
        )
        st.session_state.settings["api_key"] = api_key
    else:
        # For local providers, show endpoint config
        if selected_model == "LM Studio":
            lm_studio_url = st.text_input(
                "LM Studio URL",
                value="http://localhost:1234/v1",
                help="URL of your LM Studio server"
            )
            st.session_state.settings["base_url"] = lm_studio_url
        elif selected_model == "Local (MLX)":
            st.info("MLX runs locally on Apple Silicon. No API key needed.")
            mlx_model = st.text_input(
                "Model ID",
                value="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                help="HuggingFace model ID for MLX"
            )
            st.session_state.settings["mlx_model"] = mlx_model
    
    st.markdown("---")
    
    # Incognito Mode Toggle
    incognito = st.toggle(
        "🕵️ Incognito Mode",
        value=st.session_state.settings.get("incognito", False),
        help="When enabled, conversations won't be saved to memory"
    )
    st.session_state.settings["incognito"] = incognito
    
    # Web Search Toggle
    use_search = st.toggle(
        "🔍 Enable Web Search",
        value=st.session_state.settings.get("use_search", True),
        help="Enable DuckDuckGo search for up-to-date information"
    )
    st.session_state.settings["use_search"] = use_search
    
    # Temperature Slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings.get("temperature", 0.7),
        step=0.1,
        help="Higher values make output more creative, lower values more deterministic"
    )
    st.session_state.settings["temperature"] = temperature
    
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Memory Status
    st.markdown("---")
    if incognito:
        st.caption("🔒 Incognito Mode Active - Memory disabled")
    else:
        st.caption("💾 Memory Active - Conversations saved")


# ============================================
# PROVIDER INITIALIZATION
# ============================================
def get_configured_provider():
    """Get the configured provider based on settings."""
    settings = st.session_state.settings
    model = settings.get("model", "OpenAI")
    
    try:
        if model == "OpenAI":
            return NetworkProvider(
                model="gpt-4",
                api_key=settings.get("api_key") or os.getenv("OPENAI_API_KEY")
            )
        
        elif model == "OpenRouter":
            return NetworkProvider(
                model="openai/gpt-4-turbo",  # Default, can be changed
                api_key=settings.get("api_key"),
                base_url="https://openrouter.ai/api/v1"
            )
        
        elif model == "LM Studio":
            return NetworkProvider(
                model="local-model",
                base_url=settings.get("base_url", "http://localhost:1234/v1")
            )
        
        elif model == "Local (MLX)":
            # Try to import MLX provider
            try:
                from src.providers import MLXProvider
                return MLXProvider(
                    model=settings.get("mlx_model", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
                )
            except ImportError:
                st.error("MLX not available. Install with: pip install mlx-lm")
                return None
        
    except Exception as e:
        st.error(f"Failed to initialize provider: {e}")
        return None
    
    return None


# ============================================
# OUTPUT PARSING AND DISPLAY
# ============================================
def parse_and_display_response(response_text: str):
    """
    Parse tagged output and display in a structured way.
    
    Shows reasoning in expanders, final answer prominently.
    Supports both <tag> and :colon: style tags.
    """
    import re
    
    parser = OutputParser()
    parsed = parser.parse(response_text)
    
    # Extract XML-style tags
    thinking = parsed.thinking or parser.extract_tag(response_text, "thinking")
    analysis = parsed.analysis or parser.extract_tag(response_text, "analysis")
    answer = parsed.answer or parser.extract_tag(response_text, "answer")
    
    # Extract :colon: style tags (God Mode prompt format)
    fact_check = re.search(r':fact-check:\s*(.+?)(?=:[\w-]+:|Confidence:|$)', response_text, re.DOTALL)
    logic = re.search(r':logic:\s*(.+?)(?::[\w-]+:|Confidence:|$)', response_text, re.DOTALL)
    creative = re.search(r':creative:\s*(.+?)(?=:[\w-]+:|Confidence:|$)', response_text, re.DOTALL)
    debate = re.search(r':debate:\s*(.+?)(?=:[\w-]+:|Confidence:|$)', response_text, re.DOTALL)
    final_answer = re.search(r':final answer:\s*(.+?)(?=Confidence:|$)', response_text, re.DOTALL | re.IGNORECASE)
    confidence = re.search(r'Confidence:\s*([\d.]+)\s*/\s*10', response_text, re.IGNORECASE)
    
    # Show reasoning in expander
    has_reasoning = thinking or analysis or fact_check or logic or creative or debate
    
    if has_reasoning:
        with st.expander("🎭 Show Agent Debate", expanded=False):
            # Harper (The Skeptic) - Fact Check
            if fact_check:
                st.markdown("### 🛡️ Harper (The Skeptic)")
                st.markdown("*Fact-checker & Hallucination Killer*")
                st.markdown(fact_check.group(1).strip())
                st.markdown("---")
            
            # Benjamin (The Architect) - Logic
            if logic:
                st.markdown("### 📐 Benjamin (The Architect)")
                st.markdown("*Logic & First-Principles Thinker*")
                st.markdown(logic.group(1).strip())
                st.markdown("---")
            
            # Lucas (The Visionary) - Creative
            if creative:
                st.markdown("### 🎨 Lucas (The Visionary)")
                st.markdown("*Lateral Thinker*")
                st.markdown(creative.group(1).strip())
                st.markdown("---")
            
            # Debate Summary
            if debate:
                st.markdown("### ⚖️ Debate Summary")
                st.markdown(debate.group(1).strip())
                st.markdown("---")
            
            # Legacy tags (backward compatibility)
            if thinking and not logic:
                st.markdown("**💭 Thinking:**")
                st.markdown(thinking)
                st.markdown("")
            
            if analysis and not fact_check:
                st.markdown("**📊 Analysis:**")
                st.markdown(analysis)
    
    # Show the main answer prominently
    display_answer = answer or (final_answer.group(1).strip() if final_answer else None)
    
    if display_answer:
        st.markdown("### 🎯 Final Answer")
        st.markdown(display_answer)
        
        # Show confidence if available
        if confidence:
            score = float(confidence.group(1))
            if score >= 8:
                st.success(f"**Confidence:** {score}/10 ✅")
            elif score >= 5:
                st.warning(f"**Confidence:** {score}/10 ⚠️")
            else:
                st.error(f"**Confidence:** {score}/10 ❌")
    else:
        # If no structured tags, show the raw response
        # But try to clean it up
        clean_text = parser.remove_tags(response_text)
        if clean_text.strip():
            st.markdown(clean_text)
        else:
            st.markdown(response_text)


# ============================================
# MAIN AREA - CHAT INTERFACE
# ============================================
st.title("🧠 LLM Boost")
st.caption("Maximize your LLM's intelligence with memory, search, and structured reasoning.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            parse_and_display_response(message["content"])
        else:
            st.markdown(message["content"])

# ============================================
# CHAT INPUT
# ============================================
# Create columns for text input and mic button
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.chat_input("Type your message...")

with col2:
    # Microphone button (placeholder or functional)
    if MIC_RECORDER_AVAILABLE:
        st.markdown("### ")  # Spacing
        mic_audio = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹️",
            just_once=True,
            use_container_width=True
        )
        if mic_audio:
            # Process audio
            try:
                from src.utils import transcribe_audio
                audio_bytes = mic_audio["bytes"]
                transcription = transcribe_audio(audio_bytes)
                if transcription:
                    user_input = transcription  # Use transcribed text
            except Exception as e:
                st.warning(f"Voice transcription failed: {e}")
    else:
        # Placeholder button
        st.markdown("### ")
        if st.button("🎤", help="Voice input (install streamlit-mic-recorder)"):
            st.info("Install streamlit-mic-recorder for voice input: pip install streamlit-mic-recorder")

# ============================================
# PROCESS USER INPUT
# ============================================
if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get provider
    provider = get_configured_provider()
    
    if provider is None:
        st.error("Failed to initialize LLM provider. Check your settings.")
    else:
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = run_llm_boost(
                        user_message=user_input,
                        provider=provider,
                        use_search=st.session_state.settings.get("use_search", True),
                        use_memory=not st.session_state.settings.get("incognito", False),
                        incognito=st.session_state.settings.get("incognito", False),
                        temperature=st.session_state.settings.get("temperature", 0.7)
                    )
                    
                    # Parse and display the response
                    parse_and_display_response(response)
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.exception(e)


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption(
    "LLM Boost - An intelligent LLM wrapper with memory, search, and reasoning. "
    "Built with ❤️ using Streamlit."
)