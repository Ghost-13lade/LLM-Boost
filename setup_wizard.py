#!/usr/bin/env python3
"""
LLM Boost Setup Wizard
A Plug-and-Play installer for any platform.
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the wizard header."""
    clear_screen()
    print(f"{CYAN}{'='*50}{RESET}")
    print(f"{CYAN}{BOLD}       🚀 LLM BOOST - SETUP WIZARD       {RESET}")
    print(f"{CYAN}{'='*50}{RESET}")
    print("")


def print_step(step: int, total: int, message: str):
    """Print a step message."""
    print(f"\n{CYAN}[{step}/{total}] {message}{RESET}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message: str):
    """Print an error message."""
    print(f"{RED}✗ {message}{RESET}")


def print_info(message: str):
    """Print an info message."""
    print(f"{YELLOW}ℹ {message}{RESET}")


def install_package(package: str) -> bool:
    """Install a pip package."""
    print(f"{YELLOW}Installing {package}...{RESET}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print_success(f"{package} installed")
        return True
    except subprocess.CalledProcessError:
        print_error(f"Failed to install {package}")
        return False


def install_requirements():
    """Install base requirements from requirements.txt."""
    if os.path.exists("requirements.txt"):
        print(f"{YELLOW}Installing base requirements...{RESET}")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print_success("Base requirements installed")
            return True
        except subprocess.CalledProcessError:
            print_error("Failed to install base requirements")
            return False
    else:
        print_error("requirements.txt not found!")
        return False


def main():
    """Run the setup wizard."""
    print_header()
    
    # ==========================================
    # Step 1: Base Requirements
    # ==========================================
    print_step(1, 4, "Installing Base Dependencies")
    
    if not install_requirements():
        print_error("Setup failed. Please check your Python installation.")
        input("\nPress Enter to exit...")
        return 1
    
    print_header()
    
    # ==========================================
    # Step 2: Backend Selection
    # ==========================================
    print_step(2, 4, "Configure AI Backend")
    print("Which AI engine do you want to use?\n")
    print(f"  {BOLD}1.{RESET} Apple MLX (Fastest for Mac M1/M2/M3/M4)")
    print(f"  {BOLD}2.{RESET} External Provider (Ollama, LM Studio, OpenAI, OpenRouter)")
    print("")
    
    backend_choice = input(f"{YELLOW}Select (1/2): {RESET}").strip()
    
    llm_backend = "network"
    model_path = ""
    api_base = ""
    api_key = ""
    
    if backend_choice == "1":
        # ===== MLX SETUP =====
        if sys.platform != "darwin":
            print(f"\n{RED}⚠️  Warning: MLX is optimized for macOS. You are on {sys.platform}.{RESET}")
            confirm = input("Continue anyway? (y/n): ").lower()
            if confirm != 'y':
                print_info("Setup cancelled.")
                return 1
        
        print(f"\n{YELLOW}Installing Apple MLX...{RESET}")
        install_package("mlx-lm")
        llm_backend = "mlx"
        
        print(f"\n{GREEN}{BOLD}Recommended Models for MLX:{RESET}")
        print(f"  {CYAN}70B Models (Require 48GB+ RAM):{RESET}")
        print("    - mlx-community/Llama-3.3-70B-Instruct-4bit")
        print("    - mlx-community/Qwen2.5-72B-Instruct-4bit")
        print(f"  {CYAN}Smaller Models (8GB-16GB RAM):{RESET}")
        print("    - mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        print("    - mlx-community/Llama-3.2-3B-Instruct-4bit")
        print("")
        
        model_path = input(f"{YELLOW}Enter HuggingFace Model ID (Press Enter for Mistral-7B): {RESET}").strip()
        if not model_path:
            model_path = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        
    else:
        # ===== EXTERNAL PROVIDER SETUP =====
        print(f"\n{GREEN}{BOLD}External Provider Setup{RESET}")
        print("Compatible with: LM Studio (Local), Ollama (Local), OpenAI (Cloud), OpenRouter (Cloud)")
        print("")
        
        # Preset URLs
        print(f"{CYAN}Presets:{RESET}")
        print(f"  {BOLD}1.{RESET} LM Studio  (http://localhost:1234/v1)")
        print(f"  {BOLD}2.{RESET} Ollama      (http://localhost:11434/v1)")
        print(f"  {BOLD}3.{RESET} OpenAI      (https://api.openai.com/v1)")
        print(f"  {BOLD}4.{RESET} OpenRouter  (https://openrouter.ai/api/v1)")
        print(f"  {BOLD}5.{RESET} Custom URL")
        print("")
        
        url_choice = input(f"{YELLOW}Select Preset (1-5): {RESET}").strip()
        
        if url_choice == "1":
            api_base = "http://localhost:1234/v1"
        elif url_choice == "2":
            api_base = "http://localhost:11434/v1"
        elif url_choice == "3":
            api_base = "https://api.openai.com/v1"
        elif url_choice == "4":
            api_base = "https://openrouter.ai/api/v1"
        else:
            api_base = input(f"{YELLOW}Enter API Base URL: {RESET}").strip()
        
        # API Key
        if url_choice in ["3", "4"]:
            api_key = input(f"{YELLOW}Enter API Key: {RESET}").strip()
        else:
            api_key = input(f"{YELLOW}Enter API Key (Press Enter for 'lm-studio'): {RESET}").strip()
            if not api_key:
                api_key = "lm-studio"
        
        # Model Name
        model_path = input(f"{YELLOW}Enter Model Name (e.g., 'gpt-4', 'llama3'): {RESET}").strip()
        if not model_path:
            model_path = "local-model"
    
    # ==========================================
    # Step 3: Voice Support (Optional)
    # ==========================================
    print_header()
    print_step(3, 4, "Optional Features")
    
    voice = input(f"{YELLOW}Install Speech Recognition for voice input? (y/n): {RESET}").lower()
    if voice == 'y':
        install_package("SpeechRecognition")
        # PyAudio is optional (may fail on some systems)
        print_info("Attempting to install PyAudio (may require system dependencies)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyaudio"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    whisper = input(f"{YELLOW}Install OpenAI Whisper for better accuracy? (y/n): {RESET}").lower()
    if whisper == 'y':
        install_package("openai-whisper")
    
    # ==========================================
    # Step 4: Generate .env File
    # ==========================================
    print_header()
    print_step(4, 4, "Finalizing Configuration")
    
    env_content = f"""# LLM Boost Configuration
# Generated by setup_wizard.py

# ===========================================
# Backend Selection
# ===========================================
LLM_BACKEND={llm_backend}

# ===========================================
# Model Configuration
# ===========================================
MODEL_PATH={model_path}

# Generation settings
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=4096

# ===========================================
# Network Provider Settings
# ===========================================
API_BASE_URL={api_base}
API_KEY={api_key}

# ===========================================
# Memory Settings
# ===========================================
CHROMA_PERSIST_DIR=./data/chroma
SQLITE_DB_PATH=./data/memory.db
SEARCH_ENABLED=true
"""
    
    # Write .env file
    with open(".env", "w") as f:
        f.write(env_content)
    
    print_success(".env file created")
    
    # ==========================================
    # Complete!
    # ==========================================
    print("")
    print(f"{GREEN}{BOLD}{'='*50}{RESET}")
    print(f"{GREEN}{BOLD}       🎉 SETUP COMPLETE!       {RESET}")
    print(f"{GREEN}{BOLD}{'='*50}{RESET}")
    print("")
    print(f"{CYAN}Configuration Summary:{RESET}")
    print(f"  • Backend: {BOLD}{llm_backend}{RESET}")
    print(f"  • Model:   {BOLD}{model_path}{RESET}")
    if api_base:
        print(f"  • API URL: {BOLD}{api_base}{RESET}")
    print("")
    print(f"{CYAN}Next Steps:{RESET}")
    print(f"  {BOLD}Mac/Linux:{RESET}   ./start_mac.sh")
    print(f"  {BOLD}Windows:{RESET}     start_windows.bat")
    print(f"  {BOLD}Manual:{RESET}      streamlit run app.py")
    print("")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Setup cancelled by user.{RESET}")
        sys.exit(1)