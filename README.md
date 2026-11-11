# Private LLM Assistant (Text + Voice)

An ultra-fast, local, and private personal LLM assistant with built-in voice interface (Speech-to-Text and Text-to-Speech) and daily-task automation tools. It runs completely offline on your own machine.

---

## 🌟 Main Purpose

The **Private LLM Assistant** is designed to provide users with a secure, offline digital concierge that can organize ideas, manage to-do lists, save notes, and run conversational brainstorms. By prioritizing local execution (either through PyTorch `transformers` or standard `Ollama` API), it ensures that all conversations, system prompts, and personal data remain 100% private and never leave your local environment.

---

## 🧰 Technologies Used (Tech Stack)

- **Language:** Python 3.12+
- **Local Inference Options:**
  - `transformers` + PyTorch (supports automatic GPU detection for CUDA/MPS or CPU execution)
  - `Ollama` API (communicates with local Ollama service)
- **Primary Model:** `LiquidAI/LFM2.5-1.2B-Instruct` (or `llama3.2:1b` / other small, high-throughput instruct models)
- **Speech-to-Text (STT):** `openai-whisper` (local offline audio transcription)
- **Text-to-Speech (TTS):** `pyttsx3` (cross-platform, offline synthetic voice generation)
- **Audio Capturing:** `SpeechRecognition`
- **Testing:** `pytest` + `pytest-mock`

---

## 📂 Project Structure

```text
private-llm-assistant/
├── assistant.py               # Backend Option A: PyTorch Transformers based runner
├── assistant_ollama_liquid.py  # Backend Option B: Ollama API based runner
├── tools.py                   # Modular Daily-Task Tools (Todos & Notebook Notes)
├── requirements.txt           # Dependency definition
├── README.md                  # This documentation guide
└── tests/
    └── test_assistant.py      # Full unit test suite with mock-based validation
```

---

## 🛠️ Daily-Task Tools

The assistant features safe local tool-calling capabilities. When the LLM detects that the user wants to manage their daily tasks, it returns structured JSON command blocks that are parsed and executed locally on the system:

- **Todo List (`todos.txt`):**
  - `add_todo`: Adds a new task with a custom description and due date (e.g. `{"action": "add_todo", "params": {"task": "Buy milk", "due": "today"}}`).
  - `list_todos`: Lists all existing pending tasks.
- **Notebook (`notes.txt`):**
  - `add_note`: Appends a new note with a title and content to your local notebook.
  - `list_notes`: Summarizes or lists all saved notes.

---

## 🚀 How to Deploy & Run

### 1. Installation

First, navigate to the project directory and install the necessary dependencies:

```bash
cd private-llm-assistant
pip install -r requirements.txt
```

*(Note: On Windows, if `pyaudio` fails to compile, install pipwin first: `pip install pipwin && pipwin install pyaudio`)*

---

### 2. Running Option A: PyTorch Transformers (GPU/CPU-aware)

This runner directly loads the LiquidAI model into your system RAM/VRAM using PyTorch:

```bash
python assistant.py
```

- If a compatible NVIDIA GPU with CUDA or Apple Silicon Mac (MPS) is available, it automatically maps execution to the GPU for faster generation.
- Otherwise, it falls back gracefully to CPU.

---

### 3. Running Option B: Ollama Backend (Highly Recommended for Speed)

First, download and install [Ollama](https://ollama.com).

Pull the fast, ultra-concise model from the model registry:

```bash
ollama pull LiquidAI/lfm2.5-1.2b-instruct
```

Ensure Ollama is running, then start the assistant:

```bash
python assistant_ollama_liquid.py
```

Ollama automatically manages optimal CPU/GPU offloading layers internally for lightning-fast answers.

---

## 🎙️ Interactive Modes

Once running, the assistant operates in **Voice Mode** by default:
- **Speech-to-Text:** The assistant will listen to your voice and transcribe it via local Whisper.
- **Switch to Text:** Say `"text mode"` at any time to switch to typed interactions.
- **Switch back to Voice:** Type `"voice mode"` in text mode to toggle voice recording.
- **Exit:** Say/type `"stop"`, `"quit"`, or `"exit"` to shut down safely.

---

## 🧪 Running Automated Tests

To ensure everything is correctly configured and working, run the test suite:

```bash
pytest tests/
```

The tests leverage standard mocking mechanisms to stub out hardware/microphone dependencies, allowing them to verify all prompt orchestration and tool-calling code structure in milliseconds without requiring hardware setup!
