import sys
from unittest.mock import MagicMock

# ----------------------------------------------------
# Dynamic sys.modules Mocks to run without heavy ML dependencies
# ----------------------------------------------------
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends = MagicMock()
mock_torch.backends.mps = MagicMock()
mock_torch.backends.mps.is_available.return_value = False
sys.modules['torch'] = mock_torch

mock_transformers = MagicMock()
sys.modules['transformers'] = mock_transformers

mock_whisper = MagicMock()
sys.modules['whisper'] = mock_whisper

mock_sr = MagicMock()
sys.modules['speech_recognition'] = mock_sr

mock_pyttsx3 = MagicMock()
sys.modules['pyttsx3'] = mock_pyttsx3

import os
import pytest
from unittest.mock import patch
import json

# Import the module under test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tools
import assistant
import assistant_ollama_liquid

@pytest.fixture(autouse=True)
def cleanup_files():
    """Removes temporary files before and after each test."""
    for filename in ["todos.txt", "notes.txt"]:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
    yield
    for filename in ["todos.txt", "notes.txt"]:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass

# ----------------------------------------------------
# 1. Tests for tools.py
# ----------------------------------------------------
def test_add_todo():
    res = tools.add_todo("Buy groceries", "tomorrow")
    assert "Buy groceries" in res
    assert "tomorrow" in res
    assert os.path.exists("todos.txt")

    with open("todos.txt", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Buy groceries" in content

def test_list_todos():
    # Empty case
    assert tools.list_todos() == "Your todo list is empty."

    # Non-empty case
    tools.add_todo("Write tests")
    res = tools.list_todos()
    assert "Your current tasks:" in res
    assert "Write tests" in res

def test_add_note():
    res = tools.add_note("Meeting Notes", "Discuss architecture with team.")
    assert "Meeting Notes" in res
    assert os.path.exists("notes.txt")

    with open("notes.txt", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Discuss architecture with team." in content

def test_list_notes():
    # Empty case
    assert tools.list_notes() == "No notes found in your notebook."

    # Non-empty case
    tools.add_note("Project Vision", "To build local agent systems")
    res = tools.list_notes()
    assert "Project Vision" in res

def test_execute_tool():
    # add_todo tool
    res = tools.execute_tool("add_todo", {"task": "Verify CI pipeline", "due": "soon"})
    assert "Verify CI pipeline" in res

    # add_note tool
    res = tools.execute_tool("add_note", {"title": "CI Tip", "content": "Always mock external services"})
    assert "CI Tip" in res

    # list tools
    assert "Verify CI pipeline" in tools.execute_tool("list_todos", {})
    assert "CI Tip" in tools.execute_tool("list_notes", {})

    # Edge cases
    assert "Error:" in tools.execute_tool("add_todo", {})
    assert "Error:" in tools.execute_tool("add_note", {"title": "Title Only"})
    assert "Unknown action" in tools.execute_tool("invalid_action", {})


# ----------------------------------------------------
# 2. Tests for tool execution parsing in assistants
# ----------------------------------------------------
def test_parse_and_execute_tool_success():
    llm_output_todo = '{"action": "add_todo", "params": {"task": "Learn FastAPI", "due": "today"}}'
    res, triggered = assistant.parse_and_execute_tool(llm_output_todo)
    assert triggered is True
    assert "Learn FastAPI" in res

    llm_output_note = 'Some random preamble. {"action": "add_note", "params": {"title": "Ollama Info", "content": "LiquidAI runs fast"}} and a postamble'
    res, triggered = assistant.parse_and_execute_tool(llm_output_note)
    assert triggered is True
    assert "Ollama Info" in res

def test_parse_and_execute_tool_failure():
    # No json block
    res, triggered = assistant.parse_and_execute_tool("Hello world! I am here to help.")
    assert triggered is False
    assert res is None


# ----------------------------------------------------
# 3. Tests for assistant.py with mocked LLM and Voice
# ----------------------------------------------------
@patch("assistant.AutoTokenizer.from_pretrained")
@patch("assistant.AutoModelForCausalLM.from_pretrained")
def test_call_llm_transformers(mock_model_class, mock_tokenizer_class):
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"
    mock_tokenizer.decode.return_value = "Mocked Assistant Response"
    mock_tokenizer_class.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.generate.return_value = [MagicMock()]
    mock_model_class.return_value = mock_model

    # Inject mocked device
    with patch("assistant.device", "cpu"):
        res = assistant.call_llm("Hello", mock_model, mock_tokenizer, [])
        assert res == "Mocked Assistant Response"

@patch("pyttsx3.init")
@patch("whisper.load_model")
def test_voice_system_init(mock_whisper, mock_pyttsx3):
    mock_engine = MagicMock()
    mock_pyttsx3.return_value = mock_engine

    system = assistant.VoiceSystem()
    assert system.tts is mock_engine
    assert system.stt_model is not None


# ----------------------------------------------------
# 4. Tests for assistant_ollama_liquid.py with mocked requests
# ----------------------------------------------------
@patch("requests.post")
def test_call_llm_ollama(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "This is an Ollama response."}
    mock_post.return_value = mock_resp

    res = assistant_ollama_liquid.call_llm("Who are you?", [])
    assert res == "This is an Ollama response."

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == assistant_ollama_liquid.OLLAMA_URL
