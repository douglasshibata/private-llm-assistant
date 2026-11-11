import os
from datetime import datetime

TODO_FILE = "todos.txt"
NOTE_FILE = "notes.txt"

def add_todo(task: str, due: str = "today") -> str:
    """Adds a task to the todo list."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TODO_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] TODO: {task} (Due: {due})\n")
    return f"Task added successfully: '{task}' due {due}."

def list_todos() -> str:
    """Lists all tasks from the todo list."""
    if not os.path.exists(TODO_FILE) or os.path.getsize(TODO_FILE) == 0:
        return "Your todo list is empty."
    with open(TODO_FILE, "r", encoding="utf-8") as f:
        todos = f.readlines()
    return "Your current tasks:\n" + "\n".join([f"- {line.strip()}" for line in todos])

def add_note(title: str, content: str) -> str:
    """Saves a note to the notebook."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(NOTE_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {title}: {content}\n---\n")
    return f"Note saved successfully: '{title}'."

def list_notes() -> str:
    """Lists all saved notes."""
    if not os.path.exists(NOTE_FILE) or os.path.getsize(NOTE_FILE) == 0:
        return "No notes found in your notebook."
    with open(NOTE_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    return "Saved Notes:\n" + content

def execute_tool(action: str, params: dict) -> str:
    """Executes a tool action with the given parameters."""
    try:
        if action == "add_todo":
            task = params.get("task", "")
            due = params.get("due", "today")
            if not task:
                return "Error: 'task' parameter is required for add_todo."
            return add_todo(task, due)
        elif action == "list_todos":
            return list_todos()
        elif action == "add_note":
            title = params.get("title", "Untitled Note")
            content = params.get("content", "")
            if not content:
                return "Error: 'content' parameter is required for add_note."
            return add_note(title, content)
        elif action == "list_notes":
            return list_notes()
        else:
            return f"Unknown action: '{action}'."
    except Exception as e:
        return f"Failed to execute tool action: {e}"
