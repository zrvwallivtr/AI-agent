from pathlib import Path
from datetime import datetime

from src.agent.models.llm import LLM
from src import config


def _project_path(project_name):
    PROJECT_DIR = config.PROJECTS_DIR / project_name
    return PROJECT_DIR


# ----------------------------------------------
# Main abilities:
# - create summary from disgustions
# - create task list
# - decision logging
#
# Note: The project name must be specified.
# ----------------------------------------------


# class ProjectManager:
#     def __init__(self, project_name):
#         self.project_name   = project_name
#         self.path           = _project_path(project_name)
# 
#     def create_project_workspace(self):
#         """
#         Create project workspace, includes:
# 
#         project directory + all required files
#         """
#         if self.path.exists():
#             print(f"Project '{self.project_name}' already exist")
#             return
#         # -- project directory --
#         self.path.mkdir(parents=True, exist_ok=True)
#         # -- all required files --
#         all_files = ["Summary", "Tasks", "Decisions"]
#         for filename in all_files:
#             (self.path / f"{filename}.md").write_text(f"# {filename}\n")
#         print(f"New project {self.project_name} created")
# 
#     def summarize_project(self, question, session=None):
#         """
#         Summarize project based on:
# 
#         PM_SYS_PROMPT + additional prompt + chat history + question
#         """
#         if not self.path_check("Summary"):
#             return None
#         chat        = Chat(session=session)
#         messages    = [
#             LLM.system(config.PM_PROMPT),
#             LLM.user("Summarise the project based on the following conversations."),
#             chat.to_llm,
#             LLM.user(question),
#         ]
#         response = LLM.model_response(messages, model=config.PM_MODEL)
#         self.add_entry("Summary", response)
#         return response
# 
#     def generate_tasklist(self, session=None):
#         """
#         Generate tasklist based on:
# 
#         PM_SYS_PROMPT + additional prompt + chat Summary.md
#         """
#         if not self.path_check("Tasks"):
#             return None
#         summary_txt = self.read_file("Summary")
#         tasks       = self.read_file("Tasks")
#         messages    = [
#             LLM.system(config.PM_PROMPT),
#             LLM.user(
#                 "Create/update the task list in markdown checkbox format (- [ ] task), ordered by priority based on the following summary and tasklist."
#             ),
#             LLM.user("Summary:"),
#             LLM.user(summary_txt),
#             LLM.user("Tasks:"),
#             LLM.user(tasks),
#         ]
#         response    = LLM.model_response(messages, model=config.PM_MODEL)
#         self.write("Tasks", response)
#         return response
# 
#     def add_decisions(self, question, session=None):
#         """
#         Logs all the decision made based on:
# 
#         PM_SYS_PROMPT + additional prompt + Decisions + chat history + question
#         """
#         if not self.path_check("Decisions"):
#             return None
#         decisions_txt   = self.read_file("Decisions")
#         chat            = Chat(session=session)
#         messages        = [
#             LLM.system(config.PM_PROMPT),
#             LLM.user("Here is the current decision list:"),
#             LLM.user(decisions_txt),
#             LLM.user(
#                 "Add descisions made that are NOT in the decision list based on the following converstations:"
#             ),
#             chat.to_llm,
#             LLM.user(question),
#         ]
#         response        = LLM.model_response(messages, model=config.PM_MODEL)
#         self.add_entry("Decisions", response)
#         return response
# 
#     # -- read, edit and write files ---------------------------------------
# 
#     def read_file(self, filename):
#         if not self.path_check(filename):
#             return None
#         file_path = self.path / f"{filename}.md"
#         return file_path.read_text()
# 
#     def add_entry(self, filename, content):
#         if not self.path_check(filename):
#             return None
#         file_path   = self.path / f"{filename}.md"
#         time        = datetime.now().strftime("%d-%m-%Y %H:%M")
#         entry       = f"\n## {time}\n{content}\n"
#         with open(file_path, "a") as file:
#             file.write(entry)
# 
#     def write(self, filename, content):
#         file_path = self.path / f"{filename}.md"
#         file_path.write_text(content)
# 
#     # -- path management -------------------------------------------------
# 
#     def path_check(self, filename):
#         """
#         Return True/False to check if file exists,
#         if file does not exists create file
#         """
#         # -- check if PROJECT_DIR exists --
#         if not self.path.exists():
#             print(f"{self.project_name} does not exist")
#             return False
#         file_path = self.path / f"{filename}.md"
#         # -- create new file if filename.md does not exists --
#         if not file_path.exists():
#             file_path.write_text(f"# {filename}\n")
#             print(f"{filename}.md not found - created it in {self.project_name}")
#         return True
