import os
import json
from datetime import datetime
import mesop as me
import mesop.labs as mel
import random
import time
import base64
import pdfplumber
import markdownify
from dotenv import load_dotenv, set_key
# Pydantic stype function  argument type specifications
from typing import List, Dict, Union, Optional, Callable
# tweeter library
import tweepy
# Langfuse
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.callback import CallbackHandler
# LLMS
from groq import Groq
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from openai import OpenAI
# have another thread for the kickoff function output to be displayed in the webui
import threading
import time
# asynchronous process to get stdout from agents and display to frontend
import sys
import asyncio
from io import StringIO
# import agents module
import subprocess
from subprocess import Popen
import threading
# logging
import logging
# watchdog that is going to check for the log file changes and append to state the content
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# this to avoid contet error
from flask import current_app, Flask


app = Flask(__name__)

# env vars
load_dotenv(dotenv_path='.env', override=False)
# Load dynamic environment variables from dynamic.env file
load_dotenv(dotenv_path='.dynamic.env', override=True)

#### LLMS VARS
# OLLAMA LLM
ollama_llm = Ollama(model="mistral:7b")
# oepnai mimmicing LLMS
LM_OPENAI_API_BASE = os.getenv("LM_OPENAI_API_BASE")
LM_OPENAI_MODEL_NAME = os.getenv("LM_OPENAI_MODEL_NAME")
LM_VISION_MODEL_NAME = os.getenv("LM_VISION_MODEL_NAME")
LM_OPENAI_API_KEY = os.getenv("LM_OPENAI_API_KEY")
lmstudio_llm = OpenAI(base_url=LM_OPENAI_API_BASE, api_key=LM_OPENAI_API_KEY)
lmstudio_llm_for_agent = ChatOpenAI(openai_api_base=LM_OPENAI_API_BASE, openai_api_key=LM_OPENAI_API_KEY, model_name="NA", temperature=float(os.getenv("GROQ_TEMPERATURE")))
openai_llm = ChatOpenAI() #OpenAI()
# GROQ LLM
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
groq_client=Groq()
groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")))
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")))
groq_llm_llama3_70b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")))
groq_llm_gemma_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_GEMMA_7B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")))

#### LANGFUSE VARS
# langfuse init
langfuse = Langfuse(
  secret_key=os.getenv("LANG_SECRET_KEY"),
  public_key=os.getenv("LANG_PUBLIC_KEY"),
  host=os.getenv("LANG_HOST")
)
langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANG_SECRET_KEY"),
    public_key=os.getenv("LANG_PUBLIC_KEY"),
    host=os.getenv("LANG_HOST"), # just put localhost here probably
)


#### STATES CLASSES

@me.stateclass
class FileState:
  name: str
  size: int
  mime_type: str
  contents: str
  pdf_file_path: str

@me.stateclass
class FormState:
  consumer_key: bool = False
  consumer_secret: bool = False
  access_token: bool = False
  access_token_secret: bool = False
  is_submitted: bool = False
  last_modified: str = ""
  result: Dict[str, str|bool]


@me.stateclass
class PageState:
  sidenav_open: bool = True

@me.stateclass
class TweetState:
  tweet_chat_message = ""
  tweet_message: str = ""
  tweet_posted: Dict[str, str]

@me.stateclass
class AgentState:
  agent_messages: List[str]
  tweet_agents: bool = False
  report_agents: bool = False
  

@me.stateclass
class PopupState:
  popup_visible: bool = False
  popup_message: str = ""
  popup_style: Dict[str, str]
  popup_agent_visible: bool = False
  agent_work_done: bool = False  # Add this to track completion

# INPUT EVENT MANAGEMENT VARS
tweet_auth_env_dict = {
    "CONSUMER_KEY": "",
    "CONSUMER_SECRET": "",
    "ACCESS_TOKEN": "",
    "ACCESS_TOKEN_SECRET": ""
}

result = {
  "state_update_status": False,
  "state_updated": "",
  "state_update_message": "",
  "last_modified": "",
  "notify_user": "",
  "error": ""
}

# Env Vars to set
path = "./docs/World_Largest_Floods_paper.pdf"
SIDENAV_WIDTH = 350
tweet_state = ""
chat_response_message = ""
topic = ""
env_file=".dynamic.env"
stop_log_reader = False
log_file = os.getenv("LOG_FILE")

############################################################
##### BUSINESS LOGIC HELPER FUNCTIONS
# Append new messages to the AGENT_MESSAGES environment variable in the specified .env file.

#@observe()
#def append_to_agent_messages(env_path: str, new_messages: List[str]):
"""
    Append new messages to the AGENT_MESSAGES environment variable in the specified .env file.

    Args:
        env_path (str): The path to the .env file.
        new_messages (list): The new messages to append.
"""
    # Load the current environment variables from the file
#    load_dotenv(env_path, override=True)
    
    # Get the current value of AGENT_MESSAGES
#    current_messages = os.getenv('AGENT_MESSAGES', '[]')
    
    # Convert the current value to a list
#    try:
#        current_messages_list = json.loads(current_messages)
#        if not isinstance(current_messages_list, list):
#            raise ValueError("AGENT_MESSAGES is not a list.")
#    except json.JSONDecodeError:
#        current_messages_list = []
    
    # Append the new messages
#    current_messages_list.extend(new_messages)
    
    # Convert the updated list back to a JSON string
#    updated_messages = json.dumps(current_messages_list)
    
    # Update the environment variable in the .env file
#    set_key(env_path, 'AGENT_MESSAGES', updated_messages)
#    print("AGENT_MESSAGES updated successfully.")
    
    # Refresh the Mesop state
#    popup_state = me.state(PopupState)
#    agent_state = me.state(AgentState)
#    agent_state.agent_messages = current_messages_list
#    popup_state.popup_agent_visible = True

#    print("Popup state updated successfully.")

# Manage agents state as they are working in order to show their though in a popup
@observe()
def show_agent_popup(state, e: me.ClickEvent):
  print("Inside show popup agent")
  state.popup_agent_visible = True
  # set the popup_agent_visible shared state env varsto true
  set_key(".dynamic.env", "POPUP_AGENT_VISIBLE", "True")
  load_dotenv('.dynamic.env', override=True)
  print("Inside show_agent_popup, state: ", state, "popup agent visible state: ", state.popup_agent_visible, "env var popup_agent_visible: ", os.getenv("POPUP_AGENT_VISIBLE"))

@observe()
def close_agent_popup(state, e: me.ClickEvent):
    state.popup_agent_visible = False
    # reset the agent_work_done shared state env vars
    set_key(".dynamic.env", "AGENT_WORK_DONE", "False")
    set_key(".dynamic.env", "POPUP_AGENT_VISIBLE", "False")
    load_dotenv('.dynamic.env', override=True)
    print("Popup agents closed, state reinitialized to agent_work_done=False and popup_agent_visible=False")
@observe()
def add_agent_message(state, message): # we keep this function if we need to use it
    state.agent_messages.append(message)

# Function to show popup message
@observe()
def show_popup(state, message, message_type):
  styles = {
    "error": {"color": "red"},
    "validated": {"color": "green"},
    "info": {"color": "blue"}
  }
  state.popup_message = message
  state.popup_style = styles.get(message_type, {"color": "black"})
  state.popup_visible = True
  print(f"Popup State Updated: {state}")

@observe()
def close_popup(state, e: me.ClickEvent):
  state.popup_visible = False
  state.popup_message = ""
  state.popup_style = {}
  print(f"Pop up closed: All state fields reset -> {state}")

#### BACKGROUND READ LOGS ######
# start sub process to read log file written by agents and update AgentState agent_messages
#@observe()
#def read_log_file(app, log_file_path):
#  with app.app_context():
#    print("Inside read log file app context")
#    agent_state = me.state(AgentState)
#    time.sleep(2)
#    with open(log_file_path, "r", encoding="utf-8") as log_file:
#      while not stop_log_reader:
#        where = log_file.tell()
#        line = log_file.readline()
#        if not line:
#          time.sleep(0.1)  # Sleep briefly to avoid busy waiting
#          log_file.seek(where)  # Go back to the last read position
#        else:
#          # Update the Mesop state with the new log line
#          agent_state.agent_messages.append(line.strip())
#          # Process the log line
#          print(line.strip())

#@observe()
#def start_background_log_reader(app, log_file_path):
#  print("inside start background log reader")
#  global stop_log_reader
#  stop_log_reader = False
#  if log_file_path is None:
#    raise ValueError("LOG_FILE path is not set. Please check your environment variables.")
#  log_reader_thread = threading.Thread(target=read_log_file, args=(app, log_file_path))
#  log_reader_thread.start()
#  return log_reader_thread

#@observe()
#def stop_background_log_reader():
#  global stop_log_reader
#  stop_log_reader = True

#### BACKGROUND RUN AGENTS ######
@observe()
def run_agent_process(command):
  print("Inside run agent process")
  full_command = f"bash -c '{command}'"
  agent_process = Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
  print("Inside run agent process waiitng for agent_process")
  stdout, stderr = agent_process.communicate()  # Capture the output
  print(f"stdout: {stdout}")
  print(f"stderr: {stderr}")
  #agent_process.wait()
  return agent_process

#### WATCHDOG PROCESS CHECKING LOG FILE FOR AGENT NEW OUTPUT TO UPDATE STATE
@observe()
class LogFileHandler(FileSystemEventHandler):
  def __init__(self, log_file):
    self.log_file = log_file
    self.last_position = 0
    self.agent_state = me.state(AgentState)

  def on_modified(self, event):
    if event.src_path == self.log_file:
      with open(self.log_file, "r", encoding="utf-8") as f:
        f.seek(self.last_position)
        new_lines = f.readlines()
        self.last_position = f.tell()

        for line in new_lines:
          line = line.strip()
          if line:
            self.agent_state.agent_messages.append(line)
            print("agent state messages updated with :", line)  # for debugging
            print("Agent state messages is now: ", self.agent_state.agent_messages)

@observe()
def start_log_observer(log_file):
  event_handler = LogFileHandler(log_file)
  observer = Observer()
  observer.schedule(event_handler, path=os.path.dirname(log_file), recursive=False)
  observer.start()
  return observer

def output_agent_though():
  for message in popup_state.agent_messages:
    yield message

#### START BACKGROUND PROCESSES LOG READ AND AGENT JOB ######
@observe()
def start_agents(app):
  with app.app_context():
    print("Inside app context start_agent")
    popup_state = me.state(PopupState)
    agent_state = me.state(AgentState)
    popup_state.popup_agent_visible = True
    if popup_state.popup_agent_visible:
      command = "source /home/creditizens/mesop/agents_venv/bin/activate && /home/creditizens/mesop/agents_venv/bin/python3 /home/creditizens/mesop/agents.py"

      log_file = os.getenv("LOG_FILE")
      observer = start_log_observer(log_file)

      # Start agent process in a separate thread
      agent_thread = threading.Thread(target=run_agent_process, args=(command,))
      print("Inside app context start_agent: agent_thread start line")
      agent_thread.start()

      # Wait for the agent process to complete
      agent_thread.join()

      # Stop log reading
      observer.stop()
      observer.join()
        
      set_key('.dynamic.env', 'AGENT_WORK_DONE', 'True')
      load_dotenv('.dynamic.env', override=True)
      agent_state.agent_work_done = True
      print("Inside start_agents function: all states updated again even if already done in post tweet before, therefore, env vars and state updated: popup_agent_visible=True , agent_work_done=True")


# post tweet
# Function to post a tweet using saved tokens
# add here llm agent call to work on the tweet
@observe()
def post_tweet(e: me.ClickEvent, app=app):

    # get the needed states
    tweet_state = me.state(TweetState)
    popup_state = me.state(PopupState)
    agent_state = me.state(AgentState)
    from_state = me.state(FormState)
    """
    try:
      show_agent_popup(popup_state)
      popup_state.popup_agent_visible = True
      print("*************************  ", popup_state.popup_agent_visible, "  ******************************")
    except Exception as e:
      return f"An error occured while trying to show popup agent: {e}"
    """
    ## Set env vars for agent input
    print("Tweet State Raw: ", tweet_state, type(tweet_state))
    tweet_state_json_str = json.dumps(tweet_state.__dict__)
    print("Tweet State Json: ", tweet_state_json_str, type(tweet_state_json_str))
    agent_input_vars_dynamic = {
      "CHAT_RESPONSE_MESSAGE": tweet_state.tweet_chat_message,
      "TWEET_STATE": tweet_state_json_str
    }
    print("Agent Input Vars Dynamic", agent_input_vars_dynamic)
    create_dynamic_env(".dynamic.env", agent_input_vars_dynamic)
    
    # flag here for the agents that it is the tweet agents that are going to work and not the report ones. we need to use shared env vars.
    set_key(env_file, "TWEET_AGENTS", "True")
    load_dotenv('.dynamic.env', override=True)
    print(f"Check if TWEET_AGENTS env var updated to 'True': {os.getenv('TWEET_AGENTS')}")
    
    # start tweet agent workers. They will update the state tweet_message which is the final tweet posted catched later on, on this function
    if from_state.consumer_key and from_state.consumer_secret and from_state.access_token and from_state.access_token_secret:
      if tweet_state.tweet_chat_message:
        with app.app_context():
          try:
            print("Inside post_tweet agent job will start")
            # start agent work
            set_key(".dynamic.env", "AGENT_WORK_DONE", "False")
            load_dotenv('.dynamic.env', override=True)
            agent_state.agent_work_done = False
            # Run your agent tasks
            # Start agents
            agent_thread = threading.Thread(target=start_agents, args=(app,))
            agent_thread.start()
            agent_thread.join()
            agent_state.agent_work_done = True
          except Exception as e:
            return {"error": f"Tweet Worker Agents had an issue: {e}"}
        
      else:
        show_popup(popup_state, "You need a chat response to have something to tweet from.", "error")
    
    else:
      show_popup(popup_state, "Tokens/Secrets Required! Please fill and save form before posting tweet.", "error")

"""
    # tweeter authentication
    auth = tweepy.OAuth1UserHandler(
        # No tweet will be sent for the moment we need to make sure that agent process works fine 
        # and that outputs in popup in frontend that works fine as well before trying to post any tweet
        "",
        "",
        "",
        ""
        #os.getenv("CONSUMER_KEY"),
        #os.getenv("CONSUMER_SECRET"),
        #os.getenv("ACCESS_TOKEN"),
        #os.getenv("ACCESS_TOKEN_SECRET")
    )
    api = tweepy.API(auth)
    try:
      # here have the agent team kickoff that have already done their job and saved the tweet to be posted to the state variable tweet_message
      tweet_message = os.getenv("TWEET_MESSAGE")
      response = api.update_status(tweet_message) # here change the message with the llm agent response and pass the state message to the llms
      print("Tweet posted and tokens cleared.")
      tweet_state.tweet_posted = {
        "tweet": response.text,
        "create_at": response.created_at,
        "user_name": response.user.name,
      }
      print("Tweet State: ", tweet_state)
      message_type = "validated"
      # delete env vars
      # clear_tokens() # we don't use this for the moment as we want to keep the keys in the env vars for testing purpose
      # show the popup message
      show_popup(popup_state, tweet_state.tweet_posted, message_type)
      agent_state.tweet_agents = False

    except Exception as e:
      tweet_state.tweet_posted = {
        "error": f"An error accured when posting tweet: {e}"
      }
      message_type = "error"
      # show the popup message
      print("in post tweet exception error part")
      print("tweet final message posted: ", tweet_state.tweet_posted)
      show_popup(popup_state, tweet_state.tweet_posted, message_type)
"""

# Function to clear environment variables
@observe(as_type="observation")
def clear_tokens():
    os.environ["CONSUMER_KEY"] = ""
    os.environ["CONSUMER_SECRET"] = ""
    os.environ["ACCESS_TOKEN"] = ""
    os.environ["ACCESS_TOKEN_SECRET"] = ""
    load_dotenv('.dynamic.env', override=True)
    
    print("Tokens env. vars cleared!")

# function to create dynamic env vars
@observe(as_type="observation")
def create_dynamic_env(env_path: str, env_vars: Dict[str, str]):
    """
    Create or update the dynamic environment file.

    Args:
        env_path (str): The path to the dynamic environment file.
        env_vars (dict): A dictionary containing environment variables.
    """
    # will save env vars env_path=".dynamic.env"
    for key, value in env_vars.items():
      print(f"\n{key}-{value}\n")
      set_key(env_path, key, str(value))
    load_dotenv(dotenv_path='.dynamic.env', override=True)
    print("Dynamic Env vars created and accessible using: os.getenv('variable_name')")


# CLICK EVENT MANAGEMENT
@observe()
def on_click(e: me.ClickEvent):
  s = me.state(PageState)
  # side navigation 
  s.sidenav_open = not s.sidenav_open

# update state form bool values to 'True'
@observe()
def update_form_state(env_dict):
  state = me.state(FormState)
  print("Are all value filled in dict? : ", all(env_dict.values()))
  if all(env_dict.values()):
    print("Env dict updated: ", env_dict)
    for k,v in env_dict.items():
      print("V type: ", type(v))
      if k.lower() == "consumer_key":
        state.consumer_key = True
      if k.lower() == "consumer_secret":
        state.consumer_secret = True
      if k.lower() == "access_token":
        state.access_token = True
      if k.lower() == "access_token_secret":
        state.access_token_secret = True
    if state.consumer_key and state.consumer_secret and state.access_token and state.access_token_secret:
      print("Form State: ", state)
      return True, "Saved!"
  else:
    # warn user that all fields are required
    return False, "All Fields Required!"


@observe()
def on_input(state, key, e: me.InputEvent):
  tweet_auth_env_dict[key] = e.value
 
# Function to handle new messages
@observe()
def handle_message(msg):
  state.messages.append({"text": msg, "is_user": True})


# Function to handle form submission
@observe()
def submit_form(e: me.ClickEvent):
  state = me.state(FormState)
  form_state = me.state(FormState)
  popup_state = me.state(PopupState)
  print("Submit form states: ", state)
   
  # update state form boolean values to 'True'
  state_update_status, state_update_message = update_form_state(tweet_auth_env_dict)
  result["state_update_status"] = state_update_status
  result["state_update_message"] = state_update_message
    
  # check that all values are filled from dictionary and states have been updated to 'True'
  if state_update_status and all(tweet_auth_env_dict.values()):
    print("State Updated Status to 'True' and All Env in Dict!")
      
    # check if field has been modified to inform user that previous value will be replaced
    if state.last_modified:
      print("Last Modified Exist, Rendering Message accordingly!")
      result["state_updated"] = "Previous entry has been replaced by new one."

    # update state with submissiton date and submition status to 'True' 
    state.is_submitted = True
    state.last_modified = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result["last_modified"] = state.last_modified
    print(f"Last Modified Updated to {state.last_modified}")

    # TRANSFER TOKENS TO ENV VAR FILE
    # create env vars file: this will create and save env vars to dynamic.env file
    create_dynamic_env(".dynamic.env", tweet_auth_env_dict) # make sure to delete file or content of file at the end of user session
    # clear env var dictionary, this will get rid of the values secret that were stored in the dict tweet_auth_env_dict
    for k, v in tweet_auth_env_dict.items():
      tweet_auth_env_dict[k] = ""

    print("Env dict emptied down: ", tweet_auth_env_dict)
    print(os.system("cat .dynamic.env"))

    # tell user to press the post to tweeter button
    result["notify_user"] = "Post Tweet Now!"
    show_popup(popup_state, result["notify_user"], "info")

  else:
    # warn user that all fields are required
    result["error"] = "Error: All fields are required."
    print("JUNNNNNNKOOOOO!")
    show_popup(popup_state, "All Fields Are Required!", "error")
  form_state.result = result
  print("State Result: ", form_state.result)


# UPLOADER PAGE
@observe()
@me.page(security_policy=me.SecurityPolicy(allowed_iframe_parents=["https://google.github.io"]),path="/uploader",)
def app():
  pdf_path = path
  file_state = me.state(FileState)
  file_state.pdf_file_path = pdf_path
  me.uploader(
    label="Upload File",
    accepted_file_types=["image/jpeg", "image/png", ".pdf"], # .pdf or application/pdf works
    on_upload=handle_upload,
  )

  if file_state.contents:
    with me.box(style=me.Style(margin=me.Margin.all(10))):
      me.text(f"File name: {file_state.name}")
      me.text(f"File size: {file_state.size}")
      me.text(f"File type: {file_state.mime_type}")
      if file_state.mime_type == "application/pdf":
        me.text(f"File path: {file_state.pdf_file_path}")

    with me.box(style=me.Style(margin=me.Margin.all(10),color="red")):
      if file_state.mime_type == "application/pdf":
        me.markdown(file_state.contents)
      else:
        me.image(src=file_state.contents)

@observe(as_type="observation")
def pdf_to_markdown(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return markdownify.markdownify(text, heading_style="ATX")

@observe(as_type="observation")
def handle_upload(event: me.UploadEvent):
  file_state = me.state(FileState)
  file_state.name = event.file.name
  file_state.size = event.file.size
  file_state.mime_type = event.file.mime_type
  if file_state.mime_type == "application/pdf":
    # save_path = os.path.join(os.getenv("FILE_UPLOAD_FOLDER"), state.name) # using env vars
    save_path = os.path.join("docs", state.name)[:400]
    file_state.pdf_file_path = save_path
    file_state.contents = pdf_to_markdown(save_path)
  else: 
    file_state.contents = f"data:{event.file.mime_type};base64,{base64.b64encode(event.file.getvalue()).decode()}"


### CHAT
@observe()
@me.page(security_policy=me.SecurityPolicy(allowed_iframe_parents=["https://google.github.io"]), path="/chat", title="Mesop Demo Chat",)
def page():
  page_state = me.state(PageState)
  form_state = me.state(FormState)
  tweet_state = me.state(TweetState)
  popup_state = me.state(PopupState)
  agent_state = me.state(AgentState)
  #with me.box(style=me.Style(margin=me.Margin.all(10),display="flex", flex_direction="row", justify_content="center", width="100vw", align_items="center",)):
  with me.box(style=me.Style(display="flex", flex_direction="row", width="100%")):
 
    # sidenav toggle
    with me.sidenav(opened=page_state.sidenav_open, style=me.Style(width=SIDENAV_WIDTH)):
      with me.box(style=me.Style(display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap")):
        me.text("Creditizens AI App", type="headline-5", style=me.Style(color="blue"))
      with me.box(style=me.Style(display="flex", flex_direction="column", justify_content="start", align_content="center", align_items="center")):
        me.text("Download Report", type="headline-6")
        me.button("Internet Search Report")
      with me.box(style=me.Style(display="flex", flex_direction="column", justify_content="start", align_content="center", align_items="center")):
        me.text("Social Media Post", type="headline-6")
        me.button("Post to Tweeter", on_click=lambda e: show_agent_popup(popup_state, e)) # post_tweet function to create and also create a state variable to save last chat answer so that it can be used
      with me.box(style=me.Style(padding=me.Padding.all(10),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap")):
        me.text("Set Key and Tokens", type="headline-6"),
        me.text("All fields are required: (Tip) delete all fields and save again if you haven't provided all fields but saved.", type="body-2", style=me.Style(color="blue")),
        me.text("Enter consumer key"),
        me.input(type="password", value="", on_input=lambda e: on_input(form_state, "CONSUMER_KEY", e), required=True),
        me.text("Enter consumer secret"),
        me.input(type="password", value="", on_input=lambda e: on_input(form_state, "CONSUMER_SECRET", e), required=True),
        me.text("Enter auth token"),
        me.input(type="password", value="", on_input=lambda e: on_input(form_state, "ACCESS_TOKEN", e), required=True),
        me.text("Enter auth secret"),
        me.input(type="password", value="", on_input=lambda e: on_input(form_state, "ACCESS_TOKEN_SECRET", e), required=True),
      
      # Save secrets        
      with me.box(style=me.Style(display="flex", white_space="nowrap", text_overflow="ellipsis", flex_direction="column", justify_content="start", align_content="center", align_items="center")):
          me.button("Save", on_click=submit_form)

      form_state.result = result
      result_state = form_state.result 
      if result_state["state_update_status"] == True:
        with me.box(style=me.Style(padding=me.Padding.all(2),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap")):
          with me.box(style=me.Style(display="flex", text_overflow="ellipsis", position="relative", box_sizing="content-box", padding=me.Padding(top=10, right=8, bottom=10, left=16), height=20, width=150)):
            me.text(f"{result_state['state_update_message']} - {result_state['last_modified']}")
          with me.box(style=me.Style(display="flex", text_overflow="ellipsis", position="relative", box_sizing="content-box", padding=me.Padding(top=2, right=8, bottom=2, left=16), height=20, width=150, color="green")):
            me.text(result_state["notify_user"], type="headline-6")
      else:
        with me.box(style=me.Style(padding=me.Padding.all(2),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap")):
          with me.box(style=me.Style(display="flex", text_overflow="ellipsis", position="relative", box_sizing="content-box", padding=me.Padding(top=2, right=8, bottom=2, left=16), height=20, width=150, color="red",)):
            me.text(result_state["error"])
 
    # sidenav icon onclick
    with me.box(style=me.Style(margin=me.Margin(left=SIDENAV_WIDTH if page_state.sidenav_open else 0),justify_content="center", align_content="center", align_items="center"),):
      with me.content_button(on_click=on_click):
        me.icon("settings")
      me.markdown("Options")

    # Chat side
    #with me.box(style=me.Style(margin=me.Margin.all(10),display="flex", flex_direction="row", justify_content="center", align_content="stretch", width="100vw", flex_grow="1")):
    with me.box(style=me.Style(width="90vw")):
      mel.chat(transform, title="Creditizens Agent Chat Info Tweet", bot_user="Creditizens Helper")

    # Popup message management
    if popup_state.popup_visible:
      with me.box(style=me.Style(position="fixed", top="50%", left="50%", transform="translate(-50%, -50%)", padding=me.Padding.all(20), background="white", box_shadow="0 0 10px rgba(0,0,0,0.5)", z_index=1000)):
        me.text(str(popup_state.popup_message), style=popup_state.popup_style)
        me.button("Close", on_click=lambda e: close_popup(popup_state, e))

    
    # Agent job output popup
    # while os.getenv("POPUP_AGENT_VISIBLE") == "True":
    if popup_state.popup_agent_visible:
      load_dotenv(dotenv_path='.dynamic.env', override=True)
      # start background process that is going to read the log file where agents are going to write their output thoughs and communication.
      # we will capture it here in agent_state agent_messages and update the webui popup periodically
      with me.box(style=me.Style(position="fixed", bottom="10%", right="10%", padding=me.Padding.all(20), background="white", box_shadow="0 0 10px rgba(0,0,0,0.5)", z_index=1000, height="600px", overflow_y="scroll")):
        with me.box(style=me.Style(padding=me.Padding.all(2),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap")):
        
          # Title of popup
          me.text("Agents Tweeter Team Job", type="headline-6")
          me.button("Start Agent Job", on_click=post_tweet)
          mel.text_to_text( output_agent_though, title="Creditizens Tweeter Agent:",)
          """
          # display messages with colors depending on type of log   
          for message in agent_state.agent_messages[:]:
            # display the message with different styling colors
            if message.lower().startswith("action"):
              with me.box(style=me.Style(padding=me.Padding.all(0.5),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap", color="#6a03a6",)):
                me.text(message)
            elif message.lower().startswith("tool"):
              with me.box(style=me.Style(padding=me.Padding.all(0.5),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap", color="#ff6600",)):
                me.text(message)
            elif message.lower().startswith("log"):
              with me.box(style=me.Style(padding=me.Padding.all(0.5),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap", color="#888f88",)):
                me.text(message)
            elif message.lower().startswith("action input"):
              with me.box(style=me.Style(padding=me.Padding.all(0.5),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap", color="#089403",)):
                me.text(message)
            elif message.lower().startswith("observation"):
              with me.box(style=me.Style(padding=me.Padding.all(0.5),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap", color="#0b27db",)):
                me.text(message)
            else:
              with me.box(style=me.Style(padding=me.Padding.all(0.5),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap", color="black",)):
                me.text(message)
            # remove the message from the state
            agent_state.agent_message.remove(message)
            # reload the env var file in order to get its update values to get agent_work_done
            load_dotenv(dotenv_path='.dynamic.env', override=True)
          """
          # show button to close the popup when agent work is done only
          if os.getenv("AGENT_WORK_DONE") == "True":
            me.button("Close", on_click=lambda e: close_agent_popup(popup_state, e))



@observe(as_type="observation")
def transform(input: str, history: list[mel.ChatMessage]):
  tweet_state = me.state(TweetState)
  # add here the llm agents- call function logic for chat
  message = random.sample(LINES, random.randint(3, len(LINES) - 1))
  tweet_state.tweet_chat_message = " ".join(message)
  for line in message:
    time.sleep(0.3)
    
    yield line + " "


LINES = [
  "Shibuya is in Tokyo and is a very famous crossing.",
  "Manga kissa are places where we can read mangas and surf in the internet while enjoying free drinks.",
  "JR line is a line going around Tokyo and very useful as it is stopping at all main stations and helps to easily commute.",
  "Buberry shop in Ginza is where you will find the bests sells people of Japan!.",
]



