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
import threading


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


############################################################
##### BUSINESS LOGIC HELPER FUNCTIONS
@observe(as_type="observation")
def update_agent_popup():
    if os.getenv("AGENT_WORK_DONE") == "True":
      popup_state.agent_work_done = True
    else:
      popup_state.agent_work_done = False
      while True:
        time.sleep(2)  # Check every 2 seconds
        load_dotenv('.dynamic.env', override=True)
        popup_state = me.state(PopupState)
        agent_state = me.state(AgentState)

        if os.getenv("POPUP_AGENT_VISIBLE") == "True":
          popup_state.popup_agent_visible = True
          agent_messages = json.loads(os.getenv("AGENT_MESSAGES", "[]"))
          agent_state.agent_messages = agent_messages
          for message in agent_state.agent_messages:
              me.text(message)
        else:
          popup_state.popup_agent_visible = False


# Manage agents state as they are working in order to show their though in a popup
@observe()
def show_agent_popup(state):
    state.popup_agent_visible = True
    # set the popup_agent_visible shared state env varsto true
    set_key(".dynamic.env", "POPUP_AGENT_VISIBLE", "True")
    load_dotenv('.dynamic.env', override=True)
@observe()
def close_agent_popup(state, e: me.ClickEvent):
    state.popup_agent_visible = False
    # reset the agent_work_done shared state env vars
    set_key(".dynamic.env", "AGENT_WORK_DONE", "False")
    set_key(".dynamic.env", "POPUP_AGENT_VISIBLE", "False")
    load_dotenv('.dynamic.env', override=True)
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

#def update_popup(): # see if this function can be used instead with env. vars in the agents.py 'try' of 'capture_output'
#  agent_state = me.state(AgentState)
#  popup_state = me.state(PopupState)
#  agent_output = ""
#  while os.getenv("AGENT_WORK_DONE") == "False":
#    load_dotenv(env_file, override=True)
#    new_output = os.getenv("AGENT_MESSAGES")
#    if new_output != agent_output:
#      agent_output = new_output
#      agent_state.agent_messages = agent_output
#      time.sleep(2)
#    agent_done = os.getenv("AGENT_WORK_DONE")
#  popup_state.agent_work_done = True

class StreamCapturer(StringIO):
    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, s):
        super().write(s)
        self.original_stream.write(s)
        self.original_stream.flush()

    def flush(self):
        super().flush()
        self.original_stream.flush()

def capture_output(command):
    capturer = StreamCapturer(sys.stdout)
    sys.stdout = capturer

    def read_output():
        return capturer.getvalue()

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def update_output():
        while process.poll() is None:
            time.sleep(1)
            output = read_output()
            if output:
                append_to_agent_messages('.dynamic.env', [output])
                capturer.seek(0)
                capturer.truncate(0)
            load_dotenv('.dynamic.env', override=True)
            if os.getenv("AGENT_WORK_DONE") == "True":
                break

        final_output = read_output()
        if final_output:
            append_to_agent_messages('.dynamic.env', [final_output])
        sys.stdout = capturer.original_stream

    thread = threading.Thread(target=update_output)
    thread.start()
    thread.join()

def start_agents():
  popup_state = me.state(PopupState)
  agent_state = me.state(AgentState)
  # initialize agent_work_done to False
  set_key(".dynamic.env", "AGENT_WORK_DONE", "False")
  set_key('.dynamic.env', "POPUP_AGENT_VISIBLE", "True")
  load_dotenv('.dynamic.env', override=True)
  popup_state.popup_agent_visible = True
  agent_state.agent_work_done = False
  # Activate the virtual environment and run the agents script
  command = ["bash", "-c", "source agents_venv/bin/activate && python3 agents.py"]
  capture_output(command)
  set_key('.dynamic.env', "AGENT_WORK_DONE", "True")
  load_dotenv('.dynamic.env', override=True)
  agent_state.agent_work_done = True
  if os.getenv("AGENT_WORK_DONE") == "True":
    agent_process.terminate()

# post tweet
# Function to post a tweet using saved tokens
# add here llm agent call to work on the tweet
@observe()
def post_tweet(e: me.ClickEvent):
    # get the needed states
    tweet_state = me.state(TweetState)
    popup_state = me.state(PopupState)
    agent_state = me.state(AgentState)
    from_state = me.state(FormState)
    ## Set env vars for agent input
    print("Tweet State Raw: ", tweet_state, type(tweet_state))
    tweet_state_json_str = json.dumps(tweet_state.__dict__)
    print("Tweet State Json: ", tweet_state_json_str, type(tweet_state_json_str))
    agent_input_vars_dynamic = {
      "CHAT_RESPONSE_MESSAGE": tweet_state.tweet_chat_message,
      "TWEET_STATE": tweet_state_json_str
    }
    create_dynamic_env(".dynamic.env", agent_input_vars_dynamic)
    
    # flag here for the agents that it is the tweet agents that are going to work and not the report ones. we need to use shared env vars.
    set_key(env_file, "TWEET_AGENTS", "True")
    load_dotenv('.dynamic.env', override=True)
    print(f"Check if TWEET_AGENTS env var updated to 'True': {os.getenv('TWEET_AGENTS')}")
    
    # start tweet agent workers. They will update the state tweet_message which is the final tweet posted catched later on, on this function
    if from_state.consumer_key and from_state.consumer_secret and from_state.access_token and from_state.access_token_secret:
      if tweet_state.tweet_chat_message:
        try:
          print("Inside post_tweet agent job will start")
          show_agent_popup(agent_state)
          # start agent work
          set_key(".dynamic.env", "AGENT_WORK_DONE", "False")
          start_agents()
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
        me.button("Post to Tweeter", on_click=post_tweet) # post_tweet function to create and also create a state variable to save last chat answer so that it can be used
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
      with me.box(style=me.Style(position="fixed", bottom="10%", right="10%", padding=me.Padding.all(20), background="white", box_shadow="0 0 10px rgba(0,0,0,0.5)", z_index=1000, height="900px", overflow_y="scroll")):
        me.text("Agents Output", type="headline-6")
        #for message in json.loads(os.getenv("AGENT_MESSAGES")):
        #for message in agent_state.agent_messages:
          #me.text(message)
        
        # Start the periodic update for agent popup
        update_agent_popup()
        
        #if os.getenv("AGENT_MESSAGES"):
          #load_dotenv(dotenv_path='.dynamic.env', override=True)
          #for message in json.loads(os.getenv("AGENT_MESSAGES")):
            #me.text(message)
        #if os.getenv("AGENT_WORK_DONE") == "True":
        #if popup_state.agent_work_done:  # Show button only when done
          #me.button("Close", on_click=lambda e: close_agent_popup(popup_state, e))


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



