import os
from datetime import datetime
import mesop as me
import mesop.labs as mel
import random
import time
import base64
import pdfplumber
import markdownify
from dotenv import load_dotenv
# Pydantic stype function  argument type specifications
from typing import List, Dict, Union, Optional, Callable

# env vars
load_dotenv(dotenv_path='.env')
# Load dynamic environment variables from dynamic.env file
load_dotenv(dotenv_path='.dynamic.env', override=True)

#### UPLOAD IMAGE AND SET SCHEMA FOR STATE VARIABLE THAT YOU CAN SET, PERSIST AND RENDER

@me.stateclass
class FileState:
  name: str
  size: int
  mime_type: str
  contents: str
  pdf_file_path: str

result = {
  "state_update_status": False,
  "state_updated": "",
  "state_update_message": "",
  "last_modified": "",
  "notify_user": "",
  "error": ""
}

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

# Env Vars to set
path = "./docs/World_Largest_Floods_paper.pdf"
SIDENAV_WIDTH = 350

# function to create dynamic env vars
def create_dynamic_env(env_path: str, env_vars: Dict[str, str]):
    """
    Create or update the dynamic environment file.

    Args:
        env_path (str): The path to the dynamic environment file.
        env_vars (dict): A dictionary containing environment variables.
    """
    with open(env_path, 'w') as env_file:
        for key, value in env_vars.items():
            env_file.write(f"{key}={value}\n")
    load_dotenv(dotenv_path='.dynamic.env', override=True)
    print("Dynamic Env vars created and accessible using: os.getenv('variable_name')")
 
# Function to clear environment variables
def clear_tokens():
    os.environ['CONSUMER_KEY'] = ""
    os.environ['CONSUMER_SECRET'] = ""
    os.environ['ACCESS_TOKEN'] = ""
    os.environ['ACCESS_TOKEN_SECRET'] = ""
    print("Tokens env. vars cleared!")

# CLICK EVENT MANAGEMENT
def on_click(e: me.ClickEvent):
  s = me.state(PageState)
  # side navigation 
  s.sidenav_open = not s.sidenav_open

# INPUT EVENT MANAGEMENT
env_dict = {
    "CONSUMER_KEY": "",
    "CONSUMER_SECRET": "",
    "ACCESS_TOKEN": "",
    "ACCESS_TOKEN_SECRET": ""
}

# update state form bool values to 'True'
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

# can use those functions instead of 'lambda' anduse those in input field 'on_input'
#def on_input_consumer_key(e):
#    env_dict["CONSUMER_KEY"] = e.value
#def on_input_consumer_secret(e):
#    env_dict["CONSUMER_SECRET"] = e.value
#def on_input_access_token(e):
#    env_dict["ACCESS_TOKEN"] = e.value
#def on_input_access_token_secret(e):
#    env_dict["ACCESS_TOKEN_SECRET"] = e.value

def on_input(state, key, e: me.InputEvent):
  env_dict[key] = e.value
 
# Function to handle new messages
def handle_message(msg):
  state.messages.append({"text": msg, "is_user": True})


# Function to handle form submission
def submit_form(e: me.ClickEvent):
  state = me.state(FormState)
  form_state = me.state(FormState)
  print("Submit form states: ", state)
   
  # update state form boolean values to 'True'
  state_update_status, state_update_message = update_form_state(env_dict)
  result["state_update_status"] = state_update_status
  result["state_update_message"] = state_update_message
    
  # check that all values are filled from dictionary and states have been updated to 'True'
  if state_update_status and all(env_dict.values()):
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
    # create env vars file
    create_dynamic_env(".dynamic.env", env_dict) # make sure to delete file or content of file at the end of user session
    # clear env var dictionary
    for k, v in env_dict.items():
      env_dict[k] = ""

    print("Env dict emptied down: ", env_dict)
    print(os.system("cat .dynamic.env"))

    # tell user to press the post to tweeter button
    result["notify_user"] = "Post Tweet Now!"

  else:
    # warn user that all fields are required
    result["error"] = "Error: All fields are required."

  form_state.result = result
  print("State Result: ", form_state.result)




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

def pdf_to_markdown(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return markdownify.markdownify(text, heading_style="ATX")

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

@me.page(security_policy=me.SecurityPolicy(allowed_iframe_parents=["https://google.github.io"]), path="/chat", title="Mesop Demo Chat",)
def page():
  page_state = me.state(PageState)
  form_state = me.state(FormState)
  #with me.box(style=me.Style(margin=me.Margin.all(10),display="flex", flex_direction="row", justify_content="center", width="100vw", align_items="center",)):
  with me.box(style=me.Style(display="flex", flex_direction="row", width="100%")):
 
    # sidenav toggle
    with me.sidenav(opened=page_state.sidenav_open, style=me.Style(width=SIDENAV_WIDTH)):
      me.text("Creditizens AI App")
      with me.box(style=me.Style(display="flex", flex_direction="column", justify_content="start", align_content="center", align_items="center")):
        me.text("Download Report", type="headline-6")
        me.button("Internet Search Report", on_click=post_tweet) # post_tweet function to create and also create a state variable to save last chat answer so that it can be used
      with me.box(style=me.Style(display="flex", flex_direction="column", justify_content="start", align_content="center", align_items="center")):
        me.text("Social Media Post", type="headline-6")
        me.button("Post to Tweeter")
      with me.box(style=me.Style(padding=me.Padding.all(10),display="flex", flex_direction="column", background="white", justify_content="start", align_content="center", align_items="center", flex_wrap="wrap")):
        me.text("Set Key and Tokens", type="headline-6"),
        me.text("All fields are required", type="body-2", style=me.Style(color="orange")),
        me.text("Enter consumer key"),
        me.input(type="password", value="", on_input=lambda e: on_input(form_state, "CONSUMER_KEY", e), required=True),
        me.text("Enter consumer secret"),
        me.input(type="password", value="", on_input=lambda e: on_input(form_state, "CONSUMER_SECRET", e), required=True),
        me.text("Enter auth token"),
        me.input(type="password", value="", on_input=lambda e: on_input(form_state, "ACCESS_TOKEN", e), required=True),
        me.text("Enter auth secret"),
        me.input(type="password", value="", on_input=lambda e: on_input(form_state, "ACCESS_TOKEN_SECRET", e), required=True),
        print("Env dict: ", env_dict)
      
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
      mel.chat(transform, title="Mesop Demo Chat", bot_user="Creditizens Helper")



def transform(input: str, history: list[mel.ChatMessage]):
  for line in random.sample(LINES, random.randint(3, len(LINES) - 1)):
    time.sleep(0.3)

    yield line + " "


LINES = [
  "Mesop is a Python-based UI framework designed to simplify web UI development for engineers without frontend experience.",
  "It leverages the power of the Angular web framework and Angular Material components, allowing rapid construction of web demos and internal tools.",
  "With Mesop, developers can enjoy a fast build-edit-refresh loop thanks to its hot reload feature, making UI tweaks and component integration seamless.",
  "Deployment is straightforward, utilizing standard HTTP technologies.",
  "Mesop's component library aims for comprehensive Angular Material component coverage, enhancing UI flexibility and composability.",
  "It supports custom components for specific use cases, ensuring developers can extend its capabilities to fit their unique requirements.",
  "Mesop's roadmap includes expanding its component library and simplifying the onboarding processs.",
]



