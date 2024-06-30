import os
import json
from dotenv import load_dotenv, set_key
# Pydantic stype function  argument type specifications
from typing import List, Dict, Union, Optional, Callable
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
#### FOR AGENT TEAM
# LANFCHAIN TOOLS
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
# CREWAI AGENTS
from crewai import Agent
from crewai import Task
from crewai import Crew, Process
# from crewai_tools import FileReadTool, DirectoryReadTool
# for cqpture stdout
#from io import StringIO
import logging


# env vars
load_dotenv(dotenv_path='.env', override=False)
# Load dynamic environment variables from dynamic.env file
load_dotenv(dotenv_path='.dynamic.env', override=True)

######################## AGENTS ################################
### AGENTS INPUTS
input_tweet_agents={
  "topic": os.getenv("TWEET_TOPIC"),
  "chat_response_message": os.getenv("CHAT_RESPONSE_MESSAGE"),
  "TWEET_FILE": os.getenv("TWEET_FILE"),
}

input_report_agents={
  "topic": os.getenv("REPORT_TOPIC")
}

env_file = ".dynamic.env"

chat_response_message = ""
topic = ""
TWEET_FILE = ""

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

### AGENTS HELPTER FUNCTIONS

# Configure logging
log_file_path = os.getenv("LOG_FILE")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(message)s')

# callback functions
def agent_step_callback(Agent): #step, state
    #message = f"Agent {Agent.role} at step {Step}: {State}"
    message = str(Agent)
    logging.info(message)
def task_callback(Task): # Result
    #message = f"Task {Task.description} completed with result: {Result}"
    message = str(Task)
    logging.info(message)
def capture_output_callback(Agent, Message):
    logging.info(f"Agent {Agent.role} output: {Message}")

# llm chat call function
@observe(as_type="generation")
def create_tweet_from_last_message(message, topic) -> str:
  """
    This function will call an llm to create the best tweet ever. It will also save the tweet in a file.
    
    Parameters:
    message str: ''
    topic str: 'description of how is the desired tweet by user'
    
    Returns:
    Str representing the final tweet that will posted on Tweeter.
  """
   
  try:
    ### DIFFERENT TYPES OF LLM CALLS
    # call llm to create tweet
    # tweet_from_llm = lmstudio_llm.chat.completions.create OR USE LMSTUDIO
        # model="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q3_K_M.gguf",
    # tweet_from_llm = openai_llm.chat.completions.create(  OR USE OPENAI
        # model="gpt-3.5-turbo-1106", # OPENAI
    tweet_from_llm = groq_client.chat.completions.create( # OR USE GROQ
      model=os.getenv("MODEL_MIXTRAL_7B"), # OR os.getenv("MODEL_LLAMA3_70B") FOR GROQ

      messages=[
        {
          "role": "system",
          "content": f"""
            You are an expert in creating emotional, interesting and engaging social media posts. You always wrap your answer that can be posted by the user in markdown '```python' '```' tag. You always keep your answer short and concise to not exceed character count for social media post.
          """
        },
        {
          "role": "user",
          "content": f"Please can you {topic} from this message: {message}",
        }
      ],
      temperature=float(os.getenv("GROQ_TEMPERATURE")), # 0.1 # GROQ and OPENAI and LMSTUDIO
      max_tokens=int(os.getenv("GROQ_MAX_TOKEN")), # 1024  # GROQ and OPENAI
      top_p=1, # GROQ and OPENAI
      stop=None, # GROQ and OPENAI
      stream=False, # GROQ and OPENAI AND LMSTUDIO
    )
    print("Tweet Form LLM: ", tweet_from_llm.choices[0].message.content)
    tweet_from_llm_final_answer = tweet_from_llm.choices[0].message.content.split("```")[1].strip("python").strip()
    langfuse_context.update_current_observation(
      input=f"'topic': {topic}, 'message': {message}",
      model=os.getenv("MODEL_MIXTRAL_7B"),
      metadata={"function": "create_tweet_from_last_message", "purpose": "Here LLM will create an SEO optimized tweet."}
    )
    
    # save tweet to file if needed for report generation which will check if this file exist and use it to make internet search and fact check
    with open(os.getenv("TWEET_FILE"), "w", encoding="utf-8") as tweet_file:
      tweet_file.write(tweet_from_llm_final_answer)
      print("Tweet written to file asn saved!")
    
    # update state with final tweet to be posted
    set_key(".dynamic.env", "TWEET_MESSAGE", tweet_from_llm_final_answer)

    print(f"Tweet created and saved to the state shared env var, TWEET_MESSAGE: {tweet_from_llm_final_answer}")
    return f"Success: The final tweet to be posted has been created, please keep only the following in your final output: {tweet_from_llm_final_answer}"

  except Exception as e:
    return f"Error while getting tweet from llm generation: {e}"  

########### AGENTS TOOLS

# internet search tool
internet_search_tool = DuckDuckGoSearchRun()

# create tweet tool
class TweetVars(BaseModel):
      # tweet_state parameter not needed
      message: str = Field(default=chat_response_message, description="This is the latest message answer from the chat that will be used to create our tweet post.")
      topic: str = Field(default=topic, description="This is user desired outcome and how the tweet should be formatted")

@observe()
def create_tweet_tool(message: str = chat_response_message, topic: str = topic) -> str:
  """
    This tool will get information about the product added to the {os.environ.get('store')} store. 
    
    Parameter: 
    message str: 'This is the latest message answer from the chat that will be used to create our tweet post.' = {chat_response_message}
    topic str : 'This is user desired outcome and how the tweet should be formatted' = {topic}
    
    Returns: 
    Str with information about the latest product details of the latest product created.
  """
  try:
    tweet = create_tweet_from_last_message(message, topic)
    return tweet
  except Exception as e:
    return '{"error": f"Error while creating tweet from llms create_tweet_tool: {e}"}'

create_tweet_tool = StructuredTool.from_function(
  func=create_tweet_tool,
  name="create tweet tool",
  description=  """
    This tool will create the best tweet ever.
  """,
  args_schema=TweetVars,
  # return_direct=True, # returns tool output only if no TollException raised
  # coroutine= ... <- you can specify an async method if desired as well
  # callback==callback_function # will run after task is completed
)

# check tweet tool
class CheckLenghtFile(BaseModel):
      tweet_file: str = Field(default=TWEET_FILE, description="The file where the scheduled to be posted tweet exists.")

@observe()
def tweet_check_tool(tweet_file: str = TWEET_FILE) -> int:
  """
    This tool will check the length of the tweet generated and return its length. If the length is more than 210 characters, a new tweet need to be created to respect that requirement. 
    
    Parameter: 
    tweet_file str : 'The file where the scheduled to be posted tweet exists.' = {TWEET_FILE}
    
    Returns: 
    int The length of the tweet 
  """
  
  with open(tweet_file, "r", encoding="utf-8") as scheduled_tweet_post:
    scheduled_post = scheduled_tweet_post.read()
    return len(scheduled_post)

tweet_check_tool = StructuredTool.from_function(
  func=tweet_check_tool,
  name="tweet check tool",
  description=  """
    This tool will check the length of the scheduled tweet post.
  """,
  args_schema=CheckLenghtFile,
  # return_direct=True, # returns tool output only if no TollException raised
  # coroutine= ... <- you can specify an async method if desired as well
  # callback==callback_function # will run after task is completed
)
    
### AGENTS DEFINITION
# TWEET AGENTS
tweet_creator = Agent(
  role="Create tweets",
  goal=f"Create an amazing SEO optimized tweet using ONLY available tools. topic is {topic}.",
  verbose=True,
  memory=False,
  backstory="""You are an expert known to be using tools to create the best tweets of the web.""",
  tools=[create_tweet_tool],
  allow_delegation=True,
  llm=groq_llm_mixtral_7b,
  max_rpm=3,
  max_iter=4,
  # callback=capture_output_callback,
  step_callback=agent_step_callback,
)

tweet_checker = Agent(
  role="Check tweets",
  goal=f"Check that the number of characters of the tweet is not exceeding 210 characters otherwise create one similar that is less than 210 characters long. topic is '{topic}' and message to transform in tweet is '{chat_response_message}'.",
  verbose=True,
  memory=False,
  backstory="""You are an expert known to be using tools to check that tweets don't exceed a certain number of characters and create the best tweets of the web if you notice that the number of charatcter is exceeded. You always make tweets under 210 characters.""",
  tools=[tweet_check_tool],
  allow_delegation=True,
  llm=groq_llm_llama3_70b,
  max_rpm=3,
  max_iter=4,
  # callback=capture_output_callback,
  step_callback=agent_step_callback,
)

# REPORT AGENTS
fact_checker = Agent(
  role="fact checker",
  goal=f"Fact check online about: '{chat_response_message}'.",
  verbose=True,
  memory=False,
  backstory="""You are an expert known to be using tools to create the best tweets of the web.""",
  tools=[internet_search_tool],
  allow_delegation=True,
  llm=groq_llm_llama3_70b,
  max_rpm=3,
  max_iter=4,
  # callback=capture_output_callback,
  step_callback=agent_step_callback,
)

report_outline_creator = Agent(
  role="report outline creator",
  goal=f"Get your collegue fact checker view on the topic: {topic}. Then create mardown formatted very detailed report outline on the topic '{topic}' so that your collegue will be able to create a quality report tackling different aspects of the topic.",
  verbose=True,
  memory=False,
  backstory="""You are an expert in generating markdowm formatted report outlines on different topics after having asked for a fact check online from your collegues.""",
  tools=[internet_search_tool],
  allow_delegation=True,
  llm=groq_llm_gemma_7b,
  max_rpm=3,
  max_iter=4,
  # callback=capture_output_callback,
  step_callback=agent_step_callback,
)

report_creator = Agent(
  role="Create detailed reports",
  goal=f"Get report outline from collegue and create detailed mardown formatted report investigating and answering to potential questions about this topic: {topic}.",
  verbose=True,
  memory=False,
  backstory="""You are an expert in generating very detailed markdown formatted reports from outline. Your writing style is emotional and persuasive. People reading your reports always get a very pertinent view and answer from their initial topics that they wouldn't have thought of before.""",
  allow_delegation=True,
  llm=groq_llm_mixtral_7b,
  max_rpm=3,
  max_iter=4,
  # callback=capture_output_callback,
  step_callback=agent_step_callback,
)

### AGENTS TASKS DEFINITION
# TWEET TASK
tweet_creation_task = Task(
  description=f"""First, execute tool available which will create the tweet. Do NOT try to be clever by creating variables or trying to figure out something, just execute the tool. Then, wait for the tool output message from the tool.""",
  expected_output=f"Put only the tweet generated by the tool file using the mardown format. Your output should be wrapper in a markdown '```python' '```' .",
  tools=[create_tweet_tool],
  agent=tweet_creator,
  async_execution=False,
  output_file="./tweets/agent_tweet_creation_report.md",
  callback=task_callback,
)

tweet_check_task = Task(
  description=f"""First, execute tool available. Do NOT try to be clever by creating variables or trying to figure out something, just execute the tool. Then, wait for the tool output message from the tool. If the tool message return a number under 210 then consider that your job is done. if the tool message returns a number higher than 210 then you need to create a new tweet about the topic: '{topic}'. Make sure that if you are forced to create a new tweet, that one should only 210 characters long or under.""",
  expected_output=f"if the tool returned number what equal or under 210 then you can consider your job done, just output the tweet generated by your collegue as last output. If the tool returned a number above 210, then you will need to create a new tweet under 210 characters and save it in a text format. Your output should be wrapper in a markdown '```python' '```' .",
  tools=[tweet_check_tool],
  agent=tweet_checker,
  async_execution=False,
  output_file="./tweets/agent_tweet_length_adjusted.txt",
  callback=task_callback,
)

# REPORT TASKS
fact_check_task = Task(
  description=f"""First, execute tool available to fact check the information provided by this topic: {topic}. Do NOT try to be clever by creating variables or trying to figure out something, just execute the tool. Find also links online for the sources of your fact checking to have solid ground for user to be able to read more about those. Inform your collegue report outline creator about your finding so that he can adjust his work accordingly.""",
  expected_output=f"Use markdown format to create a report about your findings during the inter search fact checking task.",
  tools=[internet_search_tool],
  agent=tweet_checker,
  async_execution=False,
  output_file="./reports/agent_fact_check_report.md",
  callback=task_callback,
)

report_outline_creation_task = Task(
  description=f"""Get information from your colleague fact checker and create a very detailed outline to create a report on the topic: {topic}. This task main goals is to create a markdown formatted report outline and the topic: 'topic'. If you believe that you need internet search for extra information, use available tool to have an enhanced quality report outline.""",
  expected_output=f"A markdown format very detailed report outline.",
  tools=[internet_search_tool],
  agent=tweet_checker,
  async_execution=False,
  output_file="./reports/agent_outline_of_report.md",
  callback=task_callback,
)

report_creation_task = Task(
  description=f"""Get information from your colleague fact checker and create a very detailed outline to create a report on the topic: {topic}. This task main goals is to create a markdown formatted report outline and the topic: 'topic'. You can search the internet using the tool provided if you believe that your report need extra information. When the report is created the job is done.""",
  expected_output=f"A markdown format very detailed report outline..",
  tools=[internet_search_tool],
  agent=tweet_checker,
  async_execution=False,
  output_file="./reports/agent_outline_of_report.md",
  callback=task_callback,
)

##### AGENT TEAMS
## PROCESS SEQUENTIAL
tweet_agent_team = Crew(
  agents=[tweet_creator, tweet_checker],
  tasks=[tweet_creation_task, tweet_check_task],
  process=Process.sequential,
  verbose=2,
    callback=capture_output_callback,
)
## PROCESS SEQUENTIAL
report_agent_team = Crew(
  agents=[fact_checker, report_outline_creator, report_creator],
  tasks=[fact_check_task, report_outline_creation_task, report_creation_task],
  process=Process.sequential,
  verbose=2,
  callback=capture_output_callback,
)

@observe()
def agent_team_job():
  load_dotenv('.dynamic.env', override=True)
  print("Inside agent team job")
  if os.getenv("TWEET_AGENTS") == "True":
    print("Inside agent team job Tweet Agents")
    tweet_workers = tweet_agent_team.kickoff(inputs=input_tweet_agents)
    return tweet_workers
  if os.getenv("REPORT_AGENTS") == "True":
    report_workers = report_agent_team.kickoff(inputs=input_report_agents)
    return report_workers

# change this to update the env var and start the agent team with the right inputs
if __name__ == '__main__':
  load_dotenv('.dynamic.env', override=True)
  print("Inside agents.py will start now agent_team_job()")
  agent_team_job()
  
  # Indicate agent work is done for the capture output to detect it and stop its while loop then mesop will continue the rest of the functions app
  set_key('.dynamic.env', "AGENT_WORK_DONE", "True")
  load_dotenv('.dynamic.env', override=True)
  # this is handled in the mesop app part
  # set_key(env_file, "AGENT_WORK_DONE", "False")
  # set_key(env_file, "POPUP_AGENT_VISIBLE", "True")
  







