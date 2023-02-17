import langchain
from langchain.agents import load_tools
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, LLMChain, VectorDBQAWithSourcesChain, PromptTemplate
import faiss
import pickle
import openai
import ast

def vectordb_qa_tool(query: str) -> str:
    print("here")
    langchain.verbose=True
    """Tool to answer a question."""
    index = faiss.read_index("docs.index")
    with open("faiss_store.pkl", "rb") as f:
      store = pickle.load(f)
    store.index = index
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store, k=4)
    result = chain({"question": query})
    return result['answer'], result['sources']

def agent(question):
  print(f'{question=}')
  tools = [
      Tool(
          name = "vector_db_qa",
          func=vectordb_qa_tool,
          description="Access contextual information such as emails, documents, databases. This should be the first place to look for information."
      )
  ]

  # load tools returns a list of tools but I don't want to add a list to a list, so I just add the first element
  tools.append(load_tools(["requests"])[0])

  # Tool(
  #   name="Requests",
  #   func=requests.run,
  #   description="Use the Requests tool to make HTTP requests which will return the text contents of the requested webpage."
  # )

  # First you must decide if you need iteratively solve the problem.

  # Example: Give me the LinkedIn urls of Marina Nester and Charlotte Gall
  # Thought: I need to solve this iteratively. Let me first choose which tool to use to find Marina's LinkedIn, then come back to Charlotte later.

  prefix = """Answer the following questions as best you can. You have access to the following tools:"""
  suffix = """
EXAMPLES:
Question: What is David Patterson-Cole's email? Check the db.
Thought: I need to use the vector_db_qa tool to find the email.
Action: vector_db_qa
Action Input: David Patterson-Cole email 

Question: Summarize this page: https://en.wikipedia.org/wiki/2023_Turkey%E2%80%93Syria_earthquake
Thought: I need to get the page data and then summarize it.
Action: Requests
Action Input: https://en.wikipedia.org/wiki/2023_Turkey%E2%80%93Syria_earthquake
    
  
Begin!"

Question: {input}
{agent_scratchpad}"""

  prompt = ZeroShotAgent.create_prompt(
      tools, 
      prefix=prefix, 
      suffix=suffix, 
      input_variables=["input", "agent_scratchpad"]
  )

  print(prompt.template)

  # llm_chain = LLMChain(llm=OpenAI(temperature=0, model_name="text-curie-001"), prompt=prompt)
  llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

  tool_names = [tool.name for tool in tools]
  tool_names = ["vector_db_qa", "Requests"]
  agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, return_intermediate_steps=True)

  agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

  # agent_executor.run("What are the names of the people I met with in February. Use the calendly emails to find who I met with.")

  # agent_executor.run("What is Marina Nester's location? Check her LinkedIn profile which is in the db.")

  # agent_executor.run("Find the locations of Marina Nester and Charlotte Gall using their LinkedIn profiles in the db")
  return agent_executor.run(question)

def main():
  # use a separate LLM to check if "iterative" and return the array to iterate over
  prompt_template_1 = """
You must extract the entities from the input question and return two things: 1) the entities in a list 2) the "Question Structure", which is the original question with a placeholder for the entities.

Question: What is the population of New York City and Paris? 
Answer:["New York City", "Paris"]--|--What is the population of (entity)?

Question: Where does David Patterson-Cole, Ganesh Thirumurthi, Josh Bitonte, and Julia Di Spirito live?
Answer:["David Patterson-Cole", "Ganesh Thirumurthi", "Josh Bitonte", "Julia Di Spirito"]--|--Where does (entity) live?

Begin!

Question: Find the emails of Marina Nester and Charlotte Gall.
Answer: 
  """
    # Question: Find the locations of Marina Nester and Charlotte Gall. First find their LinkedIn profiles in the db and then second look up the LinkedIn profiles to find the location.
    # Question: Find the emails of Marina Nester and Charlotte Gall.
  # Find the locations of Marina Nester and Charlotte Gall using their LinkedIn profiles in the db

  response = openai.Completion.create(
    model="text-davinci-003",
    # model="text-curie-001",
    prompt=prompt_template_1,
    max_tokens=500,
    temperature=0
  )['choices'][0]['text']
  print(response)
  entities, question = response.split("--|--")
  # print (f'{entities=}')
  list_entities = ast.literal_eval(entities.strip())
  print (f'{list_entities=}')
  print (f'{question=}')
  # iterate through array calling agent with the current element
  final_answer = []
  for element in list_entities:
    formatted_question = question.replace("(entity)", element)
    final_answer.append(agent(formatted_question))
  # return the aggregate result
  print (f'{final_answer=}')

main()



















# prompt_template = """
# Answer the following questions as best you can.

# First, you must ALWAYS decide if you need iteratively loop over multiple entities or if there is just a single entity.

# Question: Give me the LinkedIn urls of Marina Nester and Charlotte Gall
# Thought: I have two names and need to iteratively solve this. Let me first choose which tool to use to find Marina's LinkedIn, then come back to Charlotte's LinkedIn later.

# You have access to the following tools:

# vector_db_qa: Access contextual information such as emails, documents, databases. This should be the first place to look for information.

# Use the following format:

# Question: the input question you must answer
# Thought: iterative or not?
# Thought: you should always think about what to do
# Action: the action to take, should be one of [vector_db_qa]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!"

# Question: {input}
# {agent_scratchpad}
# """

# prompt = PromptTemplate(template=prompt_template, input_variables=["input", "agent_scratchpad"])