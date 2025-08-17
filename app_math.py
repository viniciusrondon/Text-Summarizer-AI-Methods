####################### load libraries #######################
import streamlit as st
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


####################### load env variables #######################
load_dotenv()

## openai
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## huggingface
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

## llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

## init tools
wikipedia_wrapper = WikipediaAPIWrapper()

@tool
def wikipedia(query: str) -> str:
    """Useful when you need to answer questions about Math problems and provide explanations."""
    return wikipedia_wrapper.run(query)

wikipedia_tool = wikipedia

llm_math_chain = LLMMathChain.from_llm(llm=llm)

@tool
def calculator(text: str) -> str:
    """Useful when you need to solve math problems and provide explanations. Only input the math expression, do not include any other text."""
    return llm_math_chain.run(text)

math_tool = calculator


prompt = """
You are a helpful assistant that can solve math problems and provide explanations based on Wikipedia Search.

You have access to the following tools:
- wikipedia: to search for information on Wikipedia
- calculator: to solve math problems based on math expressions 

You will be given a math problem. You need to solve the math problem and provide explanations based on Wikipedia Search.

You need to use the tools provided to you to solve the math problem.

You need to use the tools provided to you to provide explanations based on Wikipedia Search.

your answer must be in the same language as the question.

question: {input}
answer:
"""

prompt_template = PromptTemplate(
    template=prompt, 
    input_variables=["input"],
    )

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

@tool
def reasoning(text: str) -> str:
    """A tool for answering logic-based and reasoning-based questions."""
    return llm_chain.run(text)

reasoning_tool = reasoning

tools = [wikipedia_tool, math_tool, reasoning_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)

## function to generate response
def generate_response(input):
    response = agent.run(input)
    return response

####################### streamlit app #######################
st.set_page_config(page_title="Math Problem Solver Agent", page_icon=":calculator:")
st.title("Math Problem Solver Agent")
st.write("This agent can solve math problems and provide explanations based on Wikipedia Search.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm a math problem solver agent. How can I help you today?"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


question = st.chat_input("Enter your question here")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
    response = agent.invoke({"input": question}, callbacks=[st_callback])["output"]
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
