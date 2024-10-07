import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage,AIMessage,SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod
# from IPython.display import display, Image
from dotenv import load_dotenv
import os 

load_dotenv()
groq_api_key: str = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.1,
    max_tokens=500,
    api_key=groq_api_key,
)

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

def classification_node(state: State):
    ''' Classify the text into one of the categories: News, Blog, Research, or Other '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.predict_messages([message]).content.strip()
    return {"classification": classification}


def entity_extraction_node(state: State):
    ''' Extract all the entities (Person, Organization, Location) from the text '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.predict_messages([message]).content.strip().split(", ")
    return {"entities": entities}

def summarization_node(state: State):
    ''' Summarize the text in one short sentence '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

# Add edges to the graph
workflow.set_entry_point("classification_node") # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()

# display(
#     Image(
#         app.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
#)
if "messages" not in st.session_state:
        st.session_state.messages = []
st.title("Whisperer")
# for message in st.session_state.messages:
#         if isinstance(message, HumanMessage):
#             role = "user"
#         if isinstance(message, AIMessage):
#             role = "assistant"
        
#         with st.chat_message(role):
#             st.markdown(message.content)

        # if isinstance(message, AIMessage):
        #     st.markdown(message.content)

if prompt := st.chat_input("What is up?"):
        # messages = [HumanMessage(content=prompt)]
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
              st.markdown(prompt)

        result = app.invoke({"text": prompt})
        with st.chat_message("assistant"):
            # print(result)
            st.markdown(result['summary'])
#             m = result['summary'][-1]
#             if isinstance(m, AIMessage):
#                 st.markdown(m.content)
#                 st.session_state.messages.append(m)
# # else:
#     st.warning("Please enter news less than 500 tokens")