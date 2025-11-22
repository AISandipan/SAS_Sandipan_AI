from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

from tools import (
    get_calendar_events,
    create_calendar_event,
    gmail_get_emails,
    gmail_send_email,
    todoist_get_tasks,
    todoist_create_task,
    create_flux_image,
    create_note,
    vector_search,
    vector_add
)
from memory import memory

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.2
)

tools = [
    Tool(name="get_calendar_events", func=get_calendar_events, description="Get Google Calendar events"),
    Tool(name="create_calendar_event", func=create_calendar_event, description="Create a calendar event"),
    Tool(name="gmail_get_emails", func=gmail_get_emails, description="Read Gmail emails"),
    Tool(name="gmail_send_email", func=gmail_send_email, description="Send Gmail email"),
    Tool(name="todoist_get_tasks", func=todoist_get_tasks, description="Get Todoist tasks"),
    Tool(name="todoist_create_task", func=todoist_create_task, description="Create Todoist task"),
    Tool(name="create_flux_image", func=create_flux_image, description="Generate image using FLUX"),
    Tool(name="create_note", func=create_note, description="Create a note"),
    Tool(name="vector_search", func=vector_search, description="Search memory in Pinecone"),
    Tool(name="vector_add", func=vector_add, description="Add memory to Pinecone")
]

agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
