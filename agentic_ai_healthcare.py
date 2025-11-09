

# agentic_ai_healthcare.py

from langchain.llms import LlamaCpp  # or OpenAI from langchain_openai
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate

# --- Step 1: Load LLaMA or other LLM model ---
llm = LlamaCpp(
    model_path="models/llama-3-8b-instruct.gguf",  # local LLaMA model
    temperature=0.2,
    max_tokens=512
)

# --- Step 2: Define simple healthcare “tools” ---
def get_patient_summary(patient_id: str) -> str:
    """Simulate retrieving patient summary data from EHR."""
    fake_db = {
        "001": "Patient 001: 45-year-old male with hypertension, mild fever, awaiting lab results.",
        "002": "Patient 002: 63-year-old female, diabetic, postoperative day 2, mild infection signs."
    }
    return fake_db.get(patient_id, "Patient not found.")

def alert_physician(message: str) -> str:
    """Simulate sending an alert to a healthcare provider."""
    return f"[ALERT SENT TO PHYSICIAN] -> {message}"

tools = [
    Tool(name="PatientSummary", func=get_patient_summary, description="Get patient EHR summary."),
    Tool(name="AlertPhysician", func=alert_physician, description="Send an alert to doctor.")
]

# --- Step 3: Initialize Agent with reasoning loop ---
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

# --- Step 4: Test the agent with an example scenario ---
scenario = """
Patient ID: 002
The nurse noted rising temperature and elevated white blood cells.
Should we alert the physician or monitor for another hour?
"""

response = agent.run(scenario)
print("\nAgent Decision:\n", response)
