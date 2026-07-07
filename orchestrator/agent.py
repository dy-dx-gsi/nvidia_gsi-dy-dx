import os
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class OrchestratorIntent(BaseModel):
    action: str = Field(description="The primary action to perform, e.g., 'generate_and_train'")
    defect_type: str = Field(description="The specific type of defect requested, e.g., 'thermal_warping', 'micro_fracture'")
    part_type: str = Field(description="The physical part to be inspected, e.g., 'gear', 'engine_block'")
    num_images: int = Field(description="Number of synthetic images to generate for training")
    deploy: bool = Field(description="Whether to deploy to ROS 2 physical hardware after training")
    robot_joint_angles: List[float] = Field(description="A list of exactly 6 target joint angles (in radians between -3.14 and 3.14) for the 6-axis robotic arm, intelligently calculated and optimized for inspecting the physical shape of the requested part_type.")
    
    # AI-Driven Fallback Generation
    fallback_primitive: str = Field(description="The closest basic geometric primitive shape for the part if no CAD model is found. Must be one of: 'cube', 'sphere', 'cylinder', 'cone', 'torus'")
    fallback_scale: List[float] = Field(description="An intelligent guess of the 3D dimensions [X, Y, Z] scale of the part to make the primitive look like the object. e.g. a turbine blade is a long flat cube so scale=[10.0, 50.0, 2.0]. A gear is scale=[20.0, 20.0, 20.0].")
    fallback_diffuse: List[float] = Field(description="An intelligent guess of the RGB diffuse color of the part AND its defect (list of 3 floats between 0.0 and 1.0). e.g., rust=[0.7, 0.2, 0.1], scorch/burn mark=[0.1, 0.1, 0.1], pristine steel=[0.5, 0.5, 0.5]. CRITICAL: The color MUST reflect the defect_type requested!")
    fallback_metallic: float = Field(description="An intelligent guess of how metallic the part is (0.0 to 1.0). e.g. pristine metal=0.9, heavy rust=0.1, plastic/wood=0.0")
    fallback_roughness: float = Field(description="An intelligent guess of the surface roughness (0.0 to 1.0). e.g. polished mirror=0.1, heavy rust or thermal warping=0.9")

class AgentState(TypedDict):
    command: str
    reasoning: str
    intent: dict

def reason_node(state: AgentState):
    """Think and reason deeply about the physics and manufacturing context of the user's command."""
    command = state["command"]
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found in environment variables. Please set it in .env")

    # Use a higher temperature for creative reasoning
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", api_key=api_key, temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert industrial manufacturing and robotics AI orchestrator. "
                   "Analyze the following user command for a factory digital twin. "
                   "Before generating any structured data, reason step-by-step about:\n"
                   "1. What is the precise physical geometry and real-world scale of the requested part?\n"
                   "2. How does the requested defect type physically alter the part's surface materials (roughness, metallicness, RGB diffuse color)?\n"
                   "3. If no CAD model exists, what is the mathematically closest primitive shape (cube, sphere, cylinder, cone, torus) and what should its [X, Y, Z] scale be to perfectly mimic the real object?\n"
                   "4. What exact 6-axis robot joint angles (in radians) would best position a robotic arm to visually inspect this specific defect?"),
        ("human", "Command: {command}\n\nProvide your detailed, expert reasoning step-by-step.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"command": command})
    return {"reasoning": response.content}

def extract_node(state: AgentState):
    """Extract the strict structured intent based on the deep reasoning context."""
    command = state["command"]
    reasoning = state["reasoning"]
    
    api_key = os.getenv("NVIDIA_API_KEY")
    
    # Use zero temperature for strict JSON schema extraction
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", api_key=api_key, temperature=0)
    structured_llm = llm.with_structured_output(OrchestratorIntent)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a rigid structured data extractor. Based on the original user command and the expert reasoning provided, extract the precise values into the exact OrchestratorIntent JSON schema. Infer reasonable defaults for any missing quantities (e.g., 50 images)."),
        ("human", "User Command: {command}\n\nExpert Reasoning Context:\n{reasoning}\n\nGenerate the strict JSON output now.")
    ])
    
    chain = prompt | structured_llm
    response = chain.invoke({"command": command, "reasoning": reasoning})
    return {"intent": response.model_dump()}

# Build the LangGraph Orchestrator
workflow = StateGraph(AgentState)
workflow.add_node("reason", reason_node)
workflow.add_node("extract", extract_node)

workflow.add_edge(START, "reason")
workflow.add_edge("reason", "extract")
workflow.add_edge("extract", END)

orchestrator_app = workflow.compile()

def parse_intent(command: str) -> dict:
    """
    Live implementation of a LangGraph "Think -> Reason -> Act" orchestrator.
    Requires NVIDIA_API_KEY in environment or .env file.
    """
    try:
        # Invoke the state machine
        final_state = orchestrator_app.invoke({
            "command": command, 
            "reasoning": "", 
            "intent": {}
        })
        
        # Log the deep reasoning to the console for transparency
        print("\n" + "="*50)
        print("LANGGRAPH REASONING ENGINE:")
        print("="*50)
        print(final_state["reasoning"])
        print("="*50 + "\n")
        
        return final_state["intent"]
        
    except Exception as e:
        print(f"Error parsing intent with LangGraph Orchestrator: {e}")
        # Robust Fallback for hackathon execution if API/parsing fails
        return {
            "action": "generate_and_train",
            "defect_type": "thermal_warping",
            "part_type": "gear",
            "num_images": 50,
            "deploy": True,
            "robot_joint_angles": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            "fallback_primitive": "torus",
            "fallback_scale": [20.0, 20.0, 20.0],
            "fallback_diffuse": [0.15, 0.15, 0.18],
            "fallback_metallic": 0.8,
            "fallback_roughness": 0.3
        }
