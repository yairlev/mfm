import zipfile
import os
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import List, Dict, TypedDict, Any, Optional
from langchain.globals import set_debug
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.cache import SQLiteCache
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain.schema.runnable import RunnableParallel, Runnable
from dotenv import load_dotenv
from langgraph.graph.state import RunnableConfig
load_dotenv()

set_debug(False)
set_llm_cache(SQLiteCache(".langchain.db"))

tracer_provider = register(
  project_name="mfm", # Default is 'default'
  endpoint="http://localhost:4317",  # Sends traces using gRPC
)  

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Initialize LLM
MODEL_NAME=os.getenv("VERTEX_MODEL_NAME")
PROJECT=os.getenv("VERTEX_PROJECT")
PROJECT_LOCATION=os.getenv("VERTEX_PROJECT_LOCATION", "us-central1")

# --- Define the State of the Graph ---
class GraphState(TypedDict):
    zip_path: str
    jcl_summaries: Dict[str, str]
    jcl_categories: List[str]
    jcls: Dict[str, str]
    cobols: Dict[str, str]
    procs: Dict[str, str]
    copybooks: Dict[str, str]
    models: List[str]
    arbiter: str

def _get_llm(model_name: str) -> BaseChatModel:
    return ChatVertexAI(model_name=model_name, project=PROJECT, location=PROJECT_LOCATION, temperature=0) #Lower temperature for more consistent results

def _get_llms(model_names: List[str]) -> List[BaseChatModel]:
    return [_get_llm(model_name) for model_name in model_names] #Lower temperature for more consistent results

def parse_jcl_programs_and_procs(jcl_content: str) -> Dict[str, List[str]]:
    """
    Parses JCL content using an LLM to identify external programs and procedures.
    Returns a dictionary with 'programs' and 'procs' keys, each holding a list of program/procedure names.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an expert in parsing JCL code. Extract all external program names and procedure names including those referenced through symbolic parameters (e.g., &PROG). 
            For symbolic references:
            - If the symbolic value can be determined from SET statements or PROC parameters within the JCL, include the resolved name
            - If the symbolic value cannot be determined, exclude it from the results
            
            Return a JSON object with 'programs' and 'procs' keys, each containing a list of unique strings. If no programs or procs are found, return an empty list for each key."""),
            ("user", "JCL Code: ```jcl\n{jcl_content}\n```")
        ]
    )
    chain = prompt | _get_llm(MODEL_NAME) | JsonOutputParser()
    assets = chain.invoke({"jcl_content": jcl_content})
    assets = {k: [x.upper() for x in v] for k, v in assets.items()}
    return assets
    

def summarize_jcl_with_llm(jcl_content: str, referenced_program_names: List[str], referenced_proc_names: List[str], programs: Dict[str, str], procs: Dict[str, str]) -> str:
    """Summarizes a JCL job using an LLM, incorporating program and procedure context."""
    # Filter programs and procs to only include referenced ones
    referenced_programs = {name: content for name, content in programs.items() if name in referenced_program_names}
    referenced_procs = {name: content for name, content in procs.items() if name in referenced_proc_names}
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Analyze the provided JCL job and its associated programs/procedures to provide a high-level business description. Focus on:
            - The business purpose and value of this job
            - What business processes or functions it supports
            - The key business outcomes or results
            - Any critical business data or systems involved
            
            Avoid technical details unless they're essential to understanding the business context. Return only the business-focused summary as a string."""),
            ("user", f"JCL Code: ```jcl\n{jcl_content}\n```" + 
                    (f"\nAssociated Programs:\n" + "\n".join([f"{name}:\n```cobol\n{content}\n```" for name, content in referenced_programs.items()]) if referenced_programs else "") +
                    (f"\nAssociated Procedures:\n" + "\n".join([f"{name}:\n```jcl\n{content}\n```" for name, content in referenced_procs.items()]) if referenced_procs else ""))
        ]
    )
    chain = prompt | _get_llm(MODEL_NAME) | StrOutputParser()
    try:
      return chain.invoke({})
    except Exception as e:
      print(f"Error summarizing JCL: {e}")
      return f"Error summarizing JCL job. Content: {jcl_content}"

def categorize_jcl_summaries_with_llm(jcl_summaries: Dict[str,str], models: List[str], arbiter: Optional[str]) -> Dict[str, Dict[str, str]]:
  """Categorize a list of JCL summaries using an LLM"""
  prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are given one or more JCL summaries. Your task is to:
            1. Identify meaningful categories/domains that these JCL jobs belong to
            2. Assign each JCL job to one of these categories
            3. Provide a brief description for each category

            Return a JSON object with:
            - 'categories': A dictionary where:
                - keys are category names
                - values are category descriptions
            - 'jcls': A dictionary where:
                - keys are JCL file names
                - values are the assigned category names

            Example:
            {{
                'categories': {{
                    'Payroll': 'Jobs handling employee compensation and benefits',
                    'Reporting': 'Jobs generating business reports'
                }},
                'jcls': {{
                    'PAY001': 'Payroll',
                    'RPT002': 'Reporting'
                }}
            }}"""),
            ("user", f"JCL Summaries:\n{chr(10).join([f'File: {k}\nSummary: {v}' for k,v in jcl_summaries.items()])}")
        ]
    )
  if arbiter:
    chain = RunnableConsortium(prompt, _get_llms(models), _get_llm(arbiter))
  else:
    chain = prompt | _get_llm(MODEL_NAME) | JsonOutputParser()
  return chain.invoke({})


class RunnableConsortium(Runnable):
    """
    A runnable that executes the same prompt across multiple LLMs and uses an arbiter to select the best response.
    The arbiter evaluates responses from all models and synthesizes a final response based on consensus.
    """
    prompt: ChatPromptTemplate
    llms: List[BaseChatModel]
    arbiter: BaseChatModel

    def __init__(self, prompt: ChatPromptTemplate, llms: List[BaseChatModel], arbiter: BaseChatModel):
        super().__init__()
        self.prompt=prompt
        self.llms=llms 
        self.arbiter=arbiter
        
        # Create the parallel chain for model responses
        self.parallel_chain = RunnableParallel({
            llm.model_name: (self.prompt | llm | StrOutputParser())
            for llm in self.llms
        })

        # Create arbiter prompt
        self.arbiter_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Please follow these steps to complete your task:

1. Carefully analyze the original prompt and model responses.
2. Extract and list key points from each model response.
3. Compare and contrast the key points from different responses.
4. Evaluate the relevance of each response to the original prompt.
5. Identify areas of agreement and disagreement among the responses.
6. Synthesize a final response that represents the best consensus.
7. Determine your confidence level in the synthesized response.
8. Highlight any important dissenting views.
9. Assess whether further iterations are needed.
10. If further iterations are needed, provide recommendations for refinement areas.

In your thought process, consider the following questions:
- What are the key points addressed by each model response?
- How do the responses align or differ from each other?
- What are the strengths and weaknesses of each response?
- Are there any unique insights or perspectives offered by specific responses?
- How well does each response address the original prompt?

After your thought process, provide your synthesized output using the following JSON format:
{{
  "final_response": "Your synthesized output here",
  "thought_process": "Your thought process here",
  "confidence": "Your confidence level here (e.g., 'High', 'Medium', 'Low')",
  "dissenting_views": "Any important dissenting views here",
  "recommendations": "Any recommendations for refinement areas here"
}}
"""),
            ("user", """Original Question: {question}
            
            Model Responses:
            {responses}
            """)
        ])

    def invoke(self, input: Dict[str, Any]) -> str:
        # Get responses from all models in parallel
        responses = self.parallel_chain.invoke(input)
        
        # Format responses for arbiter
        formatted_responses = "\n\n".join([
            f"{name}:\n{response}" 
            for name, response in responses.items()
        ])
        
        # Have arbiter choose best response
        arbiter_input = {
            "question": self.prompt.format_messages(**input)[0].content,
            "responses": formatted_responses
        }
        
        final_response = (
            self.arbiter_prompt 
            | self.arbiter 
            | JsonOutputParser()
        ).invoke(arbiter_input)
        
        return final_response
    
def _read_file_from_zip(zip_ref: zipfile.ZipFile, file_info: zipfile.ZipInfo) -> str:
    """
    Helper to read text content from a single file inside the ZIP.
    Decodes as UTF-8, ignoring any errors. Adjust as needed for EBCDIC or other encodings.
    """
    with zip_ref.open(file_info, 'r') as f:
        return f.read().decode("utf-8", errors="replace")

# --- Define Graph Nodes ---
def extract_mainframe_assets_from_zip(state) -> Dict[str, Dict[str, str]]:
    """
    Extracts JCL, COBOL, Copybook, and PROC file contents from a ZIP.
    
    Returns a JSON object with four top-level keys: "jcls", "cobols", "copybooks", and "procs".
    Each of these is itself a dict mapping:
        { base_name_of_file : file_content_as_string }
    """
    # Define acceptable extensions for each category
    jcl_exts    = {".jcl"}
    cobol_exts  = {".cbl", ".cob", ".cobol"}  # Adjust or add as needed
    copy_exts   = {".cpy", ".copy"}          # Adjust or add as needed
    proc_exts   = {".proc", ".prc"}          # Adjust or add as needed

    # Prepare dictionaries to hold { base_name : file_content }
    jcls_dict     = {}
    cobols_dict   = {}
    copybooks_dict= {}
    procs_dict    = {}

    zip_path = state.get('zip_path')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Skip directories
                if file_info.is_dir():
                    continue

                # Extract extension
                filename_lower = file_info.filename.lower()
                _, ext = os.path.splitext(filename_lower)

                # Base name without extension
                # e.g., "MY_APP.cbl" -> "MY_APP"
                base_name, _ = os.path.splitext(os.path.basename(file_info.filename.upper()))

                # Check if extension matches one of the categories
                if ext in jcl_exts:
                    file_content = _read_file_from_zip(zip_ref, file_info)
                    jcls_dict[base_name] = file_content
                elif ext in cobol_exts:
                    file_content = _read_file_from_zip(zip_ref, file_info)
                    cobols_dict[base_name] = file_content
                elif ext in copy_exts:
                    file_content = _read_file_from_zip(zip_ref, file_info)
                    copybooks_dict[base_name] = file_content
                elif ext in proc_exts:
                    file_content = _read_file_from_zip(zip_ref, file_info)
                    procs_dict[base_name] = file_content

        return {
            "jcls": jcls_dict,
            "cobols": cobols_dict,
            "copybooks": copybooks_dict,
            "procs": procs_dict
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid zip file: {zip_path}")

def process_jcl_files(state):
    summaries = {}
    for jcl_file_name, jcl_content in state["jcls"].items():
        try:
            assets = parse_jcl_programs_and_procs(jcl_content)
            referenced_programs = assets.get("programs",[])
            referenced_procs = assets.get("procs",[])
            summary = summarize_jcl_with_llm(jcl_content, referenced_programs, referenced_procs, state["cobols"], state["procs"])
            summaries[jcl_file_name] = summary
        except Exception as e:
            print(f"Error processing JCL file {jcl_file_name}: {e}")
            summaries[jcl_file_name] = f"Error processing JCL file: {e}"
    return {"jcl_summaries": summaries}

def categorize_jcl_files(state: GraphState, config: RunnableConfig):
    categories = categorize_jcl_summaries_with_llm(state["jcl_summaries"], config["configurable"].get("models"), config["configurable"].get("arbiter"))
    return {"jcl_categories": categories}


class ConfigSchema(TypedDict):
    models: List[str]
    arbiter: Optional[str]

# --- Build the Graph ---
workflow = StateGraph(GraphState, ConfigSchema)
workflow.add_node("extract_assets", extract_mainframe_assets_from_zip)
workflow.add_node("process_jcl", process_jcl_files)
workflow.add_node("categorize_jcl", categorize_jcl_files)

workflow.add_edge("extract_assets", "process_jcl")
workflow.add_edge("process_jcl", "categorize_jcl")
workflow.add_edge("categorize_jcl", END)

workflow.set_entry_point("extract_assets")

# --- Run the Graph ---
chain = workflow.compile()

def analyze_jcl_zip(zip_file_path: str, models: List[str], arbiter: Optional[str]):
    initial_state = {"zip_path": zip_file_path, "jcls": {}, "cobols": {}, "procs": {}, "copybooks": {}, "jcl_summaries": {}, "jcl_categories": [], "temp": {}}
    config = {"configurable": {"models": models, "arbiter": arbiter}}
    try:
      result = chain.invoke(initial_state, config)
      return result["jcl_categories"]
    except Exception as e:
      print(f"An error occured: {e}")
      return {}, []

# Create a dummy zip file with jcl files for testing
def create_dummy_zip():
    if not os.path.exists("./dummy_jcls"):
      os.makedirs("./dummy_jcls")
    with open("./dummy_jcls/job1.jcl", "w") as f:
        f.write("""
        //JOB1     JOB (ACCT,CLASS),
        //         'PROGRAM RUN'
        //STEP1    EXEC PGM=PROGRAM1
        //STEP2    EXEC PROC=PROC1
        //STEP3    EXEC PGM=PROGRAM2,PARM='ABC'
        """)
    with open("./dummy_jcls/job2.jcl", "w") as f:
        f.write("""
        //JOB2     JOB (ACCT,CLASS),
        //         'ANOTHER PROGRAM'
        //STEP1    EXEC PGM=PROGRAM3
        //STEP2    EXEC PROC=PROC2
        """)

    with zipfile.ZipFile("jcl_files.zip", "w") as zipf:
      zipf.write("./dummy_jcls/job1.jcl", arcname="job1.jcl")
      zipf.write("./dummy_jcls/job2.jcl", arcname="job2.jcl")

    import shutil
    shutil.rmtree("./dummy_jcls")


if __name__ == "__main__":
    zip_file_path = "path/to/zip/file"
    categories = analyze_jcl_zip(zip_file_path, ["gemini-2.0-flash-exp", "gemini-1.5-flash-002"], "gemini-2.0-flash-exp")
    print("JCL Categories:", categories)