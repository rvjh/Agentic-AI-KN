## in this project Agents will start communicating with vector databases and start 
## questioning different questions


import typer
from typing import Optional
from phi.agent import Agent
from phi.model.groq import Groq
from phi.assistant import Assistant
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the SentenceTransformerEmbedder with the desired dimension size
embedder = SentenceTransformerEmbedder(dimensions=384)

# Set up the PostgreSQL database connection string (adjust as needed)
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Create a knowledge base from a PDF document (this example uses a public PDF URL)
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(
        collection="recipes",  # Name of the vector database collection
        db_url=db_url,  # PostgreSQL connection URL
        embedder=embedder,  # Embedder used for vector generation
    )
)

# Load the knowledge base (index PDFs for querying)
knowledge_base.load()

# Create a storage system for agents using PostgreSQL to save agent state/history
storage = PgAgentStorage(table_name="pdf_assistant", db_url=db_url)

# Function to initiate or continue an agent's session
def pdf_assistant(new: bool = False, user: str = "user"):
    """
    Start a new agent session or continue an existing one.
    Arguments:
    - new: If True, it starts a new session; if False, it continues an existing session.
    - user: Username for the assistant session.
    """

    # If no run_id is provided, generate a new one or continue from previous session.
    run_id: Optional[str] = None

    # Initialize the Assistant with the necessary components: model, knowledge base, and storage
    assistant = Agent(
        model=Groq(id="llama-3.3-70b-versatile", embedder=embedder),  # Model details
        run_id=run_id,  # Optionally set run_id for session tracking
        user_id=user,  # Identify the user
        knowledge_base=knowledge_base,  # Knowledge base for querying
        storage=storage,  # Where to store agent data
        show_tool_calls=True,  # Optionally show tool calls in responses
        search_knowledge=True,  # Enable knowledge base search
        read_chat_history=True,  # Enable reading of chat history
    )

    # Print the status of the session (whether it's new or continuing)
    if run_id is None:
        run_id = assistant.run_id  # Get the generated run_id for the session
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    # Start the assistant's interactive CLI application
    assistant.cli_app(markdown=True)

# Main entry point for the script (runs the pdf_assistant function)
if __name__ == "__main__":
    # Typer allows us to run the function via CLI with optional arguments
    typer.run(pdf_assistant)

    
    

### Libraries installed
"""
sqlalchemy
pgvector
## psycopg-binary
psycopg[binary]
pypdf
"""