import os
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

# ---- LangChain imports -------------------------------------------------
#   – PromptTemplate builds the prompt string
#   – HumanMessage is the message type expected by the Groq chat model
#   – ChatGroq is the Groq‑enabled LLM wrapper
# -----------------------------------------------------------------------
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq   


class GroqLLM:
    def __init__(self, model_name: str = "gemma2-9b-it", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY env var or pass api_key."
            )

        # Initialise the LangChain Groq chat model
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024,
        )
        print(f"Initialized Groq LLM model: {self.model_name}")


    def generate_response(self, prompt: str) -> str:
        try:
            messages = [HumanMessage(content = prompt)]
            response = self.llm.invoke(messages)          # ← Groq call
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"

