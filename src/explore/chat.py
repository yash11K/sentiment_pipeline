from utils.bedrock import BedrockClient
from typing import Dict
from utils.logger import get_logger

logger = get_logger(__name__)


class ChatEngine:
    def __init__(self):
        self.bedrock = BedrockClient()
        logger.chat("Chat engine initialized")

    def chat(self, query: str) -> Dict:
        logger.chat(f"Processing query: {query[:50]}...")
        logger.llm("Using RetrieveAndGenerate via Knowledge Base")
        result = self.bedrock.retrieve_and_generate(query)
        logger.success("Response generated")
        return result

    def chat_stream(self, query: str):
        """Streaming variant — yields event dicts from the KB stream."""
        logger.chat(f"Processing streaming query: {query[:50]}...")
        logger.llm("Using RetrieveAndGenerateStream via Knowledge Base")
        yield from self.bedrock.retrieve_and_generate_stream(query)
        logger.success("Stream completed")
