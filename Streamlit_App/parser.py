from langchain_core.output_parsers import BaseOutputParser

class CustomOutputParser(BaseOutputParser):
    """LangChain output parser"""
    
    def parse(self, text: str) -> str:
        """Parse and format response with logging"""
        formatted = text.strip()
        print(formatted)
        return formatted
    
    @property
    def _type(self) -> str:
        return "custom_rag_parser"
    
    def get_format_instructions(self) -> str:
        return "Returns raw text response with whitespace trimming"