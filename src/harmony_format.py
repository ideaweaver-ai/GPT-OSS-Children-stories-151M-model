"""
Harmony Chat Format Implementation for GPT-OSS

The Harmony format is GPT-OSS's native chat format that supports:
- System messages
- User messages  
- Assistant messages
- Tool use and responses
- Structured outputs

Based on GPT-OSS model card specifications.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json

@dataclass
class HarmonyMessage:
    """A single message in Harmony format"""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool responses

class HarmonyFormatter:
    """Formats conversations in GPT-OSS Harmony format"""
    
    # Special tokens for Harmony format
    SYSTEM_TOKEN = "<|im_start|>system"
    USER_TOKEN = "<|im_start|>user"
    ASSISTANT_TOKEN = "<|im_start|>assistant"
    TOOL_TOKEN = "<|im_start|>tool"
    END_TOKEN = "<|im_end|>"
    
    # Tool use tokens
    TOOL_CALL_START = "<|tool_call_start|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_RESPONSE_START = "<|tool_response_start|>"
    TOOL_RESPONSE_END = "<|tool_response_end|>"
    
    def __init__(self):
        self.supported_tools = {
            "browser": "Web browsing and search capabilities",
            "python": "Python code execution",
            "apply_patch": "Code modification and patching"
        }
    
    def format_message(self, message: HarmonyMessage) -> str:
        """Format a single message in Harmony format"""
        if message.role == "system":
            return f"{self.SYSTEM_TOKEN}\n{message.content}\n{self.END_TOKEN}"
        
        elif message.role == "user":
            return f"{self.USER_TOKEN}\n{message.content}\n{self.END_TOKEN}"
        
        elif message.role == "assistant":
            formatted = f"{self.ASSISTANT_TOKEN}\n"
            
            # Add content if present
            if message.content:
                formatted += message.content
            
            # Add tool calls if present
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    formatted += f"\n{self.TOOL_CALL_START}\n"
                    formatted += json.dumps(tool_call, indent=2)
                    formatted += f"\n{self.TOOL_CALL_END}"
            
            formatted += f"\n{self.END_TOKEN}"
            return formatted
        
        elif message.role == "tool":
            formatted = f"{self.TOOL_TOKEN}\n"
            formatted += f"{self.TOOL_RESPONSE_START}\n"
            formatted += f"Tool: {message.name}\n"
            formatted += f"Call ID: {message.tool_call_id}\n"
            formatted += f"Response: {message.content}\n"
            formatted += f"{self.TOOL_RESPONSE_END}\n"
            formatted += f"{self.END_TOKEN}"
            return formatted
        
        else:
            raise ValueError(f"Unknown role: {message.role}")
    
    def format_conversation(self, messages: List[HarmonyMessage]) -> str:
        """Format a full conversation in Harmony format"""
        formatted_messages = []
        
        for message in messages:
            formatted_messages.append(self.format_message(message))
        
        return "\n".join(formatted_messages)
    
    def create_system_message(self, content: str) -> HarmonyMessage:
        """Create a system message"""
        return HarmonyMessage(role="system", content=content)
    
    def create_user_message(self, content: str) -> HarmonyMessage:
        """Create a user message"""
        return HarmonyMessage(role="user", content=content)
    
    def create_assistant_message(
        self, 
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> HarmonyMessage:
        """Create an assistant message with optional tool calls"""
        return HarmonyMessage(
            role="assistant", 
            content=content, 
            tool_calls=tool_calls
        )
    
    def create_tool_response(
        self, 
        tool_name: str, 
        tool_call_id: str, 
        content: str
    ) -> HarmonyMessage:
        """Create a tool response message"""
        return HarmonyMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name
        )
    
    def create_browser_call(self, query: str, call_id: str = "browser_1") -> Dict[str, Any]:
        """Create a browser tool call"""
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "browser",
                "arguments": json.dumps({"query": query})
            }
        }
    
    def create_python_call(self, code: str, call_id: str = "python_1") -> Dict[str, Any]:
        """Create a Python execution tool call"""
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "python",
                "arguments": json.dumps({"code": code})
            }
        }
    
    def create_patch_call(self, file_path: str, patch: str, call_id: str = "patch_1") -> Dict[str, Any]:
        """Create an apply_patch tool call"""
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "apply_patch",
                "arguments": json.dumps({"file_path": file_path, "patch": patch})
            }
        }

# Example usage and templates
class HarmonyTemplates:
    """Common Harmony format templates for GPT-OSS"""
    
    @staticmethod
    def children_story_system() -> str:
        """System prompt for children's story generation"""
        return """You are a helpful AI assistant specialized in creating engaging, educational, and age-appropriate children's stories. Your stories should be:

- Safe and appropriate for children
- Educational and promote positive values
- Creative and engaging
- Well-structured with clear beginning, middle, and end
- Include simple vocabulary suitable for young readers

You have access to tools for research, code generation, and content modification when needed."""
    
    @staticmethod
    def reasoning_system() -> str:
        """System prompt for reasoning tasks"""
        return """You are GPT-OSS, an AI assistant with strong reasoning capabilities. You can:

- Adjust your reasoning effort based on task complexity
- Provide step-by-step explanations
- Use tools when needed for research or computation
- Generate structured outputs in specified formats

Think carefully about each problem and provide clear, logical reasoning."""
    
    @staticmethod
    def tool_use_example() -> List[HarmonyMessage]:
        """Example of tool use in Harmony format"""
        formatter = HarmonyFormatter()
        
        return [
            formatter.create_system_message(HarmonyTemplates.reasoning_system()),
            formatter.create_user_message("Research the history of the printing press and write a short summary."),
            formatter.create_assistant_message(
                content="I'll research the printing press for you.",
                tool_calls=[formatter.create_browser_call("history of printing press Gutenberg")]
            ),
            formatter.create_tool_response(
                tool_name="browser",
                tool_call_id="browser_1", 
                content="The printing press was invented by Johannes Gutenberg around 1440..."
            ),
            formatter.create_assistant_message(
                content="Based on my research, here's a summary of the printing press history:\n\n**The Printing Press: A Revolutionary Invention**\n\nThe printing press was invented by Johannes Gutenberg around 1440 in Mainz, Germany..."
            )
        ]

def format_for_gpt_oss(messages: List[Dict[str, str]]) -> str:
    """
    Convert standard chat format to GPT-OSS Harmony format
    
    Args:
        messages: List of {"role": str, "content": str} messages
    
    Returns:
        Harmony formatted string
    """
    formatter = HarmonyFormatter()
    harmony_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            harmony_messages.append(formatter.create_system_message(msg["content"]))
        elif msg["role"] == "user":
            harmony_messages.append(formatter.create_user_message(msg["content"]))
        elif msg["role"] == "assistant":
            harmony_messages.append(formatter.create_assistant_message(msg["content"]))
    
    return formatter.format_conversation(harmony_messages)

if __name__ == "__main__":
    # Test the Harmony formatter
    formatter = HarmonyFormatter()
    
    # Create a simple conversation
    messages = [
        formatter.create_system_message("You are a helpful assistant."),
        formatter.create_user_message("Tell me a short story about a robot."),
        formatter.create_assistant_message("Once upon a time, there was a friendly robot named Beep...")
    ]
    
    # Format and print
    formatted = formatter.format_conversation(messages)
    print("=== Harmony Format Example ===")
    print(formatted)
    print("\n=== Tool Use Example ===")
    
    # Tool use example
    tool_messages = HarmonyTemplates.tool_use_example()
    tool_formatted = formatter.format_conversation(tool_messages)
    print(tool_formatted)
