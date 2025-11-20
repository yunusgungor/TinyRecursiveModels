"""
Tool Registry and Management System
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class ToolCall:
    """Represents a tool call with parameters and results"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any = None
    execution_time: float = 0.0
    success: bool = True
    error_message: str = ""
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """Create from dictionary"""
        return cls(**data)


class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.call_count = 0
        self.total_execution_time = 0.0
        
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for parameter validation"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameter_schema()
        }
    
    @abstractmethod
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for this tool"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against schema"""
        # Basic validation - can be extended
        schema = self._get_parameter_schema()
        required = schema.get("required", [])
        
        for param in required:
            if param not in parameters:
                return False
        
        return True
    
    def __call__(self, **kwargs) -> Any:
        """Make tool callable"""
        return self.execute(**kwargs)


class ToolRegistry:
    """Registry for managing and executing tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.call_history: List[ToolCall] = []
        self.cache: Dict[str, Any] = {}
        self.cache_enabled = True
        self.cache_ttl = 3600  # 1 hour default TTL
        
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry"""
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str):
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            print(f"Unregistered tool: {tool_name}")
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool"""
        if tool_name in self.tools:
            return self.tools[tool_name].get_schema()
        return None
    
    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all tools"""
        return {name: tool.get_schema() for name, tool in self.tools.items()}
    
    def _generate_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for tool call"""
        # Create deterministic hash of tool name and parameters
        # Handle non-serializable objects like GiftItem
        serializable_params = {}
        for key, value in parameters.items():
            if isinstance(value, list):
                # Handle list of objects (like GiftItem)
                serializable_list = []
                for item in value:
                    if hasattr(item, 'to_dict'):
                        serializable_list.append(item.to_dict())
                    elif hasattr(item, '__dict__'):
                        serializable_list.append(item.__dict__)
                    else:
                        serializable_list.append(item)
                serializable_params[key] = serializable_list
            elif hasattr(value, 'to_dict'):
                # Object with to_dict method
                serializable_params[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                # Generic object with __dict__
                serializable_params[key] = value.__dict__
            else:
                # Primitive type
                serializable_params[key] = value
        
        param_str = json.dumps(serializable_params, sort_keys=True)
        cache_input = f"{tool_name}:{param_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        if "timestamp" not in cached_data:
            return False
        
        return time.time() - cached_data["timestamp"] < self.cache_ttl
    
    def call_tool(self, tool_call: ToolCall) -> ToolCall:
        """
        Execute a tool call
        
        Args:
            tool_call: ToolCall object with tool_name and parameters
            
        Returns:
            Updated ToolCall object with result
        """
        tool_call.timestamp = time.time()
        
        if tool_call.tool_name not in self.tools:
            tool_call.success = False
            tool_call.error_message = f"Tool '{tool_call.tool_name}' not found"
            return tool_call
        
        tool = self.tools[tool_call.tool_name]
        
        # Validate parameters
        if not tool.validate_parameters(tool_call.parameters):
            tool_call.success = False
            tool_call.error_message = f"Invalid parameters for tool '{tool_call.tool_name}'"
            return tool_call
        
        # Check cache
        cache_key = None
        if self.cache_enabled:
            cache_key = self._generate_cache_key(tool_call.tool_name, tool_call.parameters)
            if self._is_cache_valid(cache_key):
                cached_data = self.cache[cache_key]
                tool_call.result = cached_data["result"]
                tool_call.execution_time = cached_data["execution_time"]
                tool_call.success = True
                # Cache hit - silent for cleaner logs
                return tool_call
        
        # Execute tool
        start_time = time.time()
        try:
            result = tool.execute(**tool_call.parameters)
            tool_call.result = result
            tool_call.success = True
            
            # Update tool statistics
            tool.call_count += 1
            
        except Exception as e:
            tool_call.success = False
            tool_call.error_message = str(e)
            print(f"Tool execution failed: {tool_call.tool_name} - {e}")
        
        # Record execution time
        tool_call.execution_time = time.time() - start_time
        tool.total_execution_time += tool_call.execution_time
        
        # Cache result if successful
        if tool_call.success and self.cache_enabled and cache_key:
            self.cache[cache_key] = {
                "result": tool_call.result,
                "execution_time": tool_call.execution_time,
                "timestamp": time.time()
            }
        
        # Add to history
        self.call_history.append(tool_call)
        
        # Limit history size
        if len(self.call_history) > 1000:
            self.call_history = self.call_history[-500:]
        
        return tool_call
    
    def call_tool_by_name(self, tool_name: str, **parameters) -> ToolCall:
        """Convenience method to call tool by name"""
        tool_call = ToolCall(tool_name=tool_name, parameters=parameters)
        return self.call_tool(tool_call)
    
    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tools"""
        stats = {}
        for name, tool in self.tools.items():
            stats[name] = {
                "call_count": tool.call_count,
                "total_execution_time": tool.total_execution_time,
                "average_execution_time": (
                    tool.total_execution_time / tool.call_count 
                    if tool.call_count > 0 else 0
                )
            }
        return stats
    
    def get_call_history(self, tool_name: Optional[str] = None, 
                        limit: Optional[int] = None) -> List[ToolCall]:
        """Get call history, optionally filtered by tool name"""
        history = self.call_history
        
        if tool_name:
            history = [call for call in history if call.tool_name == tool_name]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_cache(self):
        """Clear the tool call cache"""
        self.cache.clear()
        print("Tool cache cleared")
    
    def clear_history(self):
        """Clear the call history"""
        self.call_history.clear()
        print("Tool call history cleared")
    
    def enable_cache(self, ttl: int = 3600):
        """Enable caching with specified TTL"""
        self.cache_enabled = True
        self.cache_ttl = ttl
        print(f"Tool caching enabled with TTL: {ttl} seconds")
    
    def disable_cache(self):
        """Disable caching"""
        self.cache_enabled = False
        print("Tool caching disabled")
    
    def export_history(self, filepath: str):
        """Export call history to JSON file"""
        history_data = [call.to_dict() for call in self.call_history]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"Tool call history exported to {filepath}")
    
    def import_history(self, filepath: str):
        """Import call history from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        self.call_history = [ToolCall.from_dict(data) for data in history_data]
        print(f"Tool call history imported from {filepath}")


# Example tool implementations
class EchoTool(BaseTool):
    """Simple echo tool for testing"""
    
    def __init__(self):
        super().__init__("echo", "Echoes back the input message")
    
    def execute(self, message: str) -> str:
        return f"Echo: {message}"
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo back"
                }
            },
            "required": ["message"]
        }


class CalculatorTool(BaseTool):
    """Simple calculator tool"""
    
    def __init__(self):
        super().__init__("calculator", "Performs basic arithmetic operations")
    
    def execute(self, operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First operand"
                },
                "b": {
                    "type": "number", 
                    "description": "Second operand"
                }
            },
            "required": ["operation", "a", "b"]
        }


if __name__ == "__main__":
    # Test the tool registry
    registry = ToolRegistry()
    
    # Register tools
    registry.register_tool(EchoTool())
    registry.register_tool(CalculatorTool())
    
    print("Available tools:", registry.list_tools())
    
    # Test echo tool
    echo_call = registry.call_tool_by_name("echo", message="Hello, World!")
    print(f"Echo result: {echo_call.result}")
    
    # Test calculator tool
    calc_call = registry.call_tool_by_name("calculator", operation="add", a=5, b=3)
    print(f"Calculator result: {calc_call.result}")
    
    # Test caching
    calc_call2 = registry.call_tool_by_name("calculator", operation="add", a=5, b=3)
    print(f"Cached result: {calc_call2.result}")
    
    # Get statistics
    stats = registry.get_tool_stats()
    print("Tool statistics:", stats)
    
    # Get schemas
    schemas = registry.get_all_schemas()
    print("Tool schemas:", json.dumps(schemas, indent=2))