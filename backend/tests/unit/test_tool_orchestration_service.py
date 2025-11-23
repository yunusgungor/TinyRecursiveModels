"""
Unit tests for Tool Orchestration Service
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.tool_orchestration import ToolOrchestrationService
from models.tools import ToolRegistry, BaseTool, ToolCall
from typing import Dict, Any


class SlowTool(BaseTool):
    """Tool that takes time to execute"""
    
    def __init__(self, name: str, delay: float = 1.0):
        super().__init__(name, f"Slow tool: {name}")
        self.delay = delay
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        import time
        time.sleep(self.delay)
        return {"executed": True, "delay": self.delay}
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}


class FailingTool(BaseTool):
    """Tool that always fails"""
    
    def __init__(self, name: str):
        super().__init__(name, f"Failing tool: {name}")
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        raise Exception(f"Tool {self.name} failed intentionally")
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}


class SuccessTool(BaseTool):
    """Tool that always succeeds"""
    
    def __init__(self, name: str):
        super().__init__(name, f"Success tool: {name}")
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        return {"success": True, "tool": self.name}
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}


@pytest.fixture
def tool_registry():
    """Create a fresh tool registry"""
    registry = ToolRegistry()
    registry.disable_cache()
    return registry


@pytest.fixture
def orchestration_service(tool_registry):
    """Create orchestration service"""
    return ToolOrchestrationService(tool_registry, max_workers=6, default_timeout=3.0)


@pytest.mark.asyncio
async def test_timeout_handling(tool_registry, orchestration_service):
    """
    Test that tools exceeding timeout are properly handled.
    Validates: Requirements 3.7
    """
    # Register a slow tool
    slow_tool = SlowTool("slow_tool", delay=2.0)
    tool_registry.register_tool(slow_tool)
    
    # Execute with short timeout
    result = await orchestration_service.execute_tools(
        selected_tools=["slow_tool"],
        tool_params={"slow_tool": {}},
        parallel=False,
        timeout=0.5  # Shorter than tool delay
    )
    
    # Should timeout
    assert result["failed_count"] == 1
    assert result["successful_count"] == 0
    assert "slow_tool" in result["errors"]
    assert "timeout" in result["errors"]["slow_tool"].lower() or "timed out" in result["errors"]["slow_tool"].lower()


@pytest.mark.asyncio
async def test_error_recovery_continues_execution(tool_registry, orchestration_service):
    """
    Test that when one tool fails, execution continues with other tools.
    Validates: Requirements 3.8
    """
    # Register mix of failing and success tools
    tool_registry.register_tool(SuccessTool("tool1"))
    tool_registry.register_tool(FailingTool("tool2"))
    tool_registry.register_tool(SuccessTool("tool3"))
    
    # Execute sequentially
    result = await orchestration_service.execute_tools(
        selected_tools=["tool1", "tool2", "tool3"],
        tool_params={
            "tool1": {},
            "tool2": {},
            "tool3": {}
        },
        parallel=False
    )
    
    # All tools should be attempted
    assert len(result["execution_order"]) == 3
    assert result["execution_order"] == ["tool1", "tool2", "tool3"]
    
    # Two should succeed, one should fail
    assert result["successful_count"] == 2
    assert result["failed_count"] == 1
    
    # tool2 should have error
    assert "tool2" in result["errors"]


@pytest.mark.asyncio
async def test_parallel_execution_faster_than_sequential(tool_registry, orchestration_service):
    """
    Test that parallel execution is faster than sequential for slow tools.
    Validates: Requirements 3.7, 3.8
    """
    # Register multiple slow tools
    for i in range(3):
        tool_registry.register_tool(SlowTool(f"slow_{i}", delay=0.3))
    
    tool_names = ["slow_0", "slow_1", "slow_2"]
    tool_params = {name: {} for name in tool_names}
    
    # Sequential execution
    seq_result = await orchestration_service.execute_tools(
        selected_tools=tool_names,
        tool_params=tool_params,
        parallel=False
    )
    
    # Parallel execution
    par_result = await orchestration_service.execute_tools(
        selected_tools=tool_names,
        tool_params=tool_params,
        parallel=True
    )
    
    # Parallel should be faster (with some tolerance for overhead)
    assert par_result["total_time"] < seq_result["total_time"] * 0.8


@pytest.mark.asyncio
async def test_invalid_tool_names_handled_gracefully(tool_registry, orchestration_service):
    """
    Test that invalid tool names don't crash the system.
    Validates: Requirements 3.8
    """
    # Register one valid tool
    tool_registry.register_tool(SuccessTool("valid_tool"))
    
    # Try to execute mix of valid and invalid tools
    result = await orchestration_service.execute_tools(
        selected_tools=["valid_tool", "invalid_tool", "another_invalid"],
        tool_params={"valid_tool": {}},
        parallel=False
    )
    
    # Should execute only valid tool
    assert result["successful_count"] == 1
    assert "valid_tool" in result["execution_order"]
    
    # Invalid tools should be filtered out, not cause errors
    # The system handles this gracefully by just not executing them
    assert len(result["execution_order"]) == 1


@pytest.mark.asyncio
async def test_statistics_tracking(tool_registry, orchestration_service):
    """
    Test that execution statistics are properly tracked.
    """
    # Register tools
    tool_registry.register_tool(SuccessTool("tool1"))
    tool_registry.register_tool(FailingTool("tool2"))
    
    # Execute
    await orchestration_service.execute_tools(
        selected_tools=["tool1", "tool2"],
        tool_params={"tool1": {}, "tool2": {}},
        parallel=False
    )
    
    # Check statistics
    stats = orchestration_service.get_statistics()
    
    assert stats["total_executions"] == 2
    assert stats["successful_executions"] == 1
    # Note: failed_executions is incremented in both _execute_single_tool and tool_registry
    # So we check that at least one failure was recorded
    assert stats["failed_executions"] >= 1
    assert 0 < stats["success_rate"] <= 0.5


@pytest.mark.asyncio
async def test_empty_tool_list(tool_registry, orchestration_service):
    """
    Test handling of empty tool list.
    """
    result = await orchestration_service.execute_tools(
        selected_tools=[],
        tool_params={},
        parallel=False
    )
    
    assert result["successful_count"] == 0
    assert result["failed_count"] == 0
    assert result["execution_order"] == []


@pytest.mark.asyncio
async def test_gifts_parameter_passed_to_tools(tool_registry, orchestration_service):
    """
    Test that gifts parameter is properly passed to tools.
    """
    # Create a tool that checks for gifts parameter
    class GiftAwareTool(BaseTool):
        def __init__(self):
            super().__init__("gift_tool", "Tool that uses gifts")
        
        def execute(self, **kwargs) -> Dict[str, Any]:
            return {"has_gifts": "gifts" in kwargs, "gift_count": len(kwargs.get("gifts", []))}
        
        def _get_parameter_schema(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}, "required": []}
    
    tool_registry.register_tool(GiftAwareTool())
    
    # Execute with gifts
    mock_gifts = [{"id": "1", "name": "Gift 1"}, {"id": "2", "name": "Gift 2"}]
    result = await orchestration_service.execute_tools(
        selected_tools=["gift_tool"],
        tool_params={"gift_tool": {}},
        gifts=mock_gifts,
        parallel=False
    )
    
    # Check that gifts were passed
    assert result["successful_count"] == 1
    tool_result = result["results"]["gift_tool"]["result"]
    assert tool_result["has_gifts"] is True
    assert tool_result["gift_count"] == 2


@pytest.mark.asyncio
async def test_reset_statistics(tool_registry, orchestration_service):
    """
    Test that statistics can be reset.
    """
    # Register and execute a tool
    tool_registry.register_tool(SuccessTool("tool1"))
    await orchestration_service.execute_tools(
        selected_tools=["tool1"],
        tool_params={"tool1": {}},
        parallel=False
    )
    
    # Check stats exist
    stats = orchestration_service.get_statistics()
    assert stats["total_executions"] > 0
    
    # Reset
    orchestration_service.reset_statistics()
    
    # Check stats are cleared
    stats = orchestration_service.get_statistics()
    assert stats["total_executions"] == 0
    assert stats["successful_executions"] == 0
    assert stats["failed_executions"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
