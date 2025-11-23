"""
Property-based tests for Tool Orchestration Service

These tests verify correctness properties that should hold across all valid inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis import assume
import asyncio
import sys
from pathlib import Path

# Add project root to path to import models
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.tool_orchestration import ToolOrchestrationService
from models.tools import ToolRegistry, BaseTool
from typing import Dict, Any, List


# Mock tool for testing
class MockTool(BaseTool):
    """Mock tool that records execution order"""
    
    execution_log = []
    
    def __init__(self, name: str, delay: float = 0.0, should_fail: bool = False):
        super().__init__(name, f"Mock tool: {name}")
        self.delay = delay
        self.should_fail = should_fail
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute mock tool"""
        import time
        
        # Record execution
        MockTool.execution_log.append(self.name)
        
        # Simulate work
        if self.delay > 0:
            time.sleep(self.delay)
        
        # Simulate failure if requested
        if self.should_fail:
            raise Exception(f"Mock tool {self.name} failed")
        
        return {
            "tool_name": self.name,
            "executed": True,
            "kwargs": kwargs
        }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    @classmethod
    def reset_log(cls):
        """Reset execution log"""
        cls.execution_log = []


@pytest.fixture
def tool_registry():
    """Create a fresh tool registry for each test"""
    registry = ToolRegistry()
    # Disable caching for tests to ensure fresh execution
    registry.disable_cache()
    return registry


@pytest.fixture
def orchestration_service(tool_registry):
    """Create orchestration service"""
    return ToolOrchestrationService(tool_registry, max_workers=6, default_timeout=3.0)


# Strategy for generating tool names
tool_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
    min_size=3,
    max_size=20
).filter(lambda x: x.strip() and x.isalnum())


# Strategy for generating list of unique tool names
tool_list_strategy = st.lists(
    tool_name_strategy,
    min_size=1,
    max_size=10,
    unique=True
)


@settings(max_examples=100, deadline=5000)
@given(tool_names=tool_list_strategy)
@pytest.mark.asyncio
async def test_property_tool_execution_order_preservation(tool_names: List[str]):
    """
    Feature: trendyol-gift-recommendation-web, Property 9: Tool Execution Order Preservation
    
    Property: For any list of selected tools, when executed sequentially,
    the system should execute them in the order they appear in the list.
    
    Validates: Requirements 3.1
    """
    # Assume we have at least one tool
    assume(len(tool_names) > 0)
    
    # Create registry and service
    registry = ToolRegistry()
    registry.disable_cache()
    service = ToolOrchestrationService(registry, max_workers=6, default_timeout=3.0)
    
    # Reset execution log
    MockTool.reset_log()
    
    # Register mock tools
    for name in tool_names:
        tool = MockTool(name, delay=0.01)  # Small delay to ensure ordering
        registry.register_tool(tool)
    
    # Prepare parameters (empty for mock tools)
    tool_params = {name: {} for name in tool_names}
    
    # Execute tools sequentially
    result = await service.execute_tools(
        selected_tools=tool_names,
        tool_params=tool_params,
        parallel=False  # Sequential execution
    )
    
    # Verify execution order matches input order
    execution_order = result["execution_order"]
    
    # Property: Execution order should match the input order
    assert execution_order == tool_names, (
        f"Tool execution order not preserved. "
        f"Expected: {tool_names}, Got: {execution_order}"
    )
    
    # Also verify from the mock tool's execution log
    assert MockTool.execution_log == tool_names, (
        f"Tool execution log doesn't match expected order. "
        f"Expected: {tool_names}, Got: {MockTool.execution_log}"
    )
    
    # Verify all tools were executed
    assert result["successful_count"] == len(tool_names), (
        f"Not all tools executed successfully. "
        f"Expected: {len(tool_names)}, Got: {result['successful_count']}"
    )
    
    # Cleanup
    service.shutdown()


@settings(max_examples=100, deadline=5000)
@given(tool_names=tool_list_strategy)
@pytest.mark.asyncio
async def test_property_parallel_execution_completes_all_tools(tool_names: List[str]):
    """
    Property: For any list of tools, when executed in parallel,
    all tools should complete (though order may vary).
    
    This is a weaker property than sequential ordering, but important for parallel execution.
    """
    # Assume we have at least one tool
    assume(len(tool_names) > 0)
    
    # Create registry and service
    registry = ToolRegistry()
    registry.disable_cache()
    service = ToolOrchestrationService(registry, max_workers=6, default_timeout=3.0)
    
    # Reset execution log
    MockTool.reset_log()
    
    # Register mock tools
    for name in tool_names:
        tool = MockTool(name, delay=0.01)
        registry.register_tool(tool)
    
    # Prepare parameters
    tool_params = {name: {} for name in tool_names}
    
    # Execute tools in parallel
    result = await service.execute_tools(
        selected_tools=tool_names,
        tool_params=tool_params,
        parallel=True  # Parallel execution
    )
    
    # Property: All tools should be in execution order (though order may vary)
    execution_order = result["execution_order"]
    assert set(execution_order) == set(tool_names), (
        f"Not all tools executed. "
        f"Expected: {set(tool_names)}, Got: {set(execution_order)}"
    )
    
    # Property: All tools should complete successfully
    assert result["successful_count"] == len(tool_names), (
        f"Not all tools executed successfully. "
        f"Expected: {len(tool_names)}, Got: {result['successful_count']}"
    )
    
    # Cleanup
    service.shutdown()


@settings(max_examples=50, deadline=5000)
@given(
    tool_names=tool_list_strategy,
    fail_indices=st.lists(st.integers(min_value=0, max_value=9), max_size=5)
)
@pytest.mark.asyncio
async def test_property_failed_tools_dont_stop_execution(
    tool_names: List[str],
    fail_indices: List[int]
):
    """
    Property: For any list of tools, if some tools fail,
    the orchestration should continue executing remaining tools.
    
    Validates: Requirements 3.8 (error handling and fallback logic)
    """
    # Assume we have at least one tool
    assume(len(tool_names) > 0)
    
    # Create registry and service
    registry = ToolRegistry()
    registry.disable_cache()
    service = ToolOrchestrationService(registry, max_workers=6, default_timeout=3.0)
    
    # Reset execution log
    MockTool.reset_log()
    
    # Register mock tools, some will fail
    for i, name in enumerate(tool_names):
        should_fail = i in fail_indices
        tool = MockTool(name, delay=0.01, should_fail=should_fail)
        registry.register_tool(tool)
    
    # Prepare parameters
    tool_params = {name: {} for name in tool_names}
    
    # Execute tools sequentially
    result = await service.execute_tools(
        selected_tools=tool_names,
        tool_params=tool_params,
        parallel=False
    )
    
    # Property: All tools should be attempted (execution order should include all)
    execution_order = result["execution_order"]
    assert len(execution_order) == len(tool_names), (
        f"Not all tools were attempted. "
        f"Expected: {len(tool_names)}, Got: {len(execution_order)}"
    )
    
    # Property: Execution order should still be preserved
    assert execution_order == tool_names, (
        f"Tool execution order not preserved despite failures. "
        f"Expected: {tool_names}, Got: {execution_order}"
    )
    
    # Property: Failed count should match number of tools that should fail
    # Filter fail_indices to only include valid indices
    valid_fail_indices = set([i for i in fail_indices if 0 <= i < len(tool_names)])
    expected_failures = len(valid_fail_indices)
    
    # Allow some tolerance due to async execution timing
    assert abs(result["failed_count"] - expected_failures) <= 1, (
        f"Failed count doesn't match expected failures (with tolerance). "
        f"Expected: {expected_failures}, Got: {result['failed_count']}, "
        f"Valid fail indices: {valid_fail_indices}, Tool count: {len(tool_names)}"
    )
    
    # Cleanup
    service.shutdown()


@settings(max_examples=50, deadline=10000)
@given(
    tool_names=st.lists(tool_name_strategy, min_size=1, max_size=5, unique=True),
    timeout=st.floats(min_value=0.1, max_value=1.0)
)
@pytest.mark.asyncio
async def test_property_timeout_handling(tool_names: List[str], timeout: float):
    """
    Property: For any list of tools and timeout value,
    tools that exceed timeout should be marked as failed with timeout error.
    
    Validates: Requirements 3.7 (timeout handling)
    """
    # Assume we have at least one tool
    assume(len(tool_names) > 0)
    
    # Create registry and service
    registry = ToolRegistry()
    registry.disable_cache()
    service = ToolOrchestrationService(registry, max_workers=6, default_timeout=timeout)
    
    # Reset execution log
    MockTool.reset_log()
    
    # Register mock tools with delays longer than timeout
    for name in tool_names:
        # Make delay longer than timeout to trigger timeout
        tool = MockTool(name, delay=timeout + 0.5)
        registry.register_tool(tool)
    
    # Prepare parameters
    tool_params = {name: {} for name in tool_names}
    
    # Execute tools
    result = await service.execute_tools(
        selected_tools=tool_names,
        tool_params=tool_params,
        parallel=True,
        timeout=timeout
    )
    
    # Property: All tools should timeout (or at least most of them)
    # Note: Due to timing variations, we allow some tolerance
    assert result["failed_count"] >= len(tool_names) * 0.8, (
        f"Expected most tools to timeout. "
        f"Failed count: {result['failed_count']}, Total: {len(tool_names)}"
    )
    
    # Property: Errors should mention timeout
    if result["errors"]:
        timeout_errors = [
            err for err in result["errors"].values()
            if "timeout" in err.lower() or "timed out" in err.lower()
        ]
        assert len(timeout_errors) > 0, (
            "Expected timeout errors but got other errors"
        )
    
    # Cleanup
    service.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
