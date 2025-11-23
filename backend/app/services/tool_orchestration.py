"""
Tool Orchestration Service

This service manages the execution of multiple tools in parallel or sequentially,
with timeout handling, error recovery, and result aggregation.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging

import sys
from pathlib import Path

# Add project root to path to import models
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.exceptions import ToolExecutionError, ToolTimeoutError
from models.tools import ToolRegistry, ToolCall


logger = logging.getLogger(__name__)


class ToolOrchestrationService:
    """
    Service for orchestrating tool execution with timeout, parallel execution,
    and error handling capabilities.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_workers: int = 6,
        default_timeout: float = 3.0
    ):
        """
        Initialize the tool orchestration service.
        
        Args:
            tool_registry: Registry containing available tools
            max_workers: Maximum number of parallel tool executions
            default_timeout: Default timeout for tool execution in seconds
        """
        self.tool_registry = tool_registry
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.default_timeout = default_timeout
        
        # Statistics tracking
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "total_execution_time": 0.0
        }
        
        logger.info(
            f"ToolOrchestrationService initialized with {max_workers} workers, "
            f"default timeout: {default_timeout}s"
        )
    
    async def execute_tools(
        self,
        selected_tools: List[str],
        tool_params: Dict[str, Dict[str, Any]],
        gifts: Optional[List[Any]] = None,
        parallel: bool = True,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute selected tools with given parameters.
        
        Args:
            selected_tools: List of tool names to execute
            tool_params: Dictionary mapping tool names to their parameters
            gifts: Optional list of gift items to pass to tools
            parallel: Whether to execute tools in parallel (True) or sequentially (False)
            timeout: Timeout for each tool execution (uses default if None)
            
        Returns:
            Dictionary with tool results and execution metadata
            
        Raises:
            ToolExecutionError: If critical tool execution fails
        """
        if timeout is None:
            timeout = self.default_timeout
        
        start_time = time.time()
        
        logger.info(
            f"Executing {len(selected_tools)} tools: {selected_tools}, "
            f"parallel={parallel}, timeout={timeout}s"
        )
        
        # Validate tools exist
        available_tools = self.tool_registry.list_tools()
        invalid_tools = [t for t in selected_tools if t not in available_tools]
        if invalid_tools:
            logger.warning(f"Invalid tools requested: {invalid_tools}")
            selected_tools = [t for t in selected_tools if t in available_tools]
        
        if not selected_tools:
            logger.warning("No valid tools to execute")
            return {
                "results": {},
                "execution_order": [],
                "total_time": 0.0,
                "successful_count": 0,
                "failed_count": 0,
                "errors": {"all_tools_invalid": invalid_tools}
            }
        
        # Execute tools
        if parallel:
            results, execution_order = await self._execute_parallel(
                selected_tools, tool_params, gifts, timeout
            )
        else:
            results, execution_order = await self._execute_sequential(
                selected_tools, tool_params, gifts, timeout
            )
        
        # Aggregate results
        total_time = time.time() - start_time
        successful_count = sum(1 for r in results.values() if r.get("success", False))
        failed_count = len(results) - successful_count
        
        # Extract errors
        errors = {
            tool_name: result.get("error")
            for tool_name, result in results.items()
            if not result.get("success", False) and result.get("error")
        }
        
        # Update statistics
        self.execution_stats["total_executions"] += len(selected_tools)
        self.execution_stats["successful_executions"] += successful_count
        self.execution_stats["failed_executions"] += failed_count
        self.execution_stats["total_execution_time"] += total_time
        
        logger.info(
            f"Tool execution completed: {successful_count} successful, "
            f"{failed_count} failed, total time: {total_time:.2f}s"
        )
        
        return {
            "results": results,
            "execution_order": execution_order,
            "total_time": total_time,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "errors": errors if errors else None
        }
    
    async def _execute_parallel(
        self,
        tool_names: List[str],
        tool_params: Dict[str, Dict[str, Any]],
        gifts: Optional[List[Any]],
        timeout: float
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute tools in parallel.
        
        Args:
            tool_names: List of tool names to execute
            tool_params: Parameters for each tool
            gifts: Optional gift items
            timeout: Timeout for each tool
            
        Returns:
            Tuple of (results dict, execution order list)
        """
        # Create tasks for parallel execution
        tasks = []
        for tool_name in tool_names:
            params = tool_params.get(tool_name, {})
            # Add gifts to params if provided
            if gifts is not None:
                params = {**params, "gifts": gifts}
            
            task = self._execute_single_tool(tool_name, params, timeout)
            tasks.append(task)
        
        # Execute all tasks in parallel
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        results = {}
        execution_order = []
        
        for tool_name, result in zip(tool_names, results_list):
            if isinstance(result, Exception):
                logger.error(f"Tool {tool_name} raised exception: {result}")
                results[tool_name] = {
                    "success": False,
                    "error": str(result),
                    "execution_time": 0.0
                }
            else:
                results[tool_name] = result
            
            execution_order.append(tool_name)
        
        return results, execution_order
    
    async def _execute_sequential(
        self,
        tool_names: List[str],
        tool_params: Dict[str, Dict[str, Any]],
        gifts: Optional[List[Any]],
        timeout: float
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute tools sequentially in order.
        
        Args:
            tool_names: List of tool names to execute in order
            tool_params: Parameters for each tool
            gifts: Optional gift items
            timeout: Timeout for each tool
            
        Returns:
            Tuple of (results dict, execution order list)
        """
        results = {}
        execution_order = []
        
        for tool_name in tool_names:
            params = tool_params.get(tool_name, {})
            # Add gifts to params if provided
            if gifts is not None:
                params = {**params, "gifts": gifts}
            
            try:
                result = await self._execute_single_tool(tool_name, params, timeout)
                results[tool_name] = result
                execution_order.append(tool_name)
                
                # Log if tool failed but continue with next tool
                if not result.get("success", False):
                    logger.warning(
                        f"Tool {tool_name} failed but continuing: {result.get('error')}"
                    )
            
            except Exception as e:
                logger.error(f"Tool {tool_name} raised exception: {e}")
                results[tool_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": 0.0
                }
                execution_order.append(tool_name)
                # Continue with next tool despite error
        
        return results, execution_order
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """
        Execute a single tool with timeout.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with execution result
        """
        logger.debug(f"Executing tool: {tool_name} with timeout {timeout}s")
        
        try:
            # Run tool execution in thread pool with timeout
            loop = asyncio.get_event_loop()
            tool_call = ToolCall(tool_name=tool_name, parameters=params)
            
            # Execute with timeout
            result_call = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor,
                    self.tool_registry.call_tool,
                    tool_call
                ),
                timeout=timeout
            )
            
            # Check if tool execution was successful
            if result_call.success:
                logger.debug(
                    f"Tool {tool_name} completed successfully in "
                    f"{result_call.execution_time:.2f}s"
                )
                return {
                    "success": True,
                    "result": result_call.result,
                    "execution_time": result_call.execution_time
                }
            else:
                logger.warning(
                    f"Tool {tool_name} failed: {result_call.error_message}"
                )
                self.execution_stats["failed_executions"] += 1
                return {
                    "success": False,
                    "error": result_call.error_message,
                    "execution_time": result_call.execution_time
                }
        
        except asyncio.TimeoutError:
            logger.warning(f"Tool {tool_name} timed out after {timeout}s")
            self.execution_stats["timeout_executions"] += 1
            return {
                "success": False,
                "error": f"Tool execution timed out after {timeout}s",
                "execution_time": timeout,
                "timeout": True
            }
        
        except Exception as e:
            logger.error(f"Unexpected error executing tool {tool_name}: {e}")
            self.execution_stats["failed_executions"] += 1
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "execution_time": 0.0
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        total = self.execution_stats["total_executions"]
        
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / total
                if total > 0 else 0.0
            ),
            "average_execution_time": (
                self.execution_stats["total_execution_time"] / total
                if total > 0 else 0.0
            )
        }
    
    def reset_statistics(self):
        """Reset execution statistics."""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "total_execution_time": 0.0
        }
        logger.info("Execution statistics reset")
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
        logger.info("ToolOrchestrationService executor shutdown")
