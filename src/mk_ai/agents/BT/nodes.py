from typing import List, Callable, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum

# =======================================================
# ENUM & ABSTRACT BASE NODE
# =======================================================

class NodeStatus(Enum):
    """
    Enum representing possible statuses for a Behavior Tree node.
    
    Attributes:
        SUCCESS: The node completed its task successfully.
        FAILURE: The node failed to complete its task.
        RUNNING: The node is still in progress.
    """
    SUCCESS = 1
    FAILURE = 2 
    RUNNING = 3


class Node(ABC):
    """
    Abstract base class for all Behavior Tree nodes.
    
    This class defines the basic interface that every node (leaf or composite)
    must implement:
      - tick(): Evaluate the node for the current tick.
      - get_action_id(): Optionally expose an action id for leaf nodes.
      - reset(): Reset the node’s internal state.
    """
    def __init__(self, name: str):
        # Store the node's name (useful for debugging and identification)
        self.name = name

    @abstractmethod
    def tick(self, context: Any) -> NodeStatus:
        """
        Evaluate the node's logic for the current tick.
        
        Parameters:
            context (Any): Shared data or blackboard information.
            
        Returns:
            NodeStatus: The status after evaluation (SUCCESS, FAILURE, or RUNNING).
        """
        raise NotImplementedError

    def get_action_id(self) -> Optional[int]:
        """
        Optionally return an action identifier associated with this node.
        
        Returns:
            Optional[int]: The action identifier if applicable; otherwise, None.
        """
        return None

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the node's internal state so that it is ready for a fresh evaluation.
        
        Composite nodes should reset their child pointers, and leaf nodes should
        reset any internal counters.
        """
        pass

# =======================================================
# LEAF NODES
# =======================================================

class Action(Node):
    """
    A leaf node representing an action that requires a specific number of ticks
    before it completes.
    
    For example, if frames_needed==1:
      - On the first tick, the action returns RUNNING.
      - On the second tick, the action resets and returns SUCCESS.
    
    Attributes:
        action_id (int): Identifier for this action (e.g., button press).
        frames_needed (int): Number of ticks required to complete the action.
        elapsed_frames (int): Counter tracking how many ticks have occurred.
    """
    def __init__(self, name: str, action_id: int, frames_needed: int = 1) -> None:
        # Initialize the base node with a name.
        super().__init__(name)
        # Set the action id used to represent this action.
        self.action_id: int = action_id
        # Define how many ticks the action requires to complete.
        self.frames_needed: int = frames_needed
        # Initialize the tick counter.
        self.elapsed_frames: int = 0

    def tick(self, context: Any) -> NodeStatus:
        """
        Tick the action node by incrementing its internal counter.
        
        If the number of ticks is less than frames_needed, the action is still running.
        Once enough ticks have passed, the action resets its counter and returns SUCCESS.
        
        Parameters:
            context (Any): The shared context (unused in this action).
            
        Returns:
            NodeStatus: RUNNING until complete, then SUCCESS.
        """
        # Check if the action still needs more ticks to complete.
        if self.elapsed_frames < self.frames_needed:
            # Increment the tick counter.
            self.elapsed_frames += 1
            # Return RUNNING indicating the action is still in progress.
            return NodeStatus.RUNNING
        else:
            # Reset the counter for repeatability.
            self.elapsed_frames = 0
            # Return SUCCESS to indicate the action has completed.
            return NodeStatus.SUCCESS

    def get_action_id(self) -> Optional[int]:
        """
        Return the action id associated with this node.
        
        Returns:
            Optional[int]: The action id.
        """
        return self.action_id

    def reset(self) -> None:
        """
        Reset the action node's internal state by setting the tick counter to zero.
        """
        self.elapsed_frames = 0


class Condition(Node):
    """
    A leaf node that evaluates a given condition (callable) against the context.
    
    Returns SUCCESS if the condition is met (True), otherwise returns FAILURE.
    """
    def __init__(self, name: str, condition: Callable[[Any], bool]) -> None:
        # Initialize the node with a name.
        super().__init__(name)
        # Store the condition callable that takes the context and returns a boolean.
        self.condition = condition

    def tick(self, context: Any) -> NodeStatus:
        """
        Evaluate the condition using the provided context.
        
        Parameters:
            context (Any): Shared data for evaluating the condition.
            
        Returns:
            NodeStatus: SUCCESS if the condition evaluates to True; otherwise, FAILURE.
        """
        # Evaluate the condition.
        result = self.condition(context)
        # Output debug information showing the condition's result.
        # print(f"[DEBUG Condition: {self.name}] {result} => {'SUCCESS' if result else 'FAILURE'}")
        # Return SUCCESS if condition is True, else FAILURE.
        return NodeStatus.SUCCESS if result else NodeStatus.FAILURE

    def get_action_id(self) -> Optional[int]:
        """
        Conditions do not have an associated action id.
        
        Returns:
            None.
        """
        return None

    def reset(self) -> None:
        """
        Reset the condition node. Since it maintains no internal state,
        no action is needed.
        """
        pass

# =======================================================
# COMPOSITE NODES
# =======================================================

class Sequence(Node):
    """
    A composite node that executes its child nodes in order (a sequence).
    
    The Sequence node "drills down" by ticking children in order:
      - If a child returns SUCCESS immediately, it advances to the next child.
      - If a child returns RUNNING, the sequence stops and returns RUNNING.
      - If a child returns FAILURE, the sequence resets and returns FAILURE.
      
    Example:
      For two Action nodes (each with frames_needed==1):
        - Tick 1: First action returns RUNNING → Sequence returns RUNNING.
        - Tick 2: First action returns SUCCESS; Sequence immediately ticks second child,
                  which returns RUNNING → Sequence returns RUNNING.
        - Tick 3: Second action returns SUCCESS → Sequence returns SUCCESS.
    """
    def __init__(self, name: str, children: List[Node]) -> None:
        # Initialize the base node.
        super().__init__(name)
        # Store the child nodes that form the sequence.
        self.children: List[Node] = children
        # Initialize the index of the currently active child.
        self.current_child_idx: int = 0

    def tick(self, context: Any) -> NodeStatus:
        """
        Tick the sequence node by processing its children in order.
        
        The method loops through the children starting at the current index:
          - If a child returns RUNNING, the sequence returns RUNNING.
          - If a child returns SUCCESS, the sequence immediately advances to the next child.
          - If a child returns FAILURE, the sequence resets and returns FAILURE.
        
        If all children succeed, the sequence resets and returns SUCCESS.
        
        Parameters:
            context (Any): Shared context for the evaluation.
            
        Returns:
            NodeStatus: SUCCESS if all children succeed, FAILURE on any failure, or RUNNING otherwise.
        """
        # Loop through children until we run out or hit a RUNNING status.
        while self.current_child_idx < len(self.children):
            # Tick the current child.
            status = self.children[self.current_child_idx].tick(context)
            if status == NodeStatus.RUNNING:
                # If the child is still running, return RUNNING immediately.
                return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                # If the child succeeded, advance to the next child.
                self.current_child_idx += 1
                # Continue looping to tick the next child within the same tick.
                continue
            elif status == NodeStatus.FAILURE:
                # On failure, reset the sequence and return FAILURE.
                self.reset()
                return NodeStatus.FAILURE
        # If all children have been successfully processed, reset and return SUCCESS.
        self.reset()
        return NodeStatus.SUCCESS

    def get_action_id(self) -> Optional[int]:
        """
        Return the action id from the currently active child.
        
        If the sequence is still processing a child, returns that child's action id.
        Otherwise, per test expectations, returns the first child's action id.
        
        Returns:
            Optional[int]: The active child's action id, or None if no children exist.
        """
        if self.current_child_idx < len(self.children):
            return self.children[self.current_child_idx].get_action_id()
        return self.children[0].get_action_id() if self.children else None

    def reset(self) -> None:
        """
        Reset the sequence node and all its children.
        
        This resets the active child index to 0 and calls reset() on each child.
        """
        self.current_child_idx = 0
        for child in self.children:
            child.reset()


class Selector(Node):
    """
    A composite node that selects among its children until one succeeds.
    
    The Selector node ticks its children in order:
      - If a child returns RUNNING, the Selector returns RUNNING immediately.
      - If a child returns SUCCESS, the Selector resets and returns SUCCESS.
      - If a child returns FAILURE, the Selector advances to the next child.
      
    If all children return FAILURE, the Selector resets and returns FAILURE.
    """
    def __init__(self, name: str, children: List[Node]) -> None:
        # Initialize the base node.
        super().__init__(name)
        # Store the child nodes.
        self.children: List[Node] = children
        # Initialize the index for the current child.
        self.current_child_idx: int = 0

    def tick(self, context: Any) -> NodeStatus:
        """
        Tick the selector node by evaluating its children sequentially.
        
        Parameters:
            context (Any): Shared context for evaluation.
            
        Returns:
            NodeStatus: SUCCESS if any child succeeds, RUNNING if a child is in progress,
                        or FAILURE if all children fail.
        """
        # Loop through children starting from the current index.
        while self.current_child_idx < len(self.children):
            # Tick the current child.
            status = self.children[self.current_child_idx].tick(context)
            if status == NodeStatus.RUNNING:
                # If the child is still running, return RUNNING immediately.
                return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                # If the child succeeded, reset the selector and return SUCCESS.
                self.current_child_idx = 0
                return NodeStatus.SUCCESS
            elif status == NodeStatus.FAILURE:
                # On failure, move to the next child.
                self.current_child_idx += 1
                continue
        # If no children succeeded, reset and return FAILURE.
        self.current_child_idx = 0
        return NodeStatus.FAILURE

    def get_action_id(self) -> Optional[int]:
        """
        Return the action id from the currently active child.
        
        Returns:
            Optional[int]: The action id of the active child, or None if no such child exists.
        """
        if self.children and self.current_child_idx < len(self.children):
            return self.children[self.current_child_idx].get_action_id()
        return None

    def reset(self) -> None:
        """
        Reset the selector node and all its children.
        
        This resets the active child index to 0 and calls reset() on each child.
        """
        self.current_child_idx = 0
        for child in self.children:
            child.reset()

# =======================================================
# DECORATOR NODE
# =======================================================

class Inverter(Node):
    """
    A decorator node that inverts the result of its child node.
    
    Behavior:
      - If the child returns RUNNING, Inverter returns RUNNING.
      - If the child returns SUCCESS, Inverter returns FAILURE.
      - If the child returns FAILURE, Inverter returns SUCCESS.
    """
    def __init__(self, name: str, child: Node) -> None:
        # Initialize the base node with a name.
        super().__init__(name)
        # Store the child node whose result will be inverted.
        self.child: Node = child

    def tick(self, context: Any) -> NodeStatus:
        """
        Tick the inverter node by evaluating its child and inverting the result.
        
        Parameters:
            context (Any): Shared context for evaluation.
            
        Returns:
            NodeStatus: The inverted result of the child's evaluation.
        """
        # Tick the child node.
        status = self.child.tick(context)
        # If the child is still running, propagate the RUNNING status.
        if status == NodeStatus.RUNNING:
            return NodeStatus.RUNNING
        # If the child succeeded, invert to FAILURE.
        elif status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        # Otherwise (if the child failed), invert to SUCCESS.
        else:
            return NodeStatus.SUCCESS

    def get_action_id(self) -> Optional[int]:
        """
        Return the action id from the child node.
        
        Returns:
            Optional[int]: The action id of the child.
        """
        return self.child.get_action_id()

    def reset(self) -> None:
        """
        Reset the inverter node by resetting its child.
        """
        self.child.reset()
