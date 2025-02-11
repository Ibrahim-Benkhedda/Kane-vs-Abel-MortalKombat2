import unittest
from mk_ai.agents.BT.nodes import Action, Condition, Selector, Sequence, Inverter, NodeStatus

class TestActionNode(unittest.TestCase):
    def test_single_frame_action(self):
        """An action requiring a single tick should run then succeed."""
        # Create an Action that needs only one tick
        action = Action("SingleFrameAction", action_id=101, frames_needed=1)
        context = {}

        # First tick: Should return RUNNING.
        self.assertEqual(action.tick(context), NodeStatus.RUNNING)
        self.assertEqual(action.get_action_id(), 101)
        # Second tick: Should complete and return SUCCESS, and reset.
        self.assertEqual(action.tick(context), NodeStatus.SUCCESS)
        self.assertEqual(action.get_action_id(), 101)

    def test_multi_frame_action(self):
        """An action with multiple frames remains running until complete."""
        frames_needed = 3
        action = Action("MultiFrameAction", action_id=102, frames_needed=frames_needed)
        context = {}
        
        # Tick for the number of frames required.
        for tick in range(frames_needed):
            self.assertEqual(action.tick(context), NodeStatus.RUNNING, f"Tick {tick+1} should be RUNNING")
            self.assertEqual(action.get_action_id(), 102)
        # Next tick should yield SUCCESS.
        self.assertEqual(action.tick(context), NodeStatus.SUCCESS)

class TestConditionNode(unittest.TestCase):
    def test_condition_true(self):
        """A condition that evaluates to True should return SUCCESS."""
        condition = Condition("AlwaysTrue", condition=lambda ctx: True)
        self.assertEqual(condition.tick({}), NodeStatus.SUCCESS)

    def test_condition_false(self):
        """A condition that evaluates to False should return FAILURE."""
        condition = Condition("AlwaysFalse", condition=lambda ctx: False)
        self.assertEqual(condition.tick({}), NodeStatus.FAILURE)

class TestSelectorNode(unittest.TestCase):
    def test_selector_first_success(self):
        """
        A selector should return SUCCESS immediately if its first child succeeds.
        """
        # Create an action that returns SUCCESS immediately.
        action = Action("ImmediateSuccess", action_id=201, frames_needed=0)
        # For immediate success, override tick for testing:
        action.tick = lambda ctx: NodeStatus.SUCCESS
        
        # Second child will not be executed.
        condition_fail = Condition("NeverTrue", condition=lambda ctx: False)
        selector = Selector("TestSelector", [action, condition_fail])
        context = {}
        
        self.assertEqual(selector.tick(context), NodeStatus.SUCCESS)
        self.assertEqual(selector.get_action_id(), action.get_action_id())

    def test_selector_runs_next_child(self):
        """
        If the first child fails, the selector should process the next child.
        """
        # First child always fails.
        condition_fail = Condition("AlwaysFail", condition=lambda ctx: False)
        # Second child is an action that takes one tick.
        action = Action("DelayedAction", action_id=202, frames_needed=1)
        selector = Selector("TestSelector2", [condition_fail, action])
        context = {}

        # Tick 1: first child fails, second child runs.
        status = selector.tick(context)
        self.assertEqual(status, NodeStatus.RUNNING)
        self.assertEqual(selector.get_action_id(), 202)

        # Tick 2: action should complete.
        self.assertEqual(selector.tick(context), NodeStatus.SUCCESS)

    def test_empty_selector(self):
        """A selector with no children should return FAILURE (or a defined default)."""
        selector = Selector("EmptySelector", [])
        self.assertEqual(selector.tick({}), NodeStatus.FAILURE)
        self.assertIsNone(selector.get_action_id())

    def test_selector_all_fail(self):
        """
        If all children fail, the selector should return FAILURE.
        """
        condition1 = Condition("Fail1", condition=lambda ctx: False)
        condition2 = Condition("Fail2", condition=lambda ctx: False)
        selector = Selector("SelectorAllFail", [condition1, condition2])
        self.assertEqual(selector.tick({}), NodeStatus.FAILURE)

class TestSequenceNode(unittest.TestCase):
    def test_sequence_all_success(self):
        """
        A sequence with all children succeeding should eventually return SUCCESS.
        """
        # Two actions that need one tick each.
        action1 = Action("Action1", action_id=301, frames_needed=1)
        action2 = Action("Action2", action_id=302, frames_needed=1)
        seq = Sequence("SequenceSuccess", [action1, action2])
        context = {}

        # Tick 1: First action is running.
        self.assertEqual(seq.tick(context), NodeStatus.RUNNING)
        self.assertEqual(seq.get_action_id(), 301)

        # Tick 2: First action completes, moving to second action.
        self.assertEqual(seq.tick(context), NodeStatus.RUNNING)
        self.assertEqual(seq.get_action_id(), 302)

        # Tick 3: Second action completes; sequence returns SUCCESS.
        self.assertEqual(seq.tick(context), NodeStatus.SUCCESS)

    def test_sequence_reset_after_failure(self):
        """Test that a sequence resets after a failure so that a new evaluation starts fresh."""
        action = Action("Action", action_id=303, frames_needed=1)
        condition_fail = Condition("AlwaysFail", condition=lambda ctx: False)
        seq = Sequence("SequenceReset", [action, condition_fail])
        context = {}

        # First tick: action is RUNNING.
        self.assertEqual(seq.tick(context), NodeStatus.RUNNING)
        # Second tick: action completes then condition fails, sequence returns FAILURE.
        self.assertEqual(seq.tick(context), NodeStatus.FAILURE)
        # After failure, the sequence should have reset.
        self.assertEqual(seq.current_child_idx, 0)
        # A subsequent evaluation should start with the first child again.
        self.assertEqual(seq.tick(context), NodeStatus.RUNNING)

    def test_sequence_failure(self):
        """
        A sequence should return FAILURE immediately if any child fails.
        """
        # First child: action that will succeed.
        action = Action("Action", action_id=303, frames_needed=1)
        # Second child: condition that always fails.
        condition_fail = Condition("AlwaysFail", condition=lambda ctx: False)
        seq = Sequence("SequenceFailure", [action, condition_fail])
        context = {}

        # Tick 1: action is running.
        self.assertEqual(seq.tick(context), NodeStatus.RUNNING)
        self.assertEqual(seq.get_action_id(), 303)
        # Tick 2: action completes then condition fails => sequence returns FAILURE.
        self.assertEqual(seq.tick(context), NodeStatus.FAILURE)
        # Ensure sequence resets properly.
        self.assertEqual(seq.current_child_idx, 0)

class TestInverterNode(unittest.TestCase):
    def test_inverter_success_to_failure(self):
        """
        Inverter should flip SUCCESS into FAILURE.
        """
        condition_success = Condition("AlwaysTrue", condition=lambda ctx: True)
        inverter = Inverter("InverterTest", condition_success)
        self.assertEqual(inverter.tick({}), NodeStatus.FAILURE)
        self.assertEqual(inverter.get_action_id(), condition_success.get_action_id())

    def test_inverter_failure_to_success(self):
        """
        Inverter should flip FAILURE into SUCCESS.
        """
        condition_fail = Condition("AlwaysFail", condition=lambda ctx: False)
        inverter = Inverter("InverterTest2", condition_fail)
        self.assertEqual(inverter.tick({}), NodeStatus.SUCCESS)
        self.assertEqual(inverter.get_action_id(), condition_fail.get_action_id())

    def test_inverter_propagates_running(self):
        """
        Inverter should propagate the RUNNING state unchanged.
        """
        # Use an action that requires 2 ticks so it is RUNNING on the first tick.
        action = Action("LongAction", action_id=304, frames_needed=2)
        inverter = Inverter("InverterRunning", action)
        context = {}
        self.assertEqual(inverter.tick(context), NodeStatus.RUNNING)

class TestNestedBehaviorTree(unittest.TestCase):
    def test_complex_nested_tree(self):
        """
        Test a complex tree combining Sequence, Selector, Condition, and Action:
        
        Tree Structure:
            Sequence
             ├── Condition (always true)
             └── Selector
                  ├── Condition (always false)
                  └── Action (takes 2 ticks)

        Expected Flow:
          - Tick 1: 
              * Sequence: First child (always true) succeeds.
              * Then, Selector: first child fails, second child (action) returns RUNNING.
          - Tick 2:
              * Action remains RUNNING.
          - Tick 3:
              * Action completes; Selector returns SUCCESS.
              * Sequence then returns SUCCESS.
        """
        condition_true = Condition("AlwaysTrue", condition=lambda ctx: True)
        condition_false = Condition("AlwaysFalse", condition=lambda ctx: False)
        action = Action("DelayedAction", action_id=305, frames_needed=2)
        selector = Selector("NestedSelector", [condition_false, action])
        sequence = Sequence("NestedSequence", [condition_true, selector])
        context = {}

        # Tick 1: Condition succeeds, selector starts the action.
        status = sequence.tick(context)
        self.assertEqual(status, NodeStatus.RUNNING)
        self.assertEqual(sequence.get_action_id(), 305)

        # Tick 2: Action is still running.
        status = sequence.tick(context)
        self.assertEqual(status, NodeStatus.RUNNING)
        self.assertEqual(sequence.get_action_id(), 305)

        # Tick 3: Action completes; selector and sequence succeed.
        status = sequence.tick(context)
        self.assertEqual(status, NodeStatus.SUCCESS)

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

