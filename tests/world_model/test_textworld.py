from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import types

import pytest

# The research package eagerly imports torch in vpsc/__init__.py.  These
# adapter contract tests remain runnable in a minimal data-extraction runtime.
if importlib.util.find_spec("torch") is None:
    package = types.ModuleType("vpsc")
    package.__path__ = [str(Path(__file__).parents[2] / "vpsc")]
    sys.modules.setdefault("vpsc", package)

from vpsc.world_model.textworld import (
    TextWorldAdapter,
    TextWorldProtocolError,
    TextWorldSetupError,
    open_textworld,
    transition_from_textworld_step,
)


def _state(feedback, commands, **extra):
    state = {
        "feedback": feedback,
        "admissible_commands": commands,
        "objective": "put the key in the box",
        "score": extra.pop("score", 0),
        "won": extra.pop("won", False),
        "lost": False,
    }
    state.update(extra)
    return state


class FakeTextWorldEnv:
    """Contract fake only; it is never exposed as a backend fallback."""

    def __init__(self):
        self.position = 0
        self.closed = False

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.position = 0
        return _state("You see a key.", ("take key", "look"))

    def step(self, command):
        if command == "take key":
            self.position = 1
            return _state("Taken.", ("inventory",), score=1, won=True), 1, True
        return _state("You see a key.", ("take key", "look")), 0, False

    def copy(self):
        return deepcopy(self)

    def close(self):
        self.closed = True


def test_counterfactual_uses_copy_without_mutating_live_episode():
    adapter = TextWorldAdapter(FakeTextWorldEnv(), source="fake-contract")
    assert adapter.reset(seed=7) == "You see a key."

    candidate = adapter.counterfactual("take key")

    assert candidate.done
    assert candidate.info["counterfactual"] is True
    assert adapter.env.position == 0
    assert adapter.transitions == []

    actual = adapter.step("take key")
    episode = adapter.episode()
    assert actual.done
    assert episode.done
    assert episode.return_ == 1.0
    assert [event.channel for event in episode.to_token_events()] == [
        "goal",
        "observation",
        "action",
        "observation",
        "reward",
        "terminal",
    ]


def test_textworld_converter_rejects_gym_style_step_result():
    with pytest.raises(TextWorldProtocolError, match="state, reward, done"):
        transition_from_textworld_step("obs", "look", ("obs", 0, False, {}), 0)


def test_open_textworld_never_falls_back_for_missing_game(tmp_path):
    with pytest.raises(TextWorldSetupError, match="no synthetic fallback"):
        open_textworld(tmp_path / "missing.z8")
