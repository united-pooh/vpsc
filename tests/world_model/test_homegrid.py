import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

# Avoid importing the unrelated torch-dependent VPSC training stack when the
# extractor is tested in a lightweight environment-only Python installation.
if importlib.util.find_spec("torch") is None:
    package = types.ModuleType("vpsc")
    package.__path__ = [str(Path(__file__).parents[2] / "vpsc")]
    sys.modules.setdefault("vpsc", package)

from vpsc.world_model.homegrid import (
    HOMEGRID_SEED_MODE_COMPAT,
    HOMEGRID_SEED_MODE_GYM,
    HomeGridRecorder,
    HomeGridSetupError,
    MessengerRecorder,
    MessengerSetupError,
    TrajectoryFormatError,
    environment_events,
    make_homegrid_env,
    make_messenger_env,
    parse_homegrid_reset,
    parse_messenger_reset,
)


class FakeHomeGridEnv:
    task = "get the plates"

    def reset(self, seed=None):
        return (
            {
                "image": [[[0, 0, 0]]],
                "token": 101,
                "log_language_info": "get the plates",
                "is_read_step": False,
            },
            {"symbolic_state": {"agent": {"pos": (0, 0)}}},
        )

    def step(self, action):
        return (
            {
                "image": [[[1, 1, 1]]],
                "token": 102,
                "log_language_info": "the plates are in the kitchen",
                "is_read_step": True,
            },
            1.0,
            True,
            False,
            {
                "events": [
                    {
                        "type": "future",
                        "description": "the plates are in the kitchen",
                    }
                ],
                "symbolic_state": {"agent": {"pos": (1, 0)}},
            },
        )


class _SeededSpace:
    def __init__(self):
        self.last_seed = None
        self._rng = np.random.default_rng()

    def seed(self, seed):
        self.last_seed = seed
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return int(self._rng.integers(0, 1_000_000))


class _OfficialHomeGridBase:
    task = "get the plates"

    def __init__(self):
        self._np_random = None
        self.action_space = _SeededSpace()
        self.observation_space = _SeededSpace()

    @property
    def unwrapped(self):
        return self

    def reset(self):
        if self._np_random is None:
            raise RuntimeError("_np_random was not compatibility-seeded")
        import random

        draws = (
            random.random(),
            float(np.random.random()),
            float(self._np_random.random()),
            self.action_space.sample(),
            self.observation_space.sample(),
        )
        return {"image": [[[0, 0, 0]]], "draws": draws}, {"official": True}

    def step(self, action):
        return {"image": [[[0, 0, 0]]]}, 0.0, False, False, {}


class HomeGrid:
    """Same wrapper shape and reset signature as homegrid==0.1.1."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


# The production guard intentionally checks this exact official identity.
HomeGrid.__module__ = "homegrid"


class _GymOrderWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


def _official_homegrid_mock():
    return _GymOrderWrapper(HomeGrid(_OfficialHomeGridBase()))


class _DirectSeedHomeGrid:
    task = "get the plates"

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        return {
            "image": [[[0, 0, 0]]],
            "draws": (float(rng.random()), int(rng.integers(0, 1_000_000))),
        }, {}

    def step(self, action):
        return {"image": [[[0, 0, 0]]]}, 0.0, False, False, {}


class FakeMessengerEnv:
    def reset(self):
        return (
            {"entities": [[[0]]], "avatar": [[[15]]]},
            ["the mage carries the message", "avoid the bird"],
        )

    def step(self, action):
        return {"entities": [[[2]]], "avatar": [[[15]]]}, 0.0, False, {}


def test_homegrid_five_tuple_becomes_multimodal_episode():
    recorder = HomeGridRecorder(FakeHomeGridEnv())
    initial = recorder.reset(seed=3)
    transition = recorder.step(2)
    episode = recorder.episode()

    assert initial["token"] == 101
    assert transition.done
    assert transition.info["action_name"] == "up"
    assert episode.goal == "get the plates"
    observation_events = [
        event for event in episode.to_token_events() if event.channel == "observation"
    ]
    assert [event.token_id for event in observation_events] == [101, 102]
    assert any(
        event.channel == "environment_event" for event in episode.to_token_events()
    )
    assert environment_events(transition)[0].text == "the plates are in the kitchen"


def test_official_homegrid_reset_signature_uses_audited_compat_seed():
    env = _official_homegrid_mock()
    recorder = HomeGridRecorder(env)

    observation = recorder.reset(seed=123)
    episode = recorder.episode()

    assert len(observation["draws"]) == 5
    assert env.action_space.last_seed == 123
    assert env.observation_space.last_seed == 123
    assert recorder.reset_info["seed"] == 123
    assert recorder.reset_info["seed_mode"] == HOMEGRID_SEED_MODE_COMPAT
    assert episode.metadata["seed"] == 123
    assert episode.metadata["seed_mode"] == HOMEGRID_SEED_MODE_COMPAT


@pytest.mark.parametrize(
    ("factory", "expected_mode"),
    [
        (_official_homegrid_mock, HOMEGRID_SEED_MODE_COMPAT),
        (_DirectSeedHomeGrid, HOMEGRID_SEED_MODE_GYM),
    ],
)
def test_same_seed_is_deterministic_for_both_homegrid_seed_paths(
    factory, expected_mode
):
    first = HomeGridRecorder(factory())
    second = HomeGridRecorder(factory())

    first_observation = first.reset(seed=991)
    second_observation = second.reset(seed=991)

    assert first_observation["draws"] == second_observation["draws"]
    assert first.seed_mode == second.seed_mode == expected_mode


def test_arbitrary_reset_typeerror_is_not_retried_without_seed():
    class BrokenWrapper:
        def __init__(self):
            self.calls = 0

        def reset(self, seed=None):
            self.calls += 1
            raise TypeError("internal reset bug")

    env = BrokenWrapper()
    recorder = HomeGridRecorder(env)
    with pytest.raises(TrajectoryFormatError, match="was not swallowed"):
        recorder.reset(seed=5)
    assert env.calls == 1


def test_nonofficial_seed_signature_error_does_not_trigger_compat_fallback():
    class NonOfficialWrapper:
        def __init__(self):
            self.calls = []

        def reset(self, **kwargs):
            self.calls.append(kwargs)
            if "seed" in kwargs:
                raise TypeError(
                    "NonOfficialWrapper.reset() got an unexpected keyword argument 'seed'"
                )
            return {"image": []}, {}

    env = NonOfficialWrapper()
    recorder = HomeGridRecorder(env)
    with pytest.raises(TrajectoryFormatError, match="was not swallowed"):
        recorder.reset(seed=5)
    assert env.calls == [{"seed": 5}]


def test_messenger_legacy_episode_preserves_manual_and_explicit_time_limit():
    recorder = MessengerRecorder(FakeMessengerEnv(), max_steps=1)
    recorder.reset()
    transition = recorder.step(4)
    episode = recorder.episode()

    assert transition.truncated
    assert not transition.terminated
    assert transition.info["action_name"] == "stay"
    assert transition.info["legacy_gym_api"] is True
    assert "mage carries the message" in episode.goal


def test_reset_parsers_reject_crossed_api_shapes():
    with pytest.raises(TrajectoryFormatError, match="HomeGrid reset info"):
        parse_homegrid_reset(({"image": []}, ["manual sentence"]))
    with pytest.raises(TrajectoryFormatError, match="manual"):
        parse_messenger_reset(({"entities": {}, "avatar": {}}, {"info": True}))


def test_unknown_official_ids_fail_before_dependency_loading():
    with pytest.raises(HomeGridSetupError, match="official ids"):
        make_homegrid_env("homegrid-invented")
    with pytest.raises(MessengerSetupError, match="official ids"):
        make_messenger_env("msgr-invented-v9")
