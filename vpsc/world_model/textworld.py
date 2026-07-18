"""Adapters for the official Microsoft TextWorld environment.

The dependency is intentionally optional and imported only by
``open_textworld``.  This module never substitutes a toy environment when the
real backend is missing.  TextWorld 1.7.0 supports Linux and macOS; its
official README recommends Docker on native Windows.

Official API references:
  * https://github.com/microsoft/TextWorld
  * ``textworld.Environment.step`` returns ``(state, reward, done)``.
  * ``textworld.Environment.copy`` clones the current interpreter state and is
    therefore the basis of the counterfactual action API below.

The generic event/transition dataclasses live here temporarily so the
TextWorld and HomeGrid/Messenger adapters can exchange exactly the same
records without introducing another file during the parallel implementation.
They can later be re-exported from a shared ``world_model`` types module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
import platform
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TEXTWORLD_PROJECT_URL = "https://github.com/microsoft/TextWorld"
TEXTWORLD_INSTALL = "python -m pip install textworld"
TEXTWORLD_WINDOWS_DOCKER = (
    "docker pull marccote19/textworld && "
    "docker run -it --rm marccote19/textworld"
)


class TextWorldSetupError(RuntimeError):
    """Raised when the official TextWorld runtime cannot be used."""


class TextWorldProtocolError(RuntimeError):
    """Raised when an object does not implement the official core API."""


class TextWorldCounterfactualError(RuntimeError):
    """Raised when a backend cannot clone its current interpreter state."""


@dataclass(frozen=True)
class DependencyProbe:
    """Non-mutating availability report for an optional environment backend."""

    name: str
    installed: bool
    version: Optional[str]
    platform_supported: bool
    detail: str
    install_command: str


@dataclass(frozen=True)
class TokenEvent:
    """One time-aligned item in a multimodal environment stream.

    ``text`` and ``token_id`` are explicit so language-model datasets do not
    have to infer language from an opaque payload.  ``payload`` preserves the
    original observation (including image arrays) for multimodal encoders.
    """

    step: int
    channel: str
    text: Optional[str] = None
    token_id: Optional[int] = None
    value: Any = None
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldTransition:
    """Action-conditioned transition shared by all official adapters."""

    step: int
    observation: Any
    action: Any
    next_observation: Any
    reward: float
    terminated: bool
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)

    def to_token_events(self) -> Tuple[TokenEvent, ...]:
        """Convert the transition to an ordered, time-aligned event stream."""

        events: List[TokenEvent] = [
            TokenEvent(
                step=self.step,
                channel="action",
                text=str(self.action),
                payload=self.action,
            ),
            _observation_event(self.step + 1, self.next_observation),
        ]
        raw_events = self.info.get("events", ())
        if isinstance(raw_events, (tuple, list)):
            for event in raw_events:
                if isinstance(event, Mapping):
                    description = event.get("description")
                    events.append(
                        TokenEvent(
                            step=self.step + 1,
                            channel="environment_event",
                            text=(
                                description
                                if isinstance(description, str)
                                else None
                            ),
                            payload=event,
                            metadata={"type": event.get("type", "unknown")},
                        )
                    )
        events.append(
            TokenEvent(
                step=self.step + 1,
                channel="reward",
                value=float(self.reward),
            )
        )
        if self.done:
            events.append(
                TokenEvent(
                    step=self.step + 1,
                    channel="terminal",
                    value=True,
                    metadata={
                        "terminated": bool(self.terminated),
                        "truncated": bool(self.truncated),
                    },
                )
            )
        return tuple(events)


@dataclass(frozen=True)
class WorldEpisode:
    """A complete or deliberately truncated official-environment rollout."""

    source: str
    initial_observation: Any
    transitions: Tuple[WorldTransition, ...]
    goal: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return bool(self.transitions and self.transitions[-1].done)

    @property
    def return_(self) -> float:
        return float(sum(transition.reward for transition in self.transitions))

    def to_token_events(self) -> Tuple[TokenEvent, ...]:
        events: List[TokenEvent] = []
        if self.goal:
            events.append(TokenEvent(step=0, channel="goal", text=self.goal))
        events.append(_observation_event(0, self.initial_observation))
        for transition in self.transitions:
            events.extend(transition.to_token_events())
        return tuple(events)


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _observation_text(observation: Any) -> Optional[str]:
    if isinstance(observation, str):
        return observation
    if isinstance(observation, Mapping):
        for key in (
            "log_language_info",
            "text",
            "feedback",
            "description",
            "observation",
        ):
            value = observation.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _observation_token(observation: Any) -> Optional[int]:
    if not isinstance(observation, Mapping) or "token" not in observation:
        return None
    try:
        return int(observation["token"])
    except (TypeError, ValueError, OverflowError):
        return None


def _observation_event(step: int, observation: Any) -> TokenEvent:
    metadata: Dict[str, Any] = {}
    if isinstance(observation, Mapping) and "is_read_step" in observation:
        metadata["is_read_step"] = bool(observation["is_read_step"])
    return TokenEvent(
        step=step,
        channel="observation",
        text=_observation_text(observation),
        token_id=_observation_token(observation),
        payload=observation,
        metadata=metadata,
    )


def probe_textworld() -> DependencyProbe:
    """Inspect TextWorld without importing it or starting an interpreter."""

    try:
        installed = find_spec("textworld") is not None
    except (ImportError, ValueError):
        installed = False

    try:
        version = importlib_metadata.version("textworld") if installed else None
    except importlib_metadata.PackageNotFoundError:
        version = None

    is_windows = platform.system().lower() == "windows"
    if is_windows:
        detail = (
            "TextWorld is not officially supported on native Windows. "
            "Use its official Docker image (or a Linux/macOS environment)."
        )
    elif not installed:
        detail = "The optional 'textworld' package is not installed."
    else:
        detail = "The package is present; open a real game file to verify its interpreter."

    return DependencyProbe(
        name="textworld",
        installed=installed,
        version=version,
        platform_supported=not is_windows,
        detail=detail,
        install_command=(
            TEXTWORLD_WINDOWS_DOCKER if is_windows else TEXTWORLD_INSTALL
        ),
    )


def _load_textworld(allow_unsupported_platform: bool = False) -> Any:
    probe = probe_textworld()
    if not probe.platform_supported and not allow_unsupported_platform:
        raise TextWorldSetupError(
            f"{probe.detail} Official workaround: `{probe.install_command}`. "
            f"See {TEXTWORLD_PROJECT_URL}. Set allow_unsupported_platform=True "
            "only to test an already-provisioned native installation."
        )
    try:
        import textworld  # type: ignore
    except Exception as exc:
        raise TextWorldSetupError(
            "Unable to import the official TextWorld package. Install it with "
            f"`{TEXTWORLD_INSTALL}` on Linux/macOS. On Windows use "
            f"`{TEXTWORLD_WINDOWS_DOCKER}`. Original error: {exc}"
        ) from exc
    return textworld


def _state_observation(state: Any) -> str:
    feedback = _mapping_value(state, "feedback")
    if isinstance(feedback, str):
        return feedback
    description = _mapping_value(state, "description")
    if isinstance(description, str):
        return description
    raise TextWorldProtocolError(
        "TextWorld state has neither string 'feedback' nor 'description'. "
        "Create the environment with EnvInfos(feedback=True)."
    )


def _state_info(state: Any) -> Dict[str, Any]:
    admissible = _mapping_value(state, "admissible_commands", ()) or ()
    info: Dict[str, Any] = {
        "admissible_commands": tuple(str(command) for command in admissible),
    }
    for key in (
        "objective",
        "description",
        "inventory",
        "location",
        "won",
        "lost",
        "score",
        "moves",
        "max_score",
        "last_command",
    ):
        value = _mapping_value(state, key)
        if value is not None:
            info[key] = value
    facts = _mapping_value(state, "facts")
    if facts is not None:
        info["facts"] = tuple(repr(fact) for fact in facts)
    return info


def transition_from_textworld_step(
    observation: str,
    action: str,
    result: Sequence[Any],
    step: int,
) -> Tuple[WorldTransition, Any]:
    """Validate and convert the official three-value TextWorld step result."""

    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise TextWorldProtocolError(
            "TextWorld core step() must return (state, reward, done); "
            f"received {type(result).__name__} with length "
            f"{len(result) if isinstance(result, (tuple, list)) else 'unknown'}."
        )
    state, reward, done = result
    transition = WorldTransition(
        step=int(step),
        observation=observation,
        action=str(action),
        next_observation=_state_observation(state),
        reward=float(reward),
        terminated=bool(done),
        truncated=False,
        info=_state_info(state),
    )
    return transition, state


class TextWorldAdapter:
    """Record real TextWorld transitions and branch counterfactual actions."""

    def __init__(self, env: Any, source: str = "textworld") -> None:
        for name in ("reset", "step", "copy", "close"):
            if not callable(getattr(env, name, None)):
                raise TextWorldProtocolError(
                    f"Environment is missing callable {name}(); use the core "
                    "TextWorld Environment API rather than a Gym wrapper."
                )
        self.env = env
        self.source = source
        self.state: Any = None
        self.initial_observation: Optional[str] = None
        self.objective: Optional[str] = None
        self.transitions: List[WorldTransition] = []

    @property
    def admissible_actions(self) -> Tuple[str, ...]:
        if self.state is None:
            return ()
        commands = _mapping_value(self.state, "admissible_commands", ()) or ()
        return tuple(str(command) for command in commands)

    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            seed_fn = getattr(self.env, "seed", None)
            if not callable(seed_fn):
                raise TextWorldProtocolError(
                    "A seed was requested but this TextWorld backend has no seed()."
                )
            seed_fn(seed)
        state = self.env.reset()
        observation = _state_observation(state)
        self.state = state
        self.initial_observation = observation
        objective = _mapping_value(state, "objective")
        self.objective = str(objective) if objective is not None else None
        self.transitions.clear()
        return observation

    def _require_running(self) -> None:
        if self.state is None or self.initial_observation is None:
            raise TextWorldProtocolError("Call reset() before step or counterfactual.")
        if self.transitions and self.transitions[-1].done:
            raise TextWorldProtocolError("The episode has already terminated.")

    def step(self, action: str) -> WorldTransition:
        self._require_running()
        if not isinstance(action, str) or not action.strip():
            raise TextWorldProtocolError("TextWorld actions must be non-empty strings.")
        observation = _state_observation(self.state)
        transition, state = transition_from_textworld_step(
            observation,
            action,
            self.env.step(action),
            len(self.transitions),
        )
        self.state = state
        self.transitions.append(transition)
        return transition

    def counterfactual(self, action: str) -> WorldTransition:
        """Evaluate one action on a cloned interpreter without changing history."""

        self._require_running()
        if not isinstance(action, str) or not action.strip():
            raise TextWorldProtocolError("TextWorld actions must be non-empty strings.")
        try:
            branch = self.env.copy()
        except Exception as exc:
            raise TextWorldCounterfactualError(
                "This TextWorld backend failed Environment.copy(); counterfactual "
                "evaluation is unavailable and no replay fallback was attempted."
            ) from exc
        if branch is self.env:
            raise TextWorldCounterfactualError(
                "Environment.copy() returned the live environment; refusing to mutate it."
            )
        try:
            transition, _ = transition_from_textworld_step(
                _state_observation(self.state),
                action,
                branch.step(action),
                len(self.transitions),
            )
            info = dict(transition.info)
            info["counterfactual"] = True
            return replace(transition, info=info)
        except TextWorldProtocolError:
            raise
        except Exception as exc:
            raise TextWorldCounterfactualError(
                f"Counterfactual action {action!r} failed on the cloned environment."
            ) from exc
        finally:
            close = getattr(branch, "close", None)
            if callable(close):
                close()

    def counterfactual_candidates(
        self, actions: Optional[Iterable[str]] = None
    ) -> Tuple[WorldTransition, ...]:
        candidates = self.admissible_actions if actions is None else tuple(actions)
        if not candidates:
            raise TextWorldCounterfactualError(
                "No candidate actions were supplied and state.admissible_commands is empty."
            )
        return tuple(self.counterfactual(action) for action in candidates)

    def episode(self) -> WorldEpisode:
        if self.initial_observation is None:
            raise TextWorldProtocolError("Call reset() before requesting an episode.")
        return WorldEpisode(
            source=self.source,
            initial_observation=self.initial_observation,
            transitions=tuple(self.transitions),
            goal=self.objective,
            metadata={"backend": "textworld"},
        )

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "TextWorldAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def open_textworld(
    game_file: Any,
    include_privileged_facts: bool = False,
    allow_unsupported_platform: bool = False,
    extras: Optional[Iterable[str]] = None,
) -> TextWorldAdapter:
    """Open a compiled real TextWorld game with the core cloneable API."""

    path = Path(game_file).expanduser()
    if not path.is_file():
        raise TextWorldSetupError(
            f"TextWorld game file does not exist: {path}. Generate a real game "
            "with `tw-make ... --output <game>.z8`; no synthetic fallback was used."
        )
    textworld = _load_textworld(allow_unsupported_platform)
    request_infos = textworld.EnvInfos(
        feedback=True,
        description=True,
        inventory=True,
        objective=True,
        won=True,
        lost=True,
        score=True,
        moves=True,
        max_score=True,
        admissible_commands=True,
        facts=bool(include_privileged_facts),
        extras=list(extras or ()),
    )
    try:
        env = textworld.start(str(path.resolve()), request_infos=request_infos)
    except Exception as exc:
        raise TextWorldSetupError(
            f"The official TextWorld interpreter could not open {path}: {exc}. "
            "Check the compiled game extension and native interpreter dependencies."
        ) from exc
    return TextWorldAdapter(env, source=str(path.resolve()))


def record_textworld_actions(
    adapter: TextWorldAdapter,
    actions: Iterable[str],
    seed: Optional[int] = None,
) -> WorldEpisode:
    """Reset an adapter and convert a finite real action sequence to an episode."""

    adapter.reset(seed=seed)
    for action in actions:
        transition = adapter.step(action)
        if transition.done:
            break
    return adapter.episode()


__all__ = [
    "DependencyProbe",
    "TokenEvent",
    "WorldEpisode",
    "WorldTransition",
    "TextWorldAdapter",
    "TextWorldCounterfactualError",
    "TextWorldProtocolError",
    "TextWorldSetupError",
    "open_textworld",
    "probe_textworld",
    "record_textworld_actions",
    "transition_from_textworld_step",
]
