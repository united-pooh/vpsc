"""Official HomeGrid and Messenger environment adapters.

HomeGrid is distributed as ``homegrid==0.1.1`` and uses the Gym 0.26 API:
``reset -> (observation, info)`` and
``step -> (observation, reward, terminated, truncated, info)``.

Messenger is distributed only from its official ``messenger-emma`` repository
and uses the legacy Gym API:
``reset -> (observation, manual)`` and
``step -> (observation, reward, done, info)``.

Both dependencies are lazy.  Missing, incompatible, or unregistered official
environments produce setup errors with installation instructions.  There is no
synthetic-grid fallback in this module.

Official sources:
  * https://pypi.org/project/homegrid/
  * https://github.com/jlin816/dynalang
  * https://github.com/ahjwang/messenger-emma
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
import platform
import random
import sys
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .textworld import TokenEvent, WorldEpisode, WorldTransition


HOMEGRID_ENV_IDS = (
    "homegrid-task",
    "homegrid-future",
    "homegrid-dynamics",
    "homegrid-corrections",
)
MESSENGER_ENV_IDS = tuple(
    f"msgr-{split}-v{stage}"
    for stage in (1, 2, 3)
    for split in ("train", "train-sc", "train-mc", "val", "test")
) + ("msgr-test-se-v2",)

HOMEGRID_ACTIONS = {
    0: "left",
    1: "right",
    2: "up",
    3: "down",
    4: "pickup",
    5: "drop",
    6: "get",
    7: "pedal",
    8: "grasp",
    9: "lift",
}
MESSENGER_ACTIONS = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "stay",
}

HOMEGRID_INSTALL = "python -m pip install homegrid==0.1.1"
MESSENGER_REPOSITORY = "https://github.com/ahjwang/messenger-emma"
MESSENGER_INSTALL = (
    "git clone https://github.com/ahjwang/messenger-emma.git; "
    "python -m pip install -e messenger-emma"
)


class HomeGridSetupError(RuntimeError):
    """Raised when the official HomeGrid package cannot be used."""


class MessengerSetupError(RuntimeError):
    """Raised when the official Messenger package cannot be used."""


class TrajectoryFormatError(ValueError):
    """Raised when an environment result violates its documented API."""


@dataclass(frozen=True)
class EnvironmentProbe:
    """Availability and data-asset report without creating a fake task."""

    name: str
    package_installed: bool
    gym_installed: bool
    version: Optional[str]
    gym_version: Optional[str]
    expected_env_ids: Tuple[str, ...]
    registered_env_ids: Tuple[str, ...]
    data_assets_present: Optional[bool]
    importable: Optional[bool]
    compatibility_warnings: Tuple[str, ...]
    detail: str
    install_command: str

    @property
    def available(self) -> bool:
        import_ok = self.importable is not False
        return bool(self.package_installed and self.gym_installed and import_ok)


def _has_spec(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _package_root(name: str) -> Optional[Path]:
    try:
        spec = find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        locations = tuple(spec.submodule_search_locations)
        if locations:
            return Path(locations[0])
    if spec.origin:
        return Path(spec.origin).parent
    return None


def _registered_ids(gym: Any, expected: Sequence[str]) -> Tuple[str, ...]:
    registered: List[str] = []
    for env_id in expected:
        try:
            gym.spec(env_id)
        except Exception:
            continue
        registered.append(env_id)
    return tuple(registered)


def probe_homegrid(verify_import: bool = False) -> EnvironmentProbe:
    """Check package, bundled assets, and optionally Gym registration."""

    package_installed = _has_spec("homegrid")
    gym_installed = _has_spec("gym")
    root = _package_root("homegrid")
    assets_present = None
    if root is not None:
        assets_present = all(
            (root / relative).is_file()
            for relative in (
                "homegrid_embeds.pkl",
                "homegrid_sentences.txt",
                "layout.py",
            )
        )

    registered: Tuple[str, ...] = ()
    importable: Optional[bool] = None
    import_error: Optional[str] = None
    if verify_import and package_installed and gym_installed:
        try:
            gym = import_module("gym")
            import_module("homegrid")  # Registers the four official IDs.
            registered = _registered_ids(gym, HOMEGRID_ENV_IDS)
            importable = True
        except Exception as exc:
            importable = False
            import_error = f"{type(exc).__name__}: {exc}"

    warnings: List[str] = []
    gym_version = _version("gym")
    if gym_version is not None and not gym_version.startswith("0.26"):
        warnings.append(
            f"HomeGrid 0.1.1 pins gym==0.26, but gym {gym_version} is installed."
        )
    if sys.version_info >= (3, 12):
        warnings.append(
            "HomeGrid declares Python <4 support, but its pinned, unmaintained "
            "Gym 0.26 stack may be fragile on Python 3.12+."
        )
    warnings.append(
        "HomeGrid 0.1.1's top-level homegrid.HomeGrid.reset() does not "
        "forward seed. HomeGridRecorder uses an explicit audited compatibility "
        "path only for that wrapper and that signature error."
    )

    if not package_installed or not gym_installed:
        detail = "HomeGrid and/or its pinned Gym dependency is not installed."
    elif import_error:
        detail = f"HomeGrid is installed but import failed: {import_error}"
    elif verify_import and set(registered) != set(HOMEGRID_ENV_IDS):
        missing = tuple(sorted(set(HOMEGRID_ENV_IDS) - set(registered)))
        detail = f"HomeGrid imported, but official Gym IDs are missing: {missing}."
    elif assets_present is False:
        detail = "HomeGrid is installed but official language/data assets are incomplete."
    else:
        detail = "HomeGrid package and bundled official assets were detected."

    return EnvironmentProbe(
        name="homegrid",
        package_installed=package_installed,
        gym_installed=gym_installed,
        version=_version("homegrid"),
        gym_version=gym_version,
        expected_env_ids=HOMEGRID_ENV_IDS,
        registered_env_ids=registered,
        data_assets_present=assets_present,
        importable=importable,
        compatibility_warnings=tuple(warnings),
        detail=detail,
        install_command=HOMEGRID_INSTALL,
    )


def probe_messenger(verify_import: bool = False) -> EnvironmentProbe:
    """Check the locally installed official Messenger repository and assets."""

    package_installed = _has_spec("messenger")
    gym_installed = _has_spec("gym")
    root = _package_root("messenger")
    assets_present = None
    if root is not None:
        assets_present = all(
            (root / relative).is_file()
            for relative in (
                "envs/games.json",
                "envs/texts/text_train.json",
                "envs/texts/text_val.json",
                "envs/texts/text_test.json",
            )
        )

    registered: Tuple[str, ...] = ()
    importable: Optional[bool] = None
    import_error: Optional[str] = None
    if verify_import and package_installed and gym_installed:
        try:
            gym = import_module("gym")
            import_module("messenger")  # Registers the official IDs.
            registered = _registered_ids(gym, MESSENGER_ENV_IDS)
            importable = True
        except Exception as exc:
            importable = False
            import_error = f"{type(exc).__name__}: {exc}"

    warnings: List[str] = []
    gym_version = _version("gym")
    if platform.system().lower() == "windows":
        warnings.append(
            "Messenger's official setup documents Linux SDL system packages; "
            "native Windows is not a verified runtime."
        )
    if sys.version_info >= (3, 9):
        warnings.append(
            "The official Messenger README recommends Python 3.7 when setup fails."
        )
    if gym_version is not None:
        try:
            major_minor = tuple(int(part) for part in gym_version.split(".")[:2])
        except ValueError:
            major_minor = (999, 999)
        if major_minor > (0, 22):
            warnings.append(
                "The official Messenger README recommends gym<=0.22; its reset/step "
                "contract is the legacy Gym API."
            )

    if not package_installed or not gym_installed:
        detail = (
            "Messenger is not a PyPI environment; clone and install its official "
            "messenger-emma repository in a compatible environment."
        )
    elif import_error:
        detail = f"Messenger is installed but import failed: {import_error}"
    elif verify_import and not registered:
        detail = "Messenger imported, but no official msgr-* Gym IDs were registered."
    elif assets_present is False:
        detail = "Messenger is installed but official games/manual assets are incomplete."
    else:
        detail = "Messenger package and bundled official assets were detected."

    return EnvironmentProbe(
        name="messenger",
        package_installed=package_installed,
        gym_installed=gym_installed,
        version=_version("messenger"),
        gym_version=gym_version,
        expected_env_ids=MESSENGER_ENV_IDS,
        registered_env_ids=registered,
        data_assets_present=assets_present,
        importable=importable,
        compatibility_warnings=tuple(warnings),
        detail=detail,
        install_command=MESSENGER_INSTALL,
    )


def _load_homegrid() -> Tuple[Any, Any]:
    try:
        gym = import_module("gym")
        package = import_module("homegrid")
    except Exception as exc:
        raise HomeGridSetupError(
            "Unable to import the official HomeGrid environment. Install its "
            f"published package with `{HOMEGRID_INSTALL}`. No fallback environment "
            f"was created. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return gym, package


def _load_messenger() -> Tuple[Any, Any]:
    try:
        gym = import_module("gym")
        package = import_module("messenger")
    except Exception as exc:
        raise MessengerSetupError(
            "Unable to import the official Messenger environment. Clone "
            f"{MESSENGER_REPOSITORY} and install it locally (`{MESSENGER_INSTALL}`). "
            "Its README recommends Python 3.7 and gym<=0.22 when setup fails. "
            f"No fallback environment was created. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return gym, package


def make_homegrid_env(env_id: str = "homegrid-task", **kwargs: Any) -> Any:
    """Instantiate one of the four registered official HomeGrid tasks."""

    if env_id not in HOMEGRID_ENV_IDS:
        raise HomeGridSetupError(
            f"Unknown HomeGrid id {env_id!r}; official ids are {HOMEGRID_ENV_IDS}."
        )
    gym, _ = _load_homegrid()
    try:
        gym.spec(env_id)
        return gym.make(env_id, disable_env_checker=True, **kwargs)
    except Exception as exc:
        raise HomeGridSetupError(
            f"The official HomeGrid environment {env_id!r} could not be created: "
            f"{type(exc).__name__}: {exc}. Verify `{HOMEGRID_INSTALL}` and Gym 0.26."
        ) from exc


def make_messenger_env(env_id: str = "msgr-train-v2", **kwargs: Any) -> Any:
    """Instantiate a registered official Messenger environment."""

    if env_id not in MESSENGER_ENV_IDS:
        raise MessengerSetupError(
            f"Unknown Messenger id {env_id!r}; official ids are {MESSENGER_ENV_IDS}."
        )
    gym, _ = _load_messenger()
    try:
        gym.spec(env_id)
        return gym.make(env_id, **kwargs)
    except Exception as exc:
        raise MessengerSetupError(
            f"The official Messenger environment {env_id!r} could not be created: "
            f"{type(exc).__name__}: {exc}. Use an isolated legacy environment; "
            "the official README recommends Python 3.7 and gym<=0.22."
        ) from exc


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryFormatError(
            f"{label} must be a mapping, received {type(value).__name__}."
        )
    return value


def parse_homegrid_reset(result: Any) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate HomeGrid's Gym 0.26 ``(observation, info)`` reset."""

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TrajectoryFormatError(
            "HomeGrid reset() must return (observation, info) under Gym 0.26."
        )
    observation, info = result
    return (
        _require_mapping(observation, "HomeGrid observation"),
        _require_mapping(info, "HomeGrid reset info"),
    )


def parse_messenger_reset(result: Any) -> Tuple[Mapping[str, Any], Tuple[str, ...]]:
    """Validate Messenger's legacy ``(observation, manual)`` reset."""

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TrajectoryFormatError(
            "Messenger reset() must return (observation, manual)."
        )
    observation, manual = result
    observation = _require_mapping(observation, "Messenger observation")
    if isinstance(manual, str) or not isinstance(manual, (tuple, list)):
        raise TrajectoryFormatError(
            "Messenger manual must be the official list of description strings."
        )
    if not all(isinstance(sentence, str) for sentence in manual):
        raise TrajectoryFormatError(
            "Messenger manual contains a non-string entry; official manuals are strings."
        )
    return observation, tuple(manual)


def convert_homegrid_transition(
    observation: Mapping[str, Any],
    action: int,
    result: Any,
    step: int,
) -> WorldTransition:
    """Convert and validate HomeGrid's official five-value step result."""

    if not isinstance(result, (tuple, list)) or len(result) != 5:
        raise TrajectoryFormatError(
            "HomeGrid step() must return "
            "(observation, reward, terminated, truncated, info)."
        )
    next_observation, reward, terminated, truncated, info = result
    next_observation = _require_mapping(
        next_observation, "HomeGrid next observation"
    )
    raw_info = dict(_require_mapping(info, "HomeGrid step info"))
    try:
        action_index = int(action)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrajectoryFormatError("HomeGrid action must be an integer.") from exc
    if action_index not in HOMEGRID_ACTIONS:
        raise TrajectoryFormatError(
            f"HomeGrid action {action_index} is outside the official 0..9 space."
        )
    raw_info.update(
        {
            "backend": "homegrid",
            "action_name": HOMEGRID_ACTIONS[action_index],
        }
    )
    return WorldTransition(
        step=int(step),
        observation=observation,
        action=action_index,
        next_observation=next_observation,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=raw_info,
    )


def convert_messenger_transition(
    observation: Mapping[str, Any],
    action: int,
    result: Any,
    step: int,
) -> WorldTransition:
    """Convert and validate Messenger's official legacy four-value step result."""

    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise TrajectoryFormatError(
            "Messenger step() must use its legacy API and return "
            "(observation, reward, done, info); a five-value Gymnasium result "
            "indicates an unverified compatibility wrapper."
        )
    next_observation, reward, done, info = result
    next_observation = _require_mapping(
        next_observation, "Messenger next observation"
    )
    raw_info = dict(_require_mapping(info, "Messenger step info"))
    try:
        action_index = int(action)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrajectoryFormatError("Messenger action must be an integer.") from exc
    if action_index not in MESSENGER_ACTIONS:
        raise TrajectoryFormatError(
            f"Messenger action {action_index} is outside the official 0..4 space."
        )
    raw_info.update(
        {
            "backend": "messenger",
            "action_name": MESSENGER_ACTIONS[action_index],
            "legacy_gym_api": True,
        }
    )
    return WorldTransition(
        step=int(step),
        observation=observation,
        action=action_index,
        next_observation=next_observation,
        reward=float(reward),
        terminated=bool(done),
        truncated=False,
        info=raw_info,
    )


def environment_events(transition: WorldTransition) -> Tuple[TokenEvent, ...]:
    """Expose official HomeGrid info events as language-aligned stream items."""

    converted: List[TokenEvent] = []
    raw_events = transition.info.get("events", ())
    if not isinstance(raw_events, (tuple, list)):
        return ()
    for event in raw_events:
        if isinstance(event, Mapping):
            description = event.get("description")
            converted.append(
                TokenEvent(
                    step=transition.step + 1,
                    channel="environment_event",
                    text=description if isinstance(description, str) else None,
                    payload=event,
                    metadata={"type": event.get("type", "unknown")},
                )
            )
    return tuple(converted)


def _environment_goal(env: Any) -> Optional[str]:
    value = getattr(env, "task", None)
    if value is None:
        unwrapped = getattr(env, "unwrapped", None)
        value = getattr(unwrapped, "task", None) if unwrapped is not None else None
    return str(value) if value is not None else None


HOMEGRID_SEED_MODE_UNSEEDED = "unseeded"
HOMEGRID_SEED_MODE_GYM = "gym_reset"
HOMEGRID_SEED_MODE_COMPAT = (
    "homegrid_0.1.1_compat_python_numpy_np_random_spaces"
)


def _environment_chain(env: Any) -> Tuple[Any, ...]:
    """Return the concrete Gym ``.env`` chain without invoking proxies."""

    chain: List[Any] = []
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        namespace = getattr(current, "__dict__", {})
        current = namespace.get("env") if isinstance(namespace, dict) else None
    return tuple(chain)


def _has_official_homegrid_wrapper(env: Any) -> bool:
    """Match the exact top-level class shipped in HomeGrid 0.1.1."""

    return any(
        type(node).__module__ == "homegrid" and type(node).__name__ == "HomeGrid"
        for node in _environment_chain(env)
    )


def _is_seed_signature_error(error: TypeError) -> bool:
    message = str(error).lower()
    return "unexpected keyword argument" in message and "seed" in message


def _seed_official_homegrid_compat(env: Any, seed: int) -> None:
    """Seed every RNG surface used by the official HomeGrid 0.1.1 code.

    HomeGrid uses both Python's and NumPy's global RNGs.  Its Gym base also
    expects ``unwrapped._np_random``, while action/observation spaces own their
    own RNGs.  All five surfaces are seeded before the subsequent no-argument
    reset.  Failure of any surface is fatal instead of degrading silently.
    """

    try:
        numpy = import_module("numpy")
        random.seed(seed)
        numpy.random.seed(seed)

        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is None:
            chain = _environment_chain(env)
            unwrapped = chain[-1] if chain else None
        if unwrapped is None:
            raise AttributeError("environment has no unwrapped/base object")
        unwrapped._np_random = numpy.random.default_rng(seed)

        for name in ("action_space", "observation_space"):
            space = getattr(env, name, None)
            seed_fn = getattr(space, "seed", None)
            if not callable(seed_fn):
                raise AttributeError(f"{name} has no callable seed()")
            seed_fn(seed)
    except Exception as exc:
        raise TrajectoryFormatError(
            "Audited HomeGrid 0.1.1 compatibility seeding failed; refusing an "
            f"unseeded reset. Original error: {type(exc).__name__}: {exc}"
        ) from exc


class HomeGridRecorder:
    """Record a real HomeGrid rollout using its native multimodal observation."""

    def __init__(self, env: Any, env_id: str = "homegrid-task") -> None:
        if env_id not in HOMEGRID_ENV_IDS:
            raise HomeGridSetupError(
                f"Unknown HomeGrid id {env_id!r}; official ids are {HOMEGRID_ENV_IDS}."
            )
        self.env = env
        self.env_id = env_id
        self.initial_observation: Optional[Mapping[str, Any]] = None
        self.observation: Optional[Mapping[str, Any]] = None
        self.reset_info: Mapping[str, Any] = {}
        self.transitions: List[WorldTransition] = []
        self.goal: Optional[str] = None
        self.seed: Optional[int] = None
        self.seed_mode = HOMEGRID_SEED_MODE_UNSEEDED

    def reset(self, seed: Optional[int] = None) -> Mapping[str, Any]:
        seed_mode = HOMEGRID_SEED_MODE_UNSEEDED
        try:
            if seed is None:
                result = self.env.reset()
            else:
                result = self.env.reset(seed=seed)
                seed_mode = HOMEGRID_SEED_MODE_GYM
        except TypeError as exc:
            audited_fallback = (
                seed is not None
                and _is_seed_signature_error(exc)
                and _has_official_homegrid_wrapper(self.env)
            )
            if not audited_fallback:
                raise TrajectoryFormatError(
                    "HomeGrid reset(seed=...) raised TypeError outside the audited "
                    "HomeGrid 0.1.1 signature-compatibility case; the error was not "
                    f"swallowed: {type(exc).__name__}: {exc}"
                ) from exc
            _seed_official_homegrid_compat(self.env, seed)
            result = self.env.reset()
            seed_mode = HOMEGRID_SEED_MODE_COMPAT
        observation, info = parse_homegrid_reset(result)
        self.initial_observation = observation
        self.observation = observation
        reset_info = dict(info)
        reset_info["seed"] = seed
        reset_info["seed_mode"] = seed_mode
        self.reset_info = reset_info
        self.transitions.clear()
        self.goal = _environment_goal(self.env)
        self.seed = seed
        self.seed_mode = seed_mode
        return observation

    def step(self, action: int) -> WorldTransition:
        if self.observation is None:
            raise TrajectoryFormatError("Call HomeGridRecorder.reset() before step().")
        if self.transitions and self.transitions[-1].done:
            raise TrajectoryFormatError("The HomeGrid episode has already ended.")
        transition = convert_homegrid_transition(
            self.observation,
            action,
            self.env.step(action),
            len(self.transitions),
        )
        self.observation = transition.next_observation
        self.transitions.append(transition)
        return transition

    def episode(self) -> WorldEpisode:
        if self.initial_observation is None:
            raise TrajectoryFormatError("Call HomeGridRecorder.reset() first.")
        return WorldEpisode(
            source=self.env_id,
            initial_observation=self.initial_observation,
            transitions=tuple(self.transitions),
            goal=self.goal,
            metadata={
                "backend": "homegrid",
                "env_id": self.env_id,
                "reset_info": self.reset_info,
                "seed": self.seed,
                "seed_mode": self.seed_mode,
            },
        )


class MessengerRecorder:
    """Record official Messenger trajectories with their per-episode manual."""

    def __init__(
        self,
        env: Any,
        env_id: str = "msgr-train-v2",
        max_steps: Optional[int] = None,
    ) -> None:
        if env_id not in MESSENGER_ENV_IDS:
            raise MessengerSetupError(
                f"Unknown Messenger id {env_id!r}; official ids are "
                f"{MESSENGER_ENV_IDS}."
            )
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")
        self.env = env
        self.env_id = env_id
        self.max_steps = max_steps
        self.initial_observation: Optional[Mapping[str, Any]] = None
        self.observation: Optional[Mapping[str, Any]] = None
        self.manual: Tuple[str, ...] = ()
        self.transitions: List[WorldTransition] = []

    def reset(self) -> Mapping[str, Any]:
        observation, manual = parse_messenger_reset(self.env.reset())
        self.initial_observation = observation
        self.observation = observation
        self.manual = manual
        self.transitions.clear()
        return observation

    def step(self, action: int) -> WorldTransition:
        if self.observation is None:
            raise TrajectoryFormatError("Call MessengerRecorder.reset() before step().")
        if self.transitions and self.transitions[-1].done:
            raise TrajectoryFormatError("The Messenger episode has already ended.")
        transition = convert_messenger_transition(
            self.observation,
            action,
            self.env.step(action),
            len(self.transitions),
        )
        if (
            self.max_steps is not None
            and transition.step + 1 >= self.max_steps
            and not transition.terminated
        ):
            info = dict(transition.info)
            info["adapter_time_limit"] = self.max_steps
            transition = replace(transition, truncated=True, info=info)
        self.observation = transition.next_observation
        self.transitions.append(transition)
        return transition

    def episode(self) -> WorldEpisode:
        if self.initial_observation is None:
            raise TrajectoryFormatError("Call MessengerRecorder.reset() first.")
        return WorldEpisode(
            source=self.env_id,
            initial_observation=self.initial_observation,
            transitions=tuple(self.transitions),
            goal="\n".join(self.manual),
            metadata={
                "backend": "messenger",
                "env_id": self.env_id,
                "manual": self.manual,
                "max_steps": self.max_steps,
            },
        )


def record_homegrid_actions(
    env: Any,
    actions: Iterable[int],
    env_id: str = "homegrid-task",
    seed: Optional[int] = None,
) -> WorldEpisode:
    recorder = HomeGridRecorder(env, env_id=env_id)
    recorder.reset(seed=seed)
    for action in actions:
        transition = recorder.step(action)
        if transition.done:
            break
    return recorder.episode()


def record_messenger_actions(
    env: Any,
    actions: Iterable[int],
    env_id: str = "msgr-train-v2",
    max_steps: Optional[int] = None,
) -> WorldEpisode:
    recorder = MessengerRecorder(env, env_id=env_id, max_steps=max_steps)
    recorder.reset()
    for action in actions:
        transition = recorder.step(action)
        if transition.done:
            break
    return recorder.episode()


__all__ = [
    "EnvironmentProbe",
    "HOMEGRID_ACTIONS",
    "HOMEGRID_ENV_IDS",
    "HOMEGRID_SEED_MODE_COMPAT",
    "HOMEGRID_SEED_MODE_GYM",
    "HOMEGRID_SEED_MODE_UNSEEDED",
    "MESSENGER_ACTIONS",
    "MESSENGER_ENV_IDS",
    "HomeGridRecorder",
    "HomeGridSetupError",
    "MessengerRecorder",
    "MessengerSetupError",
    "TrajectoryFormatError",
    "convert_homegrid_transition",
    "convert_messenger_transition",
    "environment_events",
    "make_homegrid_env",
    "make_messenger_env",
    "parse_homegrid_reset",
    "parse_messenger_reset",
    "probe_homegrid",
    "probe_messenger",
    "record_homegrid_actions",
    "record_messenger_actions",
]
