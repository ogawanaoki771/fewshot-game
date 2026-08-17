# ============================================================
# EVOLVED EMBODIED SELF-ORGANIZING WORLD MODEL
#
# INTEGRATED VERSION
#
# Existing:
#   EventCell
#   TransitionCell
#   PlaceCell
#   ConceptCell
#   DreamSimulator
#
# Added:
#   SelfModel
#   CuriositySystem
#   CausalModel
#   GoalSystem
#   IntrinsicReward
#   ReinforcedDream
#
# 3 agents
# shared hippocampus
# shared world model
# ============================================================

import turtle
import random
import math
import numpy as np


# ============================================================
# 1. CONFIG
# ============================================================

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 760

NUM_AGENTS = 3

STEPS_PER_EPISODE = 180
MAX_EPISODES = 15

DAY_DELAY = 30
SLEEP_DELAY = 700

NUM_ACTIONS = 7

ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
ACTION_DASH = 4
ACTION_BRAKE = 5
ACTION_WAIT = 6

ACTION_NAMES = {
    ACTION_NONE: "NONE",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_JUMP: "JUMP",
    ACTION_DASH: "DASH",
    ACTION_BRAKE: "BRAKE",
    ACTION_WAIT: "WAIT",
}


# ============================================================
# 2. PHYSICS
# ============================================================

MAX_SPEED = 8.0
GROUND_ACCEL = 0.9
AIR_ACCEL = 0.45
GROUND_FRICTION = 0.82
AIR_FRICTION = 0.96

GRAVITY = 0.65
JUMP_POWER = 11.0
DOUBLE_JUMP_POWER = 9.0
MAX_JUMPS = 2

DASH_SPEED = 18.0
DASH_DURATION = 4


# ============================================================
# 3. MEMORY
# ============================================================

MAX_EVENTS = 1200
MAX_TRANSITIONS = 4000
MAX_REPLAY = 4000

EVENT_SIM_THRESHOLD = 0.88
PLACE_RADIUS = 0.70

MAX_ERROR_HISTORY = 1500

REPLAY_COUNT = 600
CONCEPT_REPLAY_COUNT = 300

DREAM_SOURCES = 20
DREAM_SAMPLES = 30
DREAM_STEPS = 12

PLANNING_DEPTH = 5


# ============================================================
# 4. UTILS
# ============================================================

def clamp(x, a, b):
    return max(a, min(b, x))


def softmax(values, temperature=1.0):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    temperature = max(0.05, temperature)

    values = values / temperature
    values -= np.max(values)

    exp_values = np.exp(values)
    total = exp_values.sum()

    if total <= 0:
        return np.ones(len(values)) / len(values)

    return exp_values / total


# ============================================================
# 5. REPLAY MEMORY
# ============================================================

class ReplayMemory:

    def __init__(self, max_size=MAX_REPLAY):
        self.memory = []
        self.max_size = max_size

    def add(
        self,
        previous_event,
        current_event,
        action,
        reward,
        error,
        intrinsic_reward=0.0
    ):
        self.memory.append({
            "previous": previous_event,
            "current": current_event,
            "action": int(action),
            "reward": float(reward),
            "intrinsic": float(intrinsic_reward),
            "error": float(error),
        })

        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def sample_batch(self, n):

        if not self.memory:
            return []

        n = min(n, len(self.memory))

        weights = np.array([
            1.0
            + abs(item["error"])
            + abs(item["intrinsic"])
            for item in self.memory
        ])

        weights /= weights.sum()

        indices = np.random.choice(
            len(self.memory),
            size=n,
            replace=False,
            p=weights
        )

        return [
            self.memory[i]
            for i in indices
        ]

    def __len__(self):
        return len(self.memory)


# ============================================================
# 6. EVENT CELL
# ============================================================

class EventCell:

    def __init__(
        self,
        event_id,
        state,
        action,
        agent_id
    ):

        self.id = event_id

        self.state = np.array(
            state,
            dtype=float
        )

        self.action = int(action)

        self.agent_id = agent_id

        self.visits = 1

        self.energy = 1.0
        self.activation = 1.0

        self.reward_mean = 0.0
        self.error_mean = 1.0

        self.place_id = None

        self.concepts = []

    def similarity(
        self,
        state,
        action
    ):

        if self.action != action:
            return 0.0

        state = np.asarray(
            state,
            dtype=float
        )

        distance = np.linalg.norm(
            self.state - state
        )

        return math.exp(
            -distance * 1.8
        )

    def reinforce(
        self,
        state,
        reward,
        error
    ):

        self.visits += 1

        self.energy = min(
            3.0,
            self.energy + 0.05
        )

        self.activation = min(
            2.0,
            self.activation + 0.10
        )

        state = np.asarray(
            state,
            dtype=float
        )

        self.state = (
            0.90 * self.state
            +
            0.10 * state
        )

        self.reward_mean = (
            0.90 * self.reward_mean
            +
            0.10 * reward
        )

        self.error_mean = (
            0.90 * self.error_mean
            +
            0.10 * error
        )

    def decay(self):

        self.energy *= 0.997
        self.activation *= 0.985


# ============================================================
# 7. TRANSITION CELL
# ============================================================

class TransitionCell:

    def __init__(
        self,
        transition_id,
        source_id,
        action,
        target_id,
        next_state,
        reward,
        error
    ):

        self.id = transition_id

        self.source_id = source_id

        self.action = int(action)

        self.target_counts = {
            target_id: 1
        }

        self.predicted_next_state = np.array(
            next_state,
            dtype=float
        )

        self.reward_mean = float(reward)

        self.error_mean = float(error)

        self.visits = 1

        self.success_count = (
            1 if reward > 0 else 0
        )

        self.failure_count = (
            1 if reward < -3 else 0
        )

        self.energy = 1.0

        self.confidence = 0.2

    def update(
        self,
        target_id,
        next_state,
        reward,
        error
    ):

        self.target_counts[target_id] = (
            self.target_counts.get(
                target_id,
                0
            ) + 1
        )

        self.visits += 1

        self.predicted_next_state = (
            0.85 * self.predicted_next_state
            +
            0.15 * np.asarray(
                next_state,
                dtype=float
            )
        )

        self.reward_mean = (
            0.85 * self.reward_mean
            +
            0.15 * reward
        )

        self.error_mean = (
            0.85 * self.error_mean
            +
            0.15 * error
        )

        if reward > 0:
            self.success_count += 1

        if reward < -3:
            self.failure_count += 1

        self.energy = min(
            3.0,
            self.energy + 0.04
        )

        self.confidence = (
            1.0
            -
            math.exp(
                -self.visits / 8.0
            )
        )

    def best_target(self):

        if not self.target_counts:
            return None

        return max(
            self.target_counts,
            key=self.target_counts.get
        )

    def target_probability(
        self,
        target_id
    ):

        total = sum(
            self.target_counts.values()
        )

        if total <= 0:
            return 0.0

        return (
            self.target_counts.get(
                target_id,
                0
            )
            /
            total
        )

    def risk(self):

        failure_rate = (
            self.failure_count
            /
            max(1, self.visits)
        )

        return (
            failure_rate
            +
            self.error_mean * 0.15
        )

    def decay(self):

        self.energy *= 0.998


# ============================================================
# 8. PLACE CELL
# ============================================================

class PlaceCell:

    def __init__(self, place_id):

        self.id = place_id

        self.center = None

        self.events = []

        self.visits = 0

        self.mean_reward = 0.0
        self.mean_error = 0.0

        self.energy = 1.0

        self.concepts = []

    def add_event(self, event):

        if event.id not in self.events:
            self.events.append(event.id)

        if self.center is None:

            self.center = event.state.copy()

        else:

            self.center = (
                0.90 * self.center
                +
                0.10 * event.state
            )

        self.visits += 1

        self.mean_reward = (
            0.90 * self.mean_reward
            +
            0.10 * event.reward_mean
        )

        self.mean_error = (
            0.90 * self.mean_error
            +
            0.10 * event.error_mean
        )

        self.energy = min(
            2.5,
            self.energy + 0.02
        )

        event.place_id = self.id

    def distance(self, state):

        if self.center is None:
            return 999999.0

        return float(
            np.linalg.norm(
                self.center
                -
                np.asarray(
                    state,
                    dtype=float
                )
            )
        )

    def decay(self):

        self.energy *= 0.998


# ============================================================
# 9. CONCEPT CELL
# ============================================================

class ConceptCell:

    def __init__(
        self,
        concept_id,
        name
    ):

        self.id = concept_id
        self.name = name

        self.events = []
        self.places = []

        self.links = {}

        self.visits = 0

        self.value = 0.0

        self.activation = 0.0

        self.energy = 1.0

    def absorb(self, event):

        if event.id not in self.events:
            self.events.append(event.id)

        if self.id not in event.concepts:
            event.concepts.append(self.id)

        self.visits += 1

        self.activation = (
            0.80 * self.activation
            +
            0.20
        )

        self.value = (
            0.90 * self.value
            +
            0.10 * event.reward_mean
        )

        self.energy = min(
            2.5,
            self.energy + 0.02
        )

    def strengthen(
        self,
        target_id,
        amount=0.05
    ):

        self.links[target_id] = min(
            8.0,
            self.links.get(
                target_id,
                0.0
            )
            +
            amount
        )

    def decay(self):

        self.activation *= 0.97
        self.energy *= 0.998


# ============================================================
# 10. HIPPOCAMPUS
# ============================================================

class Hippocampus:

    def __init__(self):

        self.events = []
        self.transitions = []
        self.places = []
        self.concepts = []

        self.next_event_id = 0
        self.next_transition_id = 0
        self.next_place_id = 0
        self.next_concept_id = 0

        self.total_encodes = 0
        self.total_transitions = 0

    def get_event(self, event_id):

        for event in self.events:

            if event.id == event_id:
                return event

        return None

    def get_transition(
        self,
        source_id,
        action
    ):

        for transition in self.transitions:

            if (
                transition.source_id == source_id
                and
                transition.action == action
            ):
                return transition

        return None

    def get_concept(self, concept_id):

        for concept in self.concepts:

            if concept.id == concept_id:
                return concept

        return None

    def encode_event(
        self,
        state,
        action,
        reward,
        error,
        agent_id
    ):

        self.total_encodes += 1

        best = None
        best_score = 0.0

        for event in self.events:

            score = event.similarity(
                state,
                action
            )

            if score > best_score:

                best_score = score
                best = event

        if (
            best is not None
            and
            best_score >= EVENT_SIM_THRESHOLD
        ):

            best.reinforce(
                state,
                reward,
                error
            )

            return best, False

        event = EventCell(
            self.next_event_id,
            state,
            action,
            agent_id
        )

        event.reward_mean = reward
        event.error_mean = error

        self.next_event_id += 1

        self.events.append(event)

        return event, True

    def encode_transition(
        self,
        source,
        action,
        target,
        next_state,
        reward,
        error
    ):

        transition = self.get_transition(
            source.id,
            action
        )

        if transition is None:

            transition = TransitionCell(
                self.next_transition_id,
                source.id,
                action,
                target.id,
                next_state,
                reward,
                error
            )

            self.next_transition_id += 1

            self.transitions.append(
                transition
            )

            self.total_transitions += 1

            return transition, True

        transition.update(
            target.id,
            next_state,
            reward,
            error
        )

        return transition, False

    def assign_place(self, event):

        best = None
        best_distance = 999999.0

        for place in self.places:

            distance = place.distance(
                event.state
            )

            if distance < best_distance:

                best_distance = distance
                best = place

        if (
            best is not None
            and
            best_distance < PLACE_RADIUS
        ):

            best.add_event(event)

            return best

        place = PlaceCell(
            self.next_place_id
        )

        self.next_place_id += 1

        place.add_event(event)

        self.places.append(place)

        return place

    def classify(self, event):

        if event.reward_mean >= 4:
            return "SUCCESS"

        if event.reward_mean <= -5:
            return "FAILURE"

        if event.error_mean >= 6:
            return "SURPRISE"

        if event.action == ACTION_JUMP:
            return "JUMP"

        if event.action == ACTION_DASH:
            return "DASH"

        if event.action == ACTION_BRAKE:
            return "BRAKE"

        if event.action == ACTION_LEFT:
            return "TURN_LEFT"

        if event.action == ACTION_RIGHT:
            return "TURN_RIGHT"

        if event.action == ACTION_WAIT:
            return "WAIT"

        return "MOVEMENT"

    def assign_concept(
        self,
        event,
        place
    ):

        name = self.classify(event)

        concept = None

        for c in self.concepts:

            if c.name == name:

                concept = c
                break

        if concept is None:

            concept = ConceptCell(
                self.next_concept_id,
                name
            )

            self.next_concept_id += 1

            self.concepts.append(concept)

        concept.absorb(event)

        if place.id not in concept.places:
            concept.places.append(place.id)

        if concept.id not in place.concepts:
            place.concepts.append(concept.id)

        return concept

    def organize(self, event):

        place = self.assign_place(event)

        concept = self.assign_concept(
            event,
            place
        )

        return place, concept

    def link_concepts(
        self,
        previous,
        current
    ):

        if previous is None:
            return

        if current is None:
            return

        for p_id in previous.concepts:

            p = self.get_concept(p_id)

            if p is None:
                continue

            for c_id in current.concepts:

                if p_id != c_id:

                    p.strengthen(c_id)

    def replay(
        self,
        replay_memory,
        count
    ):

        batch = replay_memory.sample_batch(count)

        for item in batch:

            previous = self.get_event(
                item["previous"]
            )

            current = self.get_event(
                item["current"]
            )

            if (
                previous is None
                or
                current is None
            ):
                continue

            transition = self.get_transition(
                previous.id,
                item["action"]
            )

            if transition is not None:

                transition.update(
                    current.id,
                    current.state,
                    item["reward"],
                    item["error"]
                )

    def replay_concepts(self, count):

        if not self.concepts:
            return

        for _ in range(count):

            concept = random.choice(
                self.concepts
            )

            concept.energy = min(
                2.5,
                concept.energy + 0.015
            )

            if concept.links:

                target_id = random.choice(
                    list(
                        concept.links.keys()
                    )
                )

                concept.links[target_id] = min(
                    8.0,
                    concept.links[target_id] * 1.01
                )

    def metabolize(self):

        for event in self.events:
            event.decay()

        for transition in self.transitions:
            transition.decay()

        for place in self.places:
            place.decay()

        for concept in self.concepts:
            concept.decay()

        self.events = [
            e for e in self.events
            if (
                e.visits >= 5
                or abs(e.reward_mean) >= 3
                or e.error_mean >= 6
                or e.energy > 0.18
            )
        ]

        self.transitions = [
            t for t in self.transitions
            if (
                t.visits >= 2
                or t.energy > 0.25
                or t.reward_mean > 1
            )
        ]

        if len(self.events) > MAX_EVENTS:

            self.events.sort(
                key=lambda e:
                    e.visits
                    +
                    e.energy
                    +
                    abs(e.reward_mean),
                reverse=True
            )

            self.events = self.events[:MAX_EVENTS]

        if len(self.transitions) > MAX_TRANSITIONS:

            self.transitions.sort(
                key=lambda t:
                    t.visits
                    +
                    t.energy
                    +
                    abs(t.reward_mean)
                    +
                    t.confidence,
                reverse=True
            )

            self.transitions = (
                self.transitions[:MAX_TRANSITIONS]
            )

    def statistics(self):

        return {
            "events": len(self.events),
            "transitions": len(self.transitions),
            "places": len(self.places),
            "concepts": len(self.concepts),
            "encodes": self.total_encodes,
        }


# ============================================================
# 11. SELF MODEL
# ============================================================

class SelfModel:

    """
    Agent itself is also represented internally.

    The world model answers:

        "What happens in the world?"

    SelfModel answers:

        "What kind of agent am I?"

        "How competent am I?"

        "How certain am I?"

        "What am I currently trying to do?"
    """

    def __init__(self):

        self.position = np.zeros(2)

        self.velocity = np.zeros(2)

        self.energy = 1.0

        self.confidence = 0.2

        self.competence = 0.2

        self.curiosity = 1.0

        self.fear = 0.0

        self.successes = 0
        self.failures = 0

        self.total_reward = 0.0

        self.action_counts = np.zeros(
            NUM_ACTIONS,
            dtype=float
        )

        self.goal_id = None

        self.identity = {
            "explorer": 0.5,
            "survivor": 0.5,
            "collector": 0.5,
            "controller": 0.5,
        }

    def update(
        self,
        agent,
        reward,
        error,
        action,
        event
    ):

        self.position = np.array([
            agent.x,
            agent.y
        ])

        self.velocity = np.array([
            agent.vx,
            agent.vy
        ])

        self.total_reward += reward

        self.action_counts[action] += 1

        # prediction accuracy
        prediction_quality = math.exp(
            -min(error, 10.0)
        )

        self.confidence = (
            0.95 * self.confidence
            +
            0.05 * prediction_quality
        )

        # competence
        if reward > 0:

            self.competence = min(
                1.0,
                self.competence + 0.015
            )

            self.successes += 1

        if reward < -3:

            self.competence *= 0.995

            self.failures += 1

        # fear
        if event in ("DANGER", "FALL"):

            self.fear = min(
                1.0,
                self.fear + 0.15
            )

        else:

            self.fear *= 0.98

        # identity
        if error > 3:

            self.identity["explorer"] += 0.002

        if reward > 3:

            self.identity["collector"] += 0.003

        if event in ("DANGER", "FALL"):

            self.identity["survivor"] += 0.004

        # normalize
        for key in self.identity:

            self.identity[key] = clamp(
                self.identity[key],
                0.0,
                2.0
            )

    def decay(self):

        self.curiosity *= 0.995

        self.curiosity = max(
            0.2,
            self.curiosity
        )

    def profile(self):

        return max(
            self.identity,
            key=self.identity.get
        )


# ============================================================
# 12. CURIOSITY
# ============================================================

class CuriositySystem:

    """
    Intrinsic motivation.

    External reward:
        "world gave me reward"

    Intrinsic reward:
        "I learned something"
        "I discovered something"
        "I encountered uncertainty"
    """

    def __init__(self):

        self.visit_counts = {}

        self.prediction_errors = []

        self.novelty_history = []

    def novelty(
        self,
        event_id
    ):

        count = self.visit_counts.get(
            event_id,
            0
        )

        self.visit_counts[event_id] = (
            count + 1
        )

        return 1.0 / math.sqrt(
            count + 1
        )

    def intrinsic_reward(
        self,
        event,
        prediction_error,
        transition=None
    ):

        novelty = self.novelty(
            event.id
        )

        surprise = clamp(
            prediction_error / 5.0,
            0.0,
            2.0
        )

        uncertainty = 0.0

        if transition is not None:

            uncertainty = clamp(
                transition.error_mean / 5.0,
                0.0,
                2.0
            )

        curiosity = (
            novelty * 1.2
            +
            surprise * 1.0
            +
            uncertainty * 0.4
        )

        self.prediction_errors.append(
            prediction_error
        )

        self.novelty_history.append(
            novelty
        )

        return curiosity

    def sleep(self):

        # Forget very old curiosity traces.
        if len(self.prediction_errors) > 2000:

            self.prediction_errors = (
                self.prediction_errors[-2000:]
            )

        if len(self.novelty_history) > 2000:

            self.novelty_history = (
                self.novelty_history[-2000:]
            )

    def average_error(self):

        if not self.prediction_errors:
            return 0.0

        return float(
            np.mean(
                self.prediction_errors[-100:]
            )
        )


# ============================================================
# 13. CAUSAL MODEL
# ============================================================

class CausalLink:

    def __init__(
        self,
        cause,
        effect
    ):

        self.cause = cause
        self.effect = effect

        self.observations = 0

        self.successes = 0
        self.failures = 0

        self.strength = 0.0

    def observe(
        self,
        success
    ):

        self.observations += 1

        if success:
            self.successes += 1
        else:
            self.failures += 1

        p = (
            self.successes
            /
            max(1, self.observations)
        )

        self.strength = (
            0.9 * self.strength
            +
            0.1 * p
        )


class CausalModel:

    """
    Learns simple action/event consequences.

    Example:

        JUMP -> SUCCESS
        DASH -> DANGER
        SWITCH -> BRIDGE
        BRAKE -> SURVIVAL

    This is not full causal discovery yet.

    It is a lightweight causal hypothesis system.
    """

    def __init__(self):

        self.links = {}

    def observe(
        self,
        action,
        event,
        reward
    ):

        key = (
            int(action),
            str(event)
        )

        if key not in self.links:

            self.links[key] = CausalLink(
                action,
                event
            )

        success = reward > 0

        self.links[key].observe(
            success
        )

    def predict_event(
        self,
        action
    ):

        candidates = [
            link
            for link in self.links.values()
            if link.cause == action
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda x:
                x.strength
        )

    def causal_value(
        self,
        action
    ):

        candidates = [
            link
            for link in self.links.values()
            if link.cause == action
        ]

        if not candidates:
            return 0.0

        value = 0.0

        for link in candidates:

            event_bonus = 0.0

            if link.effect == "SUCCESS":
                event_bonus += 1.5

            if link.effect in (
                "DANGER",
                "FALL"
            ):
                event_bonus -= 2.0

            value += (
                link.strength
                *
                event_bonus
            )

        return value

    def replay(self):

        for link in self.links.values():

            link.strength *= 0.999


# ============================================================
# 14. GOAL SYSTEM
# ============================================================

class Goal:

    def __init__(
        self,
        goal_id,
        name,
        goal_type,
        priority
    ):

        self.id = goal_id

        self.name = name

        self.goal_type = goal_type

        self.priority = priority

        self.progress = 0.0

        self.successes = 0

        self.failures = 0

        self.active = True

    def update(
        self,
        event,
        reward
    ):

        if self.goal_type == "SURVIVE":

            if event in (
                "DANGER",
                "FALL"
            ):
                self.progress -= 0.15
            else:
                self.progress += 0.01

        elif self.goal_type == "COLLECT":

            if event == "SUCCESS":
                self.progress += 0.2

        elif self.goal_type == "EXPLORE":

            if event in (
                "SWITCH",
                "SUCCESS"
            ):
                self.progress += 0.1

        elif self.goal_type == "LEARN":

            if reward != 0:
                self.progress += 0.02

        self.progress = clamp(
            self.progress,
            -1.0,
            1.0
        )


class GoalSystem:

    """
    Multiple simultaneous drives.

    Not just:

        maximize reward

    but:

        survive
        explore
        collect
        learn
    """

    def __init__(self):

        self.goals = []

        self.next_goal_id = 0

        self.create_default_goals()

        self.active_goal = None

    def create_default_goals(self):

        self.goals.append(
            Goal(
                self.next_goal_id,
                "SURVIVE",
                "SURVIVE",
                1.0
            )
        )

        self.next_goal_id += 1

        self.goals.append(
            Goal(
                self.next_goal_id,
                "COLLECT",
                "COLLECT",
                0.8
            )
        )

        self.next_goal_id += 1

        self.goals.append(
            Goal(
                self.next_goal_id,
                "EXPLORE",
                "EXPLORE",
                0.6
            )
        )

        self.next_goal_id += 1

        self.goals.append(
            Goal(
                self.next_goal_id,
                "LEARN",
                "LEARN",
                0.5
            )
        )

        self.next_goal_id += 1

    def update(
        self,
        event,
        reward
    ):

        for goal in self.goals:

            goal.update(
                event,
                reward
            )

    def select_goal(
        self,
        self_model,
        curiosity
    ):

        scores = []

        for goal in self.goals:

            score = goal.priority

            score += goal.progress * 0.5

            if goal.goal_type == "SURVIVE":

                score += (
                    self_model.fear
                    *
                    1.5
                )

            if goal.goal_type == "EXPLORE":

                score += (
                    self_model.curiosity
                    *
                    0.8
                )

            if goal.goal_type == "LEARN":

                score += (
                    curiosity
                    *
                    0.4
                )

            scores.append(score)

        if not scores:
            return None

        index = int(
            np.argmax(scores)
        )

        self.active_goal = self.goals[index]

        return self.active_goal

    def action_bonus(
        self,
        action,
        goal,
        causal_model
    ):

        if goal is None:
            return 0.0

        bonus = 0.0

        if goal.goal_type == "EXPLORE":

            if action in (
                ACTION_JUMP,
                ACTION_DASH
            ):
                bonus += 0.4

        if goal.goal_type == "SURVIVE":

            if action == ACTION_BRAKE:
                bonus += 0.4

        if goal.goal_type == "LEARN":

            bonus += (
                causal_model.causal_value(
                    action
                )
                *
                0.2
            )

        if goal.goal_type == "COLLECT":

            if action in (
                ACTION_RIGHT,
                ACTION_JUMP,
                ACTION_DASH
            ):
                bonus += 0.2

        return bonus


# ============================================================
# 15. DREAM SIMULATOR
# ============================================================

class DreamSimulator:

    def __init__(
        self,
        hippocampus
    ):

        self.hippo = hippocampus

    def candidates(
        self,
        event
    ):

        result = []

        for transition in self.hippo.transitions:

            if (
                transition.source_id
                ==
                event.id
            ):

                target_id = (
                    transition.best_target()
                )

                target = (
                    self.hippo.get_event(
                        target_id
                    )
                )

                if target is not None:

                    result.append(
                        (
                            transition,
                            target
                        )
                    )

        return result

    def rollout(
        self,
        start,
        steps,
        goal=None,
        curiosity=None
    ):

        path = []

        current = start

        total_reward = 0.0

        for _ in range(steps):

            path.append(current)

            candidates = self.candidates(
                current
            )

            if not candidates:
                break

            scores = []

            for transition, target in candidates:

                score = (
                    transition.reward_mean
                    +
                    transition.confidence * 0.5
                    -
                    transition.risk() * 2.0
                    -
                    transition.error_mean * 0.1
                )

                # novelty
                score += (
                    1.0
                    /
                    math.sqrt(
                        transition.visits + 1
                    )
                )

                # goal
                if goal is not None:

                    if (
                        goal.goal_type
                        ==
                        "COLLECT"
                    ):

                        score += (
                            target.reward_mean
                            *
                            0.3
                        )

                    if (
                        goal.goal_type
                        ==
                        "SURVIVE"
                    ):

                        score -= (
                            transition.risk()
                            *
                            2.0
                        )

                    if (
                        goal.goal_type
                        ==
                        "LEARN"
                    ):

                        score += (
                            transition.error_mean
                            *
                            0.1
                        )

                scores.append(score)

            probabilities = softmax(
                scores,
                temperature=0.8
            )

            index = np.random.choice(
                len(candidates),
                p=probabilities
            )

            transition, current = (
                candidates[index]
            )

            total_reward += (
                transition.reward_mean
            )

            # curiosity
            if curiosity is not None:

                total_reward += (
                    0.15
                    *
                    transition.error_mean
                )

        return path, total_reward

    def best_dream(
        self,
        start,
        samples,
        steps,
        goal=None,
        curiosity=None
    ):

        best_path = []

        best_score = -999999.0

        for _ in range(samples):

            path, reward = self.rollout(
                start,
                steps,
                goal,
                curiosity
            )

            if not path:
                continue

            score = reward

            for event in path:

                score += (
                    event.energy * 0.2
                )

                score -= (
                    event.error_mean * 0.05
                )

            if score > best_score:

                best_score = score

                best_path = path

        return best_path, best_score


# ============================================================
# 16. WORLD MODEL
# ============================================================

class WorldModel:

    def __init__(
        self,
        hippocampus,
        dream
    ):

        self.hippo = hippocampus

        self.dream = dream

        self.replay = ReplayMemory()

        self.last_event = {}

        self.error_history = []

        self.last_dream = []

        self.last_dream_score = 0.0

        self.exploration = 1.0

        self.episode_count = 0

        # NEW SYSTEMS
        self.self_models = {}

        self.curiosity = CuriositySystem()

        self.causal = CausalModel()

        self.goals = GoalSystem()

    # ========================================================
    # REGISTER AGENT
    # ========================================================

    def register_agent(
        self,
        agent_id
    ):

        self.self_models[agent_id] = (
            SelfModel()
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        state,
        action
    ):

        best_event = None

        best_score = 0.0

        for event in self.hippo.events:

            score = event.similarity(
                state,
                action
            )

            if score > best_score:

                best_score = score
                best_event = event

        if best_event is None:

            return {
                "next_state": None,
                "reward": 0.0,
                "uncertainty": 1.0,
                "event": None,
                "transition": None,
            }

        transition = (
            self.hippo.get_transition(
                best_event.id,
                action
            )
        )

        if transition is None:

            return {
                "next_state": None,
                "reward": 0.0,
                "uncertainty": 1.0,
                "event": best_event,
                "transition": None,
            }

        uncertainty = (
            transition.error_mean
            /
            (
                1.0
                +
                transition.visits
            )
        )

        return {
            "next_state":
                transition.predicted_next_state.copy(),

            "reward":
                transition.reward_mean,

            "uncertainty":
                uncertainty,

            "event":
                best_event,

            "transition":
                transition,
        }

    # ========================================================
    # LEARN
    # ========================================================

    def learn(
        self,
        agent_id,
        state,
        action,
        next_state,
        reward,
        world_event
    ):

        previous_event = (
            self.last_event.get(
                agent_id
            )
        )

        event, created = (
            self.hippo.encode_event(
                state,
                action,
                reward,
                0.0,
                agent_id
            )
        )

        error = 1.0

        transition = None

        if previous_event is not None:

            predicted = self.predict(
                previous_event.state,
                action
            )

            if predicted["next_state"] is None:

                error = 1.0

            else:

                error = float(
                    np.linalg.norm(
                        predicted["next_state"]
                        -
                        next_state
                    )
                )

            transition, _ = (
                self.hippo.encode_transition(
                    previous_event,
                    action,
                    event,
                    next_state,
                    reward,
                    error
                )
            )

            self.hippo.link_concepts(
                previous_event,
                event
            )

            intrinsic = (
                self.curiosity.intrinsic_reward(
                    event,
                    error,
                    transition
                )
            )

            self.replay.add(
                previous_event.id,
                event.id,
                action,
                reward,
                error,
                intrinsic
            )

            self.error_history.append(
                error
            )

        else:

            intrinsic = 0.0

        # ----------------------------------------------------
        # CAUSAL LEARNING
        # ----------------------------------------------------

        self.causal.observe(
            action,
            world_event,
            reward
        )

        # ----------------------------------------------------
        # SELF MODEL
        # ----------------------------------------------------

        self_model = (
            self.self_models.get(
                agent_id
            )
        )

        if self_model is not None:

            # curiosity controls exploration
            self_model.curiosity = (
                0.98 * self_model.curiosity
                +
                0.02 * intrinsic
            )

        # ----------------------------------------------------
        # ORGANIZATION
        # ----------------------------------------------------

        place, concept = (
            self.hippo.organize(
                event
            )
        )

        # ----------------------------------------------------
        # GOALS
        # ----------------------------------------------------

        self.goals.update(
            world_event,
            reward
        )

        self.last_event[agent_id] = event

        if len(self.error_history) > MAX_ERROR_HISTORY:

            self.error_history.pop(0)

        return {
            "event": event,
            "place": place,
            "concept": concept,
            "created": created,
            "error": error,
            "intrinsic": intrinsic,
            "transition": transition,
        }

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_action(
        self,
        state,
        agent_id
    ):

        self_model = (
            self.self_models.get(
                agent_id
            )
        )

        if self_model is None:

            self.register_agent(
                agent_id
            )

            self_model = (
                self.self_models[
                    agent_id
                ]
            )

        goal = self.goals.select_goal(
            self_model,
            self.curiosity.average_error()
        )

        scores = []

        for action in range(NUM_ACTIONS):

            prediction = self.predict(
                state,
                action
            )

            transition = (
                prediction["transition"]
            )

            # ------------------------------------------------
            # UNKNOWN
            # ------------------------------------------------

            if transition is None:

                score = (
                    2.5
                    *
                    self.exploration
                )

                # curiosity
                score += (
                    self_model.curiosity
                    *
                    0.5
                )

                if action == ACTION_JUMP:
                    score += 0.4

                if action == ACTION_DASH:
                    score += 0.3

            else:

                reward = (
                    prediction["reward"]
                )

                uncertainty = (
                    prediction["uncertainty"]
                )

                confidence = (
                    transition.confidence
                )

                risk = (
                    transition.risk()
                )

                novelty = (
                    1.0
                    /
                    math.sqrt(
                        transition.visits + 1
                    )
                )

                # ------------------------------------------------
                # MODEL BASED SCORE
                # ------------------------------------------------

                score = (

                    reward * 1.5

                    +

                    uncertainty * 1.2

                    +

                    novelty * 0.8

                    +

                    confidence * 0.4

                    -

                    risk * 3.0
                )

                # curiosity
                score += (
                    uncertainty
                    *
                    self_model.curiosity
                    *
                    0.8
                )

            # ------------------------------------------------
            # GOAL
            # ------------------------------------------------

            score += self.goals.action_bonus(
                action,
                goal,
                self.causal
            )

            # ------------------------------------------------
            # CAUSAL MODEL
            # ------------------------------------------------

            score += (
                self.causal.causal_value(
                    action
                )
                *
                0.5
            )

            # ------------------------------------------------
            # PLANNING
            # ------------------------------------------------

            planning = (
                self.evaluate_action_sequence(
                    state,
                    action
                )
            )

            score += (
                planning * 0.7
            )

            # ------------------------------------------------
            # EXPLORATION
            # ------------------------------------------------

            score += (
                random.random()
                *
                0.35
                *
                self.exploration
            )

            scores.append(score)

        return int(
            np.argmax(scores)
        )

    # ========================================================
    # PLANNING
    # ========================================================

    def evaluate_action_sequence(
        self,
        state,
        first_action
    ):

        prediction = self.predict(
            state,
            first_action
        )

        transition = (
            prediction["transition"]
        )

        if transition is None:
            return 0.0

        score = prediction["reward"]

        current_id = (
            transition.best_target()
        )

        current = (
            self.hippo.get_event(
                current_id
            )
        )

        if current is None:
            return score

        for depth in range(
            PLANNING_DEPTH - 1
        ):

            candidates = []

            for t in self.hippo.transitions:

                if t.source_id == current.id:

                    candidates.append(t)

            if not candidates:
                break

            best = max(
                candidates,
                key=lambda t:
                    t.reward_mean
                    -
                    t.risk() * 2.0
                    +
                    t.confidence * 0.5
            )

            score += (
                best.reward_mean
                *
                (0.8 ** depth)
            )

            target_id = (
                best.best_target()
            )

            current = (
                self.hippo.get_event(
                    target_id
                )
            )

            if current is None:
                break

        return score

    # ========================================================
    # SLEEP
    # ========================================================

    def sleep(self):

        if not self.hippo.events:

            self.last_dream = []

            return

        # ----------------------------------------------------
        # REPLAY
        # ----------------------------------------------------

        self.hippo.replay(
            self.replay,
            REPLAY_COUNT
        )

        # ----------------------------------------------------
        # CONCEPT REPLAY
        # ----------------------------------------------------

        self.hippo.replay_concepts(
            CONCEPT_REPLAY_COUNT
        )

        # ----------------------------------------------------
        # CAUSAL REPLAY
        # ----------------------------------------------------

        self.causal.replay()

        # ----------------------------------------------------
        # CURIOSITY CONSOLIDATION
        # ----------------------------------------------------

        self.curiosity.sleep()

        # ----------------------------------------------------
        # SELECT GOAL FOR DREAM
        # ----------------------------------------------------

        if self.self_models:

            self_model = (
                next(
                    iter(
                        self.self_models.values()
                    )
                )
            )

        else:

            self_model = SelfModel()

        goal = self.goals.select_goal(
            self_model,
            self.curiosity.average_error()
        )

        # ----------------------------------------------------
        # RANK DREAM SOURCES
        # ----------------------------------------------------

        ranked = sorted(
            self.hippo.events,
            key=lambda e:
                e.energy
                +
                e.visits * 0.25
                +
                e.error_mean * 0.15
                +
                abs(e.reward_mean) * 0.25,
            reverse=True
        )

        best_path = []

        best_score = -999999.0

        # ----------------------------------------------------
        # MULTI-SOURCE DREAM
        # ----------------------------------------------------

        for start in ranked[
            :DREAM_SOURCES
        ]:

            path, score = (
                self.dream.best_dream(
                    start,
                    DREAM_SAMPLES,
                    DREAM_STEPS,
                    goal,
                    self.curiosity
                )
            )

            if (
                path
                and
                score > best_score
            ):

                best_path = path

                best_score = score

        self.last_dream = best_path

        self.last_dream_score = best_score

        # ----------------------------------------------------
        # DREAM REINFORCEMENT
        # ----------------------------------------------------

        self.reinforce_dream(
            best_path
        )

        # ----------------------------------------------------
        # METABOLISM
        # ----------------------------------------------------

        self.hippo.metabolize()

        # ----------------------------------------------------
        # SELF CONSOLIDATION
        # ----------------------------------------------------

        for self_model in (
            self.self_models.values()
        ):

            self_model.decay()

        # ----------------------------------------------------
        # EXPLORATION DECAY
        # ----------------------------------------------------

        self.exploration *= 0.92

        self.exploration = max(
            0.18,
            self.exploration
        )

        # ----------------------------------------------------
        # RESET EPISODIC POINTER
        # ----------------------------------------------------

        self.last_event.clear()

    # ========================================================
    # REINFORCED DREAM
    # ========================================================

    def reinforce_dream(
        self,
        path
    ):

        if len(path) < 2:
            return

        for i in range(
            len(path) - 1
        ):

            current = path[i]

            target = path[i + 1]

            transition = (
                self.hippo.get_transition(
                    current.id,
                    current.action
                )
            )

            if transition is None:
                continue

            # dream evidence
            transition.energy = min(
                3.0,
                transition.energy + 0.08
            )

            transition.confidence = min(
                1.0,
                transition.confidence + 0.015
            )

            # dream reward propagation
            transition.reward_mean = (
                0.98 * transition.reward_mean
                +
                0.02 * target.reward_mean
            )

            # concept reinforcement
            for concept_id in (
                current.concepts
            ):

                concept = (
                    self.hippo.get_concept(
                        concept_id
                    )
                )

                if concept is None:
                    continue

                for next_concept_id in (
                    target.concepts
                ):

                    if (
                        concept_id
                        !=
                        next_concept_id
                    ):

                        concept.strengthen(
                            next_concept_id,
                            0.08
                        )


# ============================================================
# 17. PHYSICAL WORLD
# ============================================================

class TurtleWorld:

    def __init__(self):

        self.left = -520
        self.right = -20

        self.bottom = -280
        self.top = 270

        self.platforms = [
            [-520, -460, -130],
            [-445, -350, -70],
            [-335, -240, -10],
            [-225, -120, -90],
            [-100, -25, 30],
        ]

        self.hazards = [
            [-400, -355, -130],
            [-300, -250, -10],
            [-190, -145, -90],
        ]

        self.orbs = []

        self.switch_x = -300
        self.switch_y = -95

        self.bridge_active = False

        self.moving_platform = {
            "base_x": -250.0,
            "y": 80.0,
            "width": 70.0,
            "amplitude": 90.0,
        }

        self.time = 0.0

        self.checkpoint = (
            -490.0,
            -110.0
        )

        self.drawer = turtle.Turtle()

        self.drawer.hideturtle()

        self.drawer.penup()

        self.drawer.speed(0)

    def reset(self):

        self.time = 0.0

        self.bridge_active = False

        self.orbs = [
            np.array(
                [-125.0, 55.0]
            ),
            np.array(
                [-40.0, 100.0]
            ),
        ]

        self.draw()

    def moving_x(self):

        return (
            self.moving_platform["base_x"]
            +
            self.moving_platform["amplitude"]
            *
            math.sin(
                self.time * 0.05
            )
        )

    def update(self):

        self.time += 1.0

    def platform_at(
        self,
        x,
        y
    ):

        for x1, x2, py in self.platforms:

            if (
                x1 <= x <= x2
                and
                abs(y - py) < 20
            ):

                return py

        if self.bridge_active:

            if (
                -350 <= x <= -120
                and
                abs(y + 75) < 20
            ):

                return -75

        mx = self.moving_x()

        mp = self.moving_platform

        if (
            mx - mp["width"] / 2
            <= x
            <=
            mx + mp["width"] / 2
            and
            abs(y - mp["y"]) < 20
        ):

            return mp["y"]

        return None

    def is_hazard(
        self,
        x,
        y
    ):

        for x1, x2, hy in self.hazards:

            if (
                x1 <= x <= x2
                and
                y <= hy + 18
            ):

                return True

        return False

    def near_switch(
        self,
        x,
        y
    ):

        return (
            math.hypot(
                x - self.switch_x,
                y - self.switch_y
            )
            <
            24
        )

    def collect_orb(
        self,
        x,
        y
    ):

        collected = 0

        remaining = []

        for orb in self.orbs:

            distance = np.linalg.norm(
                np.array([x, y])
                -
                orb
            )

            if distance < 22:

                collected += 1

            else:

                remaining.append(orb)

        self.orbs = remaining

        return collected

    def step(
        self,
        agent,
        action
    ):

        x = float(agent.x)
        y = float(agent.y)

        vx = float(agent.vx)
        vy = float(agent.vy)

        grounded = bool(
            agent.grounded
        )

        jumps = int(
            agent.jumps
        )

        dash_timer = int(
            agent.dash_timer
        )

        reward = -0.015

        event = "NORMAL"

        if action == ACTION_LEFT:

            vx -= (
                GROUND_ACCEL
                if grounded
                else AIR_ACCEL
            )

            event = "MOVE"

        elif action == ACTION_RIGHT:

            vx += (
                GROUND_ACCEL
                if grounded
                else AIR_ACCEL
            )

            event = "MOVE"

        elif action == ACTION_JUMP:

            if grounded:

                vy = JUMP_POWER

                grounded = False

                jumps = 1

                event = "JUMP"

            elif jumps < MAX_JUMPS:

                vy = DOUBLE_JUMP_POWER

                jumps += 1

                event = "DOUBLE_JUMP"

        elif action == ACTION_DASH:

            if dash_timer <= 0:

                direction = 1

                if abs(vx) > 0.5:

                    direction = (
                        1
                        if vx > 0
                        else -1
                    )

                vx = (
                    direction
                    *
                    DASH_SPEED
                )

                dash_timer = DASH_DURATION

                event = "DASH"

        elif action == ACTION_BRAKE:

            vx *= 0.30

            event = "BRAKE"

        elif action == ACTION_WAIT:

            vx *= 0.90

            event = "WAIT"

        vx = np.clip(
            vx,
            -MAX_SPEED,
            MAX_SPEED
        )

        if grounded:

            vx *= GROUND_FRICTION

        else:

            vx *= AIR_FRICTION

        if not grounded:

            vy -= GRAVITY

            y += vy

        x += vx

        if x < self.left + 10:

            x = self.left + 10

            vx *= -0.4

            event = "WALL"

        if x > self.right - 10:

            x = self.right - 10

            vx *= -0.4

            event = "WALL"

        if self.near_switch(x, y):

            if not self.bridge_active:

                self.bridge_active = True

                reward += 2.0

                event = "SWITCH"

        if self.is_hazard(x, y):

            reward -= 8.0

            event = "DANGER"

            x, y = self.checkpoint

            vx = 0.0
            vy = 0.0

            grounded = True
            jumps = 0

        py = self.platform_at(
            x,
            y
        )

        if (
            py is not None
            and
            vy <= 0
            and
            y <= py + 20
        ):

            y = py + 20

            vy = 0.0

            grounded = True

            jumps = 0

        elif y < -245:

            reward -= 10.0

            event = "FALL"

            x, y = self.checkpoint

            vx = 0.0
            vy = 0.0

            grounded = True

            jumps = 0

        else:

            grounded = False

        collected = self.collect_orb(
            x,
            y
        )

        if collected > 0:

            reward += (
                5.0 * collected
            )

            event = "SUCCESS"

        agent.x = x
        agent.y = y

        agent.vx = vx
        agent.vy = vy

        agent.grounded = grounded

        agent.jumps = jumps

        agent.dash_timer = max(
            0,
            dash_timer - 1
        )

        agent.turtle.goto(
            x,
            y
        )

        return {
            "reward": float(reward),
            "event": event,
        }

    def draw(self):

        self.drawer.clear()

        self.drawer.color(
            "#444455"
        )

        self.drawer.goto(
            self.left,
            self.bottom
        )

        self.drawer.pendown()

        self.drawer.goto(
            self.right,
            self.bottom
        )

        self.drawer.goto(
            self.right,
            self.top
        )

        self.drawer.goto(
            self.left,
            self.top
        )

        self.drawer.goto(
            self.left,
            self.bottom
        )

        self.drawer.penup()

        self.drawer.color(
            "#00dddd"
        )

        for x1, x2, y in self.platforms:

            self.drawer.goto(
                x1,
                y
            )

            self.drawer.pendown()

            self.drawer.goto(
                x2,
                y
            )

            self.drawer.penup()

        if self.bridge_active:

            self.drawer.color(
                "#ffaa00"
            )

            self.drawer.goto(
                -350,
                -75
            )

            self.drawer.pendown()

            self.drawer.goto(
                -120,
                -75
            )

            self.drawer.penup()

        mx = self.moving_x()

        mp = self.moving_platform

        self.drawer.color(
            "#00ff88"
        )

        self.drawer.goto(
            mx - mp["width"] / 2,
            mp["y"]
        )

        self.drawer.pendown()

        self.drawer.goto(
            mx + mp["width"] / 2,
            mp["y"]
        )

        self.drawer.penup()

        self.drawer.color(
            "#ff3344"
        )

        for x1, x2, y in self.hazards:

            self.drawer.goto(
                x1,
                y
            )

            self.drawer.pendown()

            self.drawer.goto(
                x2,
                y
            )

            self.drawer.penup()

        self.drawer.goto(
            self.switch_x,
            self.switch_y
        )

        self.drawer.dot(
            15,
            "#ff00ff"
            if self.bridge_active
            else "#555555"
        )

        self.drawer.color(
            "#ffff00"
        )

        for orb in self.orbs:

            self.drawer.goto(
                orb[0],
                orb[1]
            )

            self.drawer.dot(
                11
            )


# ============================================================
# 18. SENSOR
# ============================================================

def get_sensors(
    world,
    agent
):

    front_distance = 35.0

    direction = (
        1.0
        if agent.vx >= 0
        else -1.0
    )

    front_x = (
        agent.x
        +
        front_distance * direction
    )

    front_platform = (
        world.platform_at(
            front_x,
            agent.y
        )
        is not None
    )

    hazard = world.is_hazard(
        agent.x,
        agent.y
    )

    near_switch = world.near_switch(
        agent.x,
        agent.y
    )

    moving_x = world.moving_x()

    moving_near = (
        abs(agent.x - moving_x)
        <
        50
    )

    nearest_orb = 1.0

    if world.orbs:

        distances = [
            np.linalg.norm(
                np.array([
                    agent.x,
                    agent.y
                ])
                -
                orb
            )
            for orb in world.orbs
        ]

        nearest_orb = (
            min(distances)
            /
            400.0
        )

    return np.array([

        agent.x / 500.0,

        agent.y / 200.0,

        agent.vx / MAX_SPEED,

        agent.vy / 15.0,

        float(agent.grounded),

        float(
            agent.jumps / MAX_JUMPS
        ),

        float(
            agent.dash_timer > 0
        ),

        float(front_platform),

        float(hazard),

        float(near_switch),

        float(world.bridge_active),

        float(moving_near),

        nearest_orb,

    ], dtype=float)


# ============================================================
# 19. AGENT
# ============================================================

class Agent:

    def __init__(
        self,
        agent_id,
        color,
        world,
        model
    ):

        self.id = agent_id

        self.color = color

        self.world = world

        self.model = model

        self.turtle = turtle.Turtle()

        self.turtle.shape(
            "turtle"
        )

        self.turtle.color(
            color
        )

        self.turtle.penup()

        self.turtle.speed(0)

        self.model.register_agent(
            agent_id
        )

        self.reset()

    def reset(self):

        self.x = (
            -490.0
            +
            self.id * 18
        )

        self.y = -110.0

        self.vx = 0.0
        self.vy = 0.0

        self.grounded = True

        self.jumps = 0

        self.dash_timer = 0

        self.steps = 0

        self.episode_reward = 0.0

        self.total_reward = 0.0

        self.successes = 0

        self.failures = 0

        self.last_action = ACTION_NONE

        self.last_event = "NONE"

        self.last_error = 0.0

        self.last_intrinsic = 0.0

        self.turtle.goto(
            self.x,
            self.y
        )

    def step(self):

        # SENSOR
        state = get_sensors(
            self.world,
            self
        )

        # ACTION
        action = self.model.select_action(
            state,
            self.id
        )

        # ACT
        result = self.world.step(
            self,
            action
        )

        # OBSERVE
        next_state = get_sensors(
            self.world,
            self
        )

        # LEARN
        learned = self.model.learn(
            self.id,
            state,
            action,
            next_state,
            result["reward"],
            result["event"]
        )

        error = learned["error"]

        intrinsic = learned[
            "intrinsic"
        ]

        # SELF MODEL
        self_model = (
            self.model.self_models[
                self.id
            ]
        )

        self_model.update(
            self,
            result["reward"],
            error,
            action,
            result["event"]
        )

        # STATS
        self.steps += 1

        self.last_action = action

        self.last_event = (
            result["event"]
        )

        self.last_error = error

        self.last_intrinsic = intrinsic

        self.episode_reward += (
            result["reward"]
        )

        self.total_reward += (
            result["reward"]
        )

        if result["event"] == "SUCCESS":

            self.successes += 1

        if result["event"] in (
            "DANGER",
            "FALL"
        ):

            self.failures += 1


# ============================================================
# 20. SCREEN
# ============================================================

screen = turtle.Screen()

screen.setup(
    SCREEN_WIDTH,
    SCREEN_HEIGHT
)

screen.bgcolor(
    "#0b0b12"
)

screen.title(
    "Integrated Embodied Self-Organizing World Model"
)

screen.tracer(False)


# ============================================================
# 21. SHARED BRAIN
# ============================================================

shared_hippocampus = (
    Hippocampus()
)

shared_dream = DreamSimulator(
    shared_hippocampus
)

shared_model = WorldModel(
    shared_hippocampus,
    shared_dream
)


# ============================================================
# 22. WORLD
# ============================================================

world = TurtleWorld()


# ============================================================
# 23. AGENTS
# ============================================================

agent_colors = [
    "#00ffff",
    "#00ff7f",
    "#ffa500",
]

agents = []

for i, color in enumerate(
    agent_colors
):

    agents.append(
        Agent(
            i,
            color,
            world,
            shared_model
        )
    )


# ============================================================
# 24. DRAWERS
# ============================================================

model_drawer = turtle.Turtle()

model_drawer.hideturtle()
model_drawer.penup()
model_drawer.speed(0)

text_drawer = turtle.Turtle()

text_drawer.hideturtle()
text_drawer.penup()
text_drawer.speed(0)

dream_drawer = turtle.Turtle()

dream_drawer.hideturtle()
dream_drawer.penup()
dream_drawer.speed(0)


# ============================================================
# 25. TEXT
# ============================================================

def write_text(
    x,
    y,
    text,
    size=10,
    color="#dddddd"
):

    text_drawer.goto(
        x,
        y
    )

    text_drawer.color(
        color
    )

    text_drawer.write(
        text,
        font=(
            "Arial",
            size,
            "normal"
        )
    )


# ============================================================
# 26. DRAW MODEL
# ============================================================

def draw_world_model():

    model_drawer.clear()

    text_drawer.clear()

    dream_drawer.clear()

    stats = (
        shared_hippocampus.statistics()
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    write_text(
        -510,
        315,
        "EXTERNAL WORLD",
        15,
        "#ffffff"
    )

    write_text(
        50,
        315,
        "EVOLVED SHARED WORLD MODEL",
        15,
        "#ffffff"
    )

    # --------------------------------------------------------
    # WORLD MODEL
    # --------------------------------------------------------

    write_text(
        50,
        290,
        f"Events       : {stats['events']}",
        10
    )

    write_text(
        50,
        273,
        f"Transitions  : {stats['transitions']}",
        10,
        "#66ccff"
    )

    write_text(
        50,
        256,
        f"Places       : {stats['places']}",
        10
    )

    write_text(
        50,
        239,
        f"Concepts     : {stats['concepts']}",
        10
    )

    write_text(
        50,
        222,
        f"Replay       : {len(shared_model.replay)}",
        10
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if shared_model.error_history:

        error = np.mean(
            shared_model.error_history[-50:]
        )

    else:

        error = 0.0

    write_text(
        50,
        205,
        f"Prediction Error : {error:.3f}",
        10,
        "#ffaa00"
    )

    write_text(
        50,
        188,
        f"Exploration      : {shared_model.exploration:.3f}",
        10,
        "#aa88ff"
    )

    # --------------------------------------------------------
    # CURIOSITY
    # --------------------------------------------------------

    write_text(
        50,
        171,
        f"Curiosity Error  : "
        f"{shared_model.curiosity.average_error():.3f}",
        10,
        "#ff66ff"
    )

    # --------------------------------------------------------
    # GOAL
    # --------------------------------------------------------

    goal = (
        shared_model.goals.active_goal
    )

    goal_name = (
        goal.name
        if goal is not None
        else "NONE"
    )

    write_text(
        50,
        154,
        f"Active Goal      : {goal_name}",
        10,
        "#ffff66"
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    important_events = sorted(
        shared_hippocampus.events,
        key=lambda e:
            e.energy
            +
            e.visits * 0.5
            +
            abs(e.reward_mean)
            +
            e.error_mean * 0.15,
        reverse=True
    )[:60]

    positions = {}

    for i, event in enumerate(
        important_events
    ):

        x = (
            90
            +
            (i % 10) * 42
        )

        y = (
            130
            -
            (i // 10) * 42
        )

        positions[event.id] = (
            x,
            y
        )

    # --------------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------------

    for transition in (
        shared_hippocampus.transitions
    ):

        if (
            transition.source_id
            not in positions
        ):
            continue

        target_id = (
            transition.best_target()
        )

        if target_id not in positions:
            continue

        sx, sy = positions[
            transition.source_id
        ]

        tx, ty = positions[
            target_id
        ]

        if transition.reward_mean > 2:

            color = "#00ff7f"

        elif transition.reward_mean < -3:

            color = "#ff3344"

        elif transition.confidence > 0.7:

            color = "#668899"

        else:

            color = "#354454"

        model_drawer.color(
            color
        )

        model_drawer.pensize(
            int(
                clamp(
                    1
                    +
                    transition.confidence * 3,
                    1,
                    4
                )
            )
        )

        model_drawer.goto(
            sx,
            sy
        )

        model_drawer.pendown()

        model_drawer.goto(
            tx,
            ty
        )

        model_drawer.penup()

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    concept_colors = {

        "SUCCESS":
            "#00ff7f",

        "FAILURE":
            "#ff3344",

        "SURPRISE":
            "#ff9900",

        "JUMP":
            "#bb88ff",

        "DASH":
            "#ffff00",

        "BRAKE":
            "#ff66aa",

        "TURN_LEFT":
            "#66ccff",

        "TURN_RIGHT":
            "#66ccff",

        "WAIT":
            "#999999",

        "MOVEMENT":
            "#66ccff",
    }

    for event in important_events:

        x, y = positions[
            event.id
        ]

        color = "#66ccff"

        if event.concepts:

            concept = (
                shared_hippocampus.get_concept(
                    event.concepts[0]
                )
            )

            if concept:

                color = concept_colors.get(
                    concept.name,
                    "#66ccff"
                )

        size = int(
            clamp(
                5 + event.visits,
                5,
                18
            )
        )

        model_drawer.goto(
            x,
            y
        )

        model_drawer.dot(
            size,
            color
        )

    # --------------------------------------------------------
    # AGENTS
    # --------------------------------------------------------

    y = 175

    for agent in agents:

        self_model = (
            shared_model.self_models[
                agent.id
            ]
        )

        write_text(
            -510,
            y,
            (
                f"Agent {agent.id} "
                f"R={agent.episode_reward:+.1f} "
                f"{ACTION_NAMES.get(agent.last_action)} "
                f"{agent.last_event} "
                f"PE={agent.last_error:.2f} "
                f"IR={agent.last_intrinsic:.2f} "
                f"C={self_model.confidence:.2f} "
                f"COMP={self_model.competence:.2f}"
            ),
            8,
            agent.color
        )

        y -= 19

    # --------------------------------------------------------
    # SELF
    # --------------------------------------------------------

    write_text(
        50,
        -115,
        "SELF MODEL",
        11,
        "#ffffff"
    )

    y = -135

    for agent in agents:

        sm = shared_model.self_models[
            agent.id
        ]

        write_text(
            50,
            y,
            (
                f"A{agent.id} "
                f"identity={sm.profile()} "
                f"fear={sm.fear:.2f} "
                f"curiosity={sm.curiosity:.2f}"
            ),
            8,
            agent.color
        )

        y -= 16

    # --------------------------------------------------------
    # CONCEPTS
    # --------------------------------------------------------

    write_text(
        300,
        -115,
        "CONCEPTS",
        11,
        "#ffffff"
    )

    y = -135

    concepts = sorted(
        shared_hippocampus.concepts,
        key=lambda c:
            c.visits,
        reverse=True
    )[:8]

    for concept in concepts:

        color = concept_colors.get(
            concept.name,
            "#cccccc"
        )

        write_text(
            300,
            y,
            (
                f"{concept.name:<13} "
                f"n={concept.visits:<4} "
                f"V={concept.value:+.2f}"
            ),
            8,
            color
        )

        y -= 16

    # --------------------------------------------------------
    # CAUSAL MODEL
    # --------------------------------------------------------

    write_text(
        510,
        -115,
        "CAUSAL MODEL",
        11,
        "#ffffff"
    )

    y = -135

    causal_items = sorted(
        shared_model.causal.links.values(),
        key=lambda x:
            x.strength,
        reverse=True
    )[:7]

    for link in causal_items:

        write_text(
            510,
            y,
            (
                f"{ACTION_NAMES.get(link.cause)} "
                f"-> {link.effect:<8} "
                f"C={link.strength:.2f}"
            ),
            8,
            "#ffcc66"
        )

        y -= 16

    # --------------------------------------------------------
    # DREAM
    # --------------------------------------------------------

    if shared_model.last_dream:

        write_text(
            50,
            -285,
            "LAST REINFORCED DREAM",
            11,
            "#aa88ff"
        )

        x = 190

        for event in (
            shared_model.last_dream[:18]
        ):

            if event.reward_mean >= 3:

                color = "#00ff7f"

            elif event.reward_mean <= -5:

                color = "#ff3344"

            elif event.error_mean >= 6:

                color = "#ff9900"

            else:

                color = "#aa88ff"

            dream_drawer.goto(
                x,
                -285
            )

            dream_drawer.dot(
                10,
                color
            )

            x += 22

        write_text(
            50,
            -312,
            (
                f"dream score = "
                f"{shared_model.last_dream_score:+.2f}"
            ),
            9,
            "#9999bb"
        )


# ============================================================
# 27. EPISODE CONTROL
# ============================================================

episode = 0

current_step = 0

phase = "DAY"

finished = False


def reset_episode():

    world.reset()

    shared_model.last_event.clear()

    for agent in agents:

        agent.reset()


# ============================================================
# 28. SLEEP
# ============================================================

def run_sleep():

    global episode
    global current_step
    global phase
    global finished

    if finished:
        return

    phase = "SLEEP"

    shared_model.sleep()

    world.draw()

    draw_world_model()

    write_text(
        -510,
        285,
        "SLEEP / REPLAY / CAUSAL CONSOLIDATION / DREAM",
        12,
        "#aa88ff"
    )

    screen.update()

    episode += 1

    if episode >= MAX_EPISODES:

        finished = True

        write_text(
            -200,
            -340,
            "SIMULATION FINISHED",
            18,
            "#ffffff"
        )

        screen.update()

        return

    current_step = 0

    screen.ontimer(
        start_day,
        SLEEP_DELAY
    )


# ============================================================
# 29. DAY
# ============================================================

def run_day():

    global current_step
    global phase

    if finished:
        return

    phase = "DAY"

    world.update()

    for agent in agents:

        agent.step()

    world.draw()

    draw_world_model()

    write_text(
        -510,
        285,
        (
            f"DAY "
            f"Episode {episode + 1}/{MAX_EPISODES} "
            f"Step {current_step}/{STEPS_PER_EPISODE}"
        ),
        11,
        "#00ff7f"
    )

    screen.update()

    current_step += 1

    if (
        current_step
        <
        STEPS_PER_EPISODE
    ):

        screen.ontimer(
            run_day,
            DAY_DELAY
        )

    else:

        screen.ontimer(
            run_sleep,
            300
        )


# ============================================================
# 30. START DAY
# ============================================================

def start_day():

    if finished:
        return

    reset_episode()

    run_day()


# ============================================================
# 31. START
# ============================================================

reset_episode()

world.draw()

draw_world_model()

screen.update()

screen.ontimer(
    start_day,
    500
)

turtle.done()
