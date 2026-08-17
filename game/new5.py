import turtle
import random
import math
import numpy as np
from collections import defaultdict


# ============================================================
# WORLD MODEL EXPERIMENT
#
# Experience
#     ↓
# EventCell
#     ↓
# Shared Hippocampus
#     ↓
# Event -> Event transition graph
#     ↓
# Prediction / Surprise
#     ↓
# Replay / Counterfactual
#
# 左: 実世界
# 右: 内部世界モデル
# ============================================================


# ============================================================
# 0. Constants
# ============================================================

SCREEN_W = 1200
SCREEN_H = 700

WORLD_LEFT = -580
WORLD_RIGHT = 0

MODEL_LEFT = 20
MODEL_RIGHT = 580

STEPS_PER_EPISODE = 180
MAX_EPISODES = 40

REPLAY_COUNT = 100

ACTION_FORWARD = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3

ACTIONS = [
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_JUMP
]

ACTION_NAMES = {
    0: "FWD",
    1: "LEFT",
    2: "RIGHT",
    3: "JUMP"
}


# ============================================================
# 1. Event Cell
# ============================================================

class EventCell:

    def __init__(
        self,
        cell_id,
        context,
        action,
        expected,
        actual,
        surprise
    ):

        self.id = cell_id

        self.context = np.array(
            context,
            dtype=float
        )

        self.action = int(action)

        self.expected = np.array(
            expected,
            dtype=float
        )

        self.actual = np.array(
            actual,
            dtype=float
        )

        self.surprise = float(surprise)

        self.energy = 1.0
        self.visits = 1

        # ------------------------------------------------
        # links[action][target_id] = strength
        # ------------------------------------------------

        self.links = defaultdict(dict)

        # Number of transitions observed
        self.transition_count = defaultdict(int)

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    def similarity(self, context, action):

        if self.action != action:
            return 0.0

        context = np.asarray(
            context,
            dtype=float
        )

        d = np.linalg.norm(
            self.context - context
        )

        # Gaussian-like similarity
        return math.exp(
            -(d ** 2) / 2.0
        )

    # --------------------------------------------------------
    # Display strength
    # --------------------------------------------------------

    def importance(self):

        return (
            self.visits
            * (0.5 + self.energy)
            * (1.0 + min(self.surprise, 10.0) * 0.1)
        )


# ============================================================
# 2. Hippocampus
# ============================================================

class Hippocampus:

    def __init__(
        self,
        threshold=0.72,
        max_cells=250
    ):

        self.cells = []

        self.max_id = 0

        self.threshold = threshold

        self.max_cells = max_cells

    # --------------------------------------------------------
    # Find Event
    # --------------------------------------------------------

    def find_best(
        self,
        context,
        action
    ):

        best = None
        best_score = -1.0

        for cell in self.cells:

            score = cell.similarity(
                context,
                action
            )

            if score > best_score:

                best_score = score
                best = cell

        return best, best_score

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    def encode(
        self,
        context,
        action,
        expected,
        actual,
        surprise
    ):

        context = np.asarray(
            context,
            dtype=float
        )

        expected = np.asarray(
            expected,
            dtype=float
        )

        actual = np.asarray(
            actual,
            dtype=float
        )

        best, best_score = self.find_best(
            context,
            action
        )

        # ------------------------------------------------
        # Existing event
        # ------------------------------------------------

        if (
            best is not None
            and best_score >= self.threshold
        ):

            best.visits += 1

            best.energy = min(
                2.5,
                best.energy + 0.05
            )

            best.surprise = (
                0.8 * best.surprise
                + 0.2 * surprise
            )

            best.expected = (
                0.85 * best.expected
                + 0.15 * expected
            )

            best.actual = (
                0.80 * best.actual
                + 0.20 * actual
            )

            return best, False

        # ------------------------------------------------
        # New event
        # ------------------------------------------------

        cell = EventCell(
            self.max_id,
            context,
            action,
            expected,
            actual,
            surprise
        )

        self.max_id += 1

        self.cells.append(cell)

        # Limit memory size
        if len(self.cells) > self.max_cells:

            self.cells.sort(
                key=lambda c: c.importance(),
                reverse=True
            )

            self.cells = self.cells[
                :self.max_cells
            ]

        return cell, True

    # --------------------------------------------------------
    # Add transition
    # --------------------------------------------------------

    def add_transition(
        self,
        source,
        action,
        target
    ):

        if source is None:
            return

        if target is None:
            return

        target_id = target.id

        current = source.links[
            int(action)
        ].get(
            target_id,
            0.0
        )

        # Stronger every time it is experienced
        source.links[
            int(action)
        ][target_id] = min(
            5.0,
            current + 0.08
        )

        source.transition_count[
            target_id
        ] += 1

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(
        self,
        context,
        action
    ):

        if not self.cells:
            return None, None, None

        cell, score = self.find_best(
            context,
            action
        )

        if cell is None:
            return None, None, None

        if score < 0.40:
            return None, None, None

        return (
            cell.expected.copy(),
            cell.surprise,
            cell
        )

    # --------------------------------------------------------
    # Counterfactual replay
    # --------------------------------------------------------

    def counterfactual_replay(
        self,
        n=100
    ):

        if len(self.cells) < 2:
            return

        for _ in range(n):

            source = random.choice(
                self.cells
            )

            hypothetical_action = random.choice(
                ACTIONS
            )

            # ------------------------------------------------
            # If experience exists, strengthen it.
            # Otherwise make a weak hypothetical edge.
            # ------------------------------------------------

            existing = source.links[
                hypothetical_action
            ]

            if existing:

                target_id = random.choice(
                    list(existing.keys())
                )

                existing[target_id] = min(
                    5.0,
                    existing[target_id] + 0.015
                )

                target = self.get_cell(
                    target_id
                )

                if target is not None:

                    target.energy = min(
                        2.5,
                        target.energy + 0.003
                    )

            else:

                # Find a plausible target based on context.
                candidates = self.cells

                target = min(
                    candidates,
                    key=lambda c:
                    np.linalg.norm(
                        c.context - source.context
                    )
                )

                source.links[
                    hypothetical_action
                ][target.id] = 0.03

    # --------------------------------------------------------
    # Get cell
    # --------------------------------------------------------

    def get_cell(self, cell_id):

        for cell in self.cells:

            if cell.id == cell_id:
                return cell

        return None

    # --------------------------------------------------------
    # Pruning
    # --------------------------------------------------------

    def prune(
        self,
        radius=0.15
    ):

        if len(self.cells) < 30:
            return

        # Low-importance events die.
        retained = []

        ranked = sorted(
            self.cells,
            key=lambda c: c.importance(),
            reverse=True
        )

        for cell in ranked:

            keep = True

            for existing in retained:

                if existing.action != cell.action:
                    continue

                d = np.linalg.norm(
                    existing.context
                    - cell.context
                )

                if d < radius:

                    # Keep the more important one.
                    keep = False
                    break

            if keep:
                retained.append(cell)

            if len(retained) >= self.max_cells:
                break

        self.cells = retained

    # --------------------------------------------------------
    # Metabolism
    # --------------------------------------------------------

    def metabolize(self):

        alive = []

        for cell in self.cells:

            cell.energy -= 0.006

            # Memory that keeps being visited survives.
            if cell.visits > 8:
                cell.energy += 0.002

            if cell.energy > 0.15:
                alive.append(cell)

        self.cells = alive

    # --------------------------------------------------------
    # Get graph statistics
    # --------------------------------------------------------

    def graph_statistics(self):

        node_count = len(self.cells)

        edge_count = 0

        total_weight = 0.0

        for cell in self.cells:

            for action_edges in cell.links.values():

                edge_count += len(action_edges)

                total_weight += sum(
                    action_edges.values()
                )

        return {
            "nodes": node_count,
            "edges": edge_count,
            "weight": total_weight
        }


# ============================================================
# 3. World Model
# ============================================================

class WorldModel:

    def __init__(
        self,
        hippocampus
    ):

        self.hippo = hippocampus

        # Last event for each agent.
        self.last_event = {}

        # Prediction history
        self.prediction_errors = []

        self.surprise_history = []

        self.created_events = 0

    # --------------------------------------------------------
    # Learning
    # --------------------------------------------------------

    def learn(
        self,
        agent_id,
        context,
        action,
        expected,
        actual,
        surprise
    ):

        cell, created = self.hippo.encode(
            context,
            action,
            expected,
            actual,
            surprise
        )

        if created:
            self.created_events += 1

        # ------------------------------------------------
        # CRITICAL:
        #
        # Link the actual temporal experience of this agent.
        #
        # Not:
        #     cells[-2] -> cells[-1]
        #
        # But:
        #     previous_event(agent)
        #             ↓
        #     current_event(agent)
        # ------------------------------------------------

        previous = self.last_event.get(
            agent_id
        )

        if previous is not None:

            self.hippo.add_transition(
                previous,
                action,
                cell
            )

        self.last_event[
            agent_id
        ] = cell

        self.prediction_errors.append(
            float(surprise)
        )

        self.surprise_history.append(
            float(surprise)
        )

        # Keep history bounded
        if len(self.prediction_errors) > 2000:
            self.prediction_errors.pop(0)

        if len(self.surprise_history) > 2000:
            self.surprise_history.pop(0)

        return cell, created

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(
        self,
        context,
        action
    ):

        return self.hippo.predict(
            context,
            action
        )

    # --------------------------------------------------------
    # Reset episode-specific previous events
    # --------------------------------------------------------

    def reset_agents(self):

        self.last_event.clear()

    # --------------------------------------------------------
    # Mean prediction error
    # --------------------------------------------------------

    def mean_error(self, n=100):

        if not self.prediction_errors:
            return 0.0

        values = self.prediction_errors[
            -n:
        ]

        return float(
            np.mean(values)
        )


# ============================================================
# 4. Turtle World
# ============================================================

screen = turtle.Screen()

screen.setup(
    SCREEN_W,
    SCREEN_H
)

screen.bgcolor(
    "#090b10"
)

screen.title(
    "Embodied World Model Observatory"
)

screen.tracer(False)


# ------------------------------------------------------------
# Drawers
# ------------------------------------------------------------

world_drawer = turtle.Turtle()
world_drawer.hideturtle()
world_drawer.penup()
world_drawer.speed(0)

model_drawer = turtle.Turtle()
model_drawer.hideturtle()
model_drawer.penup()
model_drawer.speed(0)

text_drawer = turtle.Turtle()
text_drawer.hideturtle()
text_drawer.penup()
text_drawer.speed(0)

orb_drawer = turtle.Turtle()
orb_drawer.hideturtle()
orb_drawer.penup()
orb_drawer.speed(0)

moving_drawer = turtle.Turtle()
moving_drawer.hideturtle()
moving_drawer.penup()
moving_drawer.speed(0)


# ============================================================
# 5. Environment
# ============================================================

static_platforms = [
    [-520, -410, -110],
    [-250, -150, 40],
]

hazard_zones = [
    [-390, -310, -170]
]

bridge_active = False

switch_pos = np.array(
    [-430.0, -50.0]
)

orbs = [
    np.array([-470.0, -30.0]),
    np.array([-200.0, 60.0]),
    np.array([-20.0, 80.0])
]


moving_platform = {
    "center_x": -60.0,
    "y": 10.0,
    "width": 50.0,
    "time": 0.0
}


# ============================================================
# 6. Draw World
# ============================================================

def draw_world():

    world_drawer.clear()

    # --------------------------------------------------------
    # separator
    # --------------------------------------------------------

    world_drawer.color(
        "#252936"
    )

    world_drawer.goto(
        0,
        -330
    )

    world_drawer.pendown()

    world_drawer.goto(
        0,
        330
    )

    world_drawer.penup()

    # --------------------------------------------------------
    # World border
    # --------------------------------------------------------

    world_drawer.color(
        "#394050"
    )

    world_drawer.goto(
        -560,
        -320
    )

    world_drawer.pendown()

    for _ in range(4):

        world_drawer.forward(
            520
        )

        world_drawer.left(90)

    world_drawer.penup()

    # --------------------------------------------------------
    # Static platforms
    # --------------------------------------------------------

    world_drawer.color(
        "#00d9ff"
    )

    for p in static_platforms:

        world_drawer.goto(
            p[0],
            p[2]
        )

        world_drawer.pendown()

        world_drawer.goto(
            p[1],
            p[2]
        )

        world_drawer.penup()

    # --------------------------------------------------------
    # Hazard
    # --------------------------------------------------------

    world_drawer.color(
        "#ff3344"
    )

    for hz in hazard_zones:

        world_drawer.goto(
            hz[0],
            hz[2]
        )

        world_drawer.pendown()

        world_drawer.goto(
            hz[1],
            hz[2]
        )

        world_drawer.penup()

    # --------------------------------------------------------
    # Bridge
    # --------------------------------------------------------

    if bridge_active:

        world_drawer.color(
            "#ffaa00"
        )

        world_drawer.goto(
            -430,
            -50
        )

        world_drawer.pendown()

        world_drawer.goto(
            -330,
            -50
        )

        world_drawer.penup()

    # --------------------------------------------------------
    # Switch
    # --------------------------------------------------------

    world_drawer.goto(
        switch_pos[0],
        switch_pos[1]
    )

    world_drawer.dot(
        18,
        "#ff00ff"
        if bridge_active
        else "#555866"
    )


def draw_orbs():

    orb_drawer.clear()

    for orb in orbs:

        orb_drawer.goto(
            orb[0],
            orb[1]
        )

        orb_drawer.dot(
            10,
            "#ffff00"
        )


draw_world()
draw_orbs()


# ============================================================
# 7. Sensors
# ============================================================

def get_local_sensors(
    x,
    h,
    heading,
    is_grounded
):

    rad = np.radians(
        heading
    )

    front_x = (
        x
        + 20 * np.cos(rad)
    )

    has_floor = 0.0

    for p in static_platforms:

        if (
            p[0] <= front_x <= p[1]
            and abs(h - p[2]) < 15
        ):

            has_floor = 1.0

    if (
        bridge_active
        and -430 <= front_x <= -330
        and abs(h + 50) < 15
    ):

        has_floor = 1.0

    mx = (
        moving_platform["center_x"]
        + 30 * np.sin(
            moving_platform["time"]
        )
    )

    if (
        mx - 25 <= front_x <= mx + 25
        and abs(h - moving_platform["y"]) < 15
    ):

        has_floor = 1.0

    switch_dist = np.linalg.norm(
        np.array(
            [x, h]
        )
        - switch_pos
    )

    near_switch = (
        1.0
        if switch_dist < 25
        else 0.0
    )

    nearest_orb_dist = 1.0

    if orbs:

        dists = [

            np.linalg.norm(
                np.array(
                    [x, h]
                ) - orb
            )

            for orb in orbs
        ]

        nearest_orb_dist = (
            min(dists) / 200.0
        )

    danger = 0.0

    for hz in hazard_zones:

        if (
            hz[0] - 20 <= x <= hz[1] + 20
            and h <= hz[2] + 20
        ):

            danger = 1.0

    # 6-dimensional observation.
    return np.array([
        has_floor,
        1.0 if abs(front_x + 40) < 500 else 0.0,
        1.0 if is_grounded else 0.0,
        near_switch,
        nearest_orb_dist,
        danger
    ])


# ============================================================
# 8. Physics
# ============================================================

def physics_update(
    agent,
    action
):

    global bridge_active

    t = agent["turtle"]

    x = agent["x"]
    h = agent["h"]

    vh = agent["vh"]

    grounded = agent["is_grounded"]

    triggered = False

    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    if action == ACTION_FORWARD:

        t.forward(
            6
        )

    elif action == ACTION_LEFT:

        t.left(
            30
        )

    elif action == ACTION_RIGHT:

        t.right(
            30
        )

    elif action == ACTION_JUMP:

        if grounded:

            vh = 7.2

            grounded = False

    # --------------------------------------------------------
    # Gravity
    # --------------------------------------------------------

    if not grounded:

        vh -= 0.6

        h += vh

    x = t.xcor()

    # Keep turtle heading independent of vertical h.
    # The turtle is essentially the horizontal body.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Switch
    # --------------------------------------------------------

    if (
        np.linalg.norm(
            np.array(
                [x, h]
            )
            - switch_pos
        ) < 20
    ):

        bridge_active = not bridge_active

        triggered = True

        draw_world()

    # --------------------------------------------------------
    # Orb
    # --------------------------------------------------------

    for orb in orbs[:]:

        if (
            np.linalg.norm(
                np.array(
                    [x, h]
                ) - orb
            ) < 18
        ):

            orbs.remove(
                orb
            )

            agent["score"] += 1

            triggered = True

            orbs.append(
                np.array([
                    random.randint(
                        -500,
                        -20
                    ),
                    random.randint(
                        -100,
                        150
                    )
                ],
                dtype=float)
            )

            draw_orbs()

    # --------------------------------------------------------
    # Hazard
    # --------------------------------------------------------

    for hz in hazard_zones:

        if (
            hz[0] <= x <= hz[1]
            and h <= hz[2] + 5
        ):

            t.goto(
                -520,
                0
            )

            x = -520

            h = -40

            vh = 0.0

            grounded = True

            triggered = True

    # --------------------------------------------------------
    # Platform collision
    # --------------------------------------------------------

    target_h = None

    for p in static_platforms:

        if (
            p[0] <= x <= p[1]
            and abs(h - p[2]) < 18
        ):

            target_h = p[2]

    # Bridge
    if (
        bridge_active
        and -430 <= x <= -330
        and abs(h + 50) < 18
    ):

        target_h = -50

    # Moving platform
    mx = (
        moving_platform["center_x"]
        + 30 * np.sin(
            moving_platform["time"]
        )
    )

    if (
        mx - 25 <= x <= mx + 25
        and abs(h - moving_platform["y"]) < 18
    ):

        target_h = moving_platform["y"]

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    if x < -545:

        t.goto(
            -520,
            0
        )

        x = -520
        h = -40
        grounded = True
        vh = 0.0

    if x > -10:

        t.goto(
            -20,
            0
        )

        x = -20
        h = -40
        grounded = True
        vh = 0.0

    # --------------------------------------------------------
    # Landing
    # --------------------------------------------------------

    if (
        target_h is not None
        and h <= target_h + 5
    ):

        h = target_h

        grounded = True

        vh = 0.0

    elif h < -180:

        t.goto(
            -520,
            0
        )

        x = -520

        h = -40

        grounded = True

        vh = 0.0

        triggered = True

    else:

        grounded = False

    agent["x"] = x
    agent["h"] = h
    agent["vh"] = vh
    agent["is_grounded"] = grounded

    return triggered


# ============================================================
# 9. Agents
# ============================================================

shared_hippocampus = Hippocampus(
    threshold=0.72,
    max_cells=220
)

shared_model = WorldModel(
    shared_hippocampus
)


agent_colors = [
    "#00ffff",
    "#00ff7f",
    "#ffa500"
]

agents = []

for i in range(3):

    t = turtle.Turtle()

    t.shape(
        "turtle"
    )

    t.color(
        agent_colors[i]
    )

    t.penup()

    t.speed(0)

    t.goto(
        -520,
        0
    )

    agents.append({

        "id": i,

        "turtle": t,

        "x": -520.0,

        "h": -40.0,

        "vh": 0.0,

        "is_grounded": True,

        "score": 0
    })


# ============================================================
# 10. Visualization helpers
# ============================================================

def normalize01(
    values
):

    if not values:
        return []

    mn = min(values)
    mx = max(values)

    if abs(mx - mn) < 1e-8:

        return [
            0.5
            for _ in values
        ]

    return [
        (v - mn) / (mx - mn)
        for v in values
    ]


# ============================================================
# 11. Draw Internal World Model
# ============================================================

def draw_internal_model():

    model_drawer.clear()

    cells = shared_hippocampus.cells

    if not cells:
        return

    # --------------------------------------------------------
    # Select important nodes
    # --------------------------------------------------------

    ranked = sorted(
        cells,
        key=lambda c: c.importance(),
        reverse=True
    )

    # Draw at most 45 nodes.
    display_cells = ranked[:45]

    display_ids = {
        c.id
        for c in display_cells
    }

    # --------------------------------------------------------
    # Layout
    #
    # Context[4] = orb distance
    # Context[3] = switch
    # surprise controls y
    # --------------------------------------------------------

    positions = {}

    for i, cell in enumerate(
        display_cells
    ):

        angle = (
            2.0
            * math.pi
            * i
            / max(
                1,
                len(display_cells)
            )
        )

        radius = 210

        x = (
            300
            + radius * math.cos(angle)
        )

        y = (
            radius * math.sin(angle)
        )

        positions[cell.id] = (
            x,
            y
        )

    # --------------------------------------------------------
    # Draw edges FIRST.
    # --------------------------------------------------------

    for source in display_cells:

        sx, sy = positions[
            source.id
        ]

        for action, targets in source.links.items():

            for target_id, weight in targets.items():

                if (
                    target_id
                    not in display_ids
                ):
                    continue

                tx, ty = positions[
                    target_id
                ]

                # Edge intensity by weight.
                width = max(
                    1,
                    min(
                        5,
                        int(
                            weight * 1.2
                        )
                    )
                )

                model_drawer.color(
                    "#596273"
                )

                model_drawer.pensize(
                    width
                )

                model_drawer.goto(
                    MODEL_LEFT + sx,
                    sy
                )

                model_drawer.pendown()

                model_drawer.goto(
                    MODEL_LEFT + tx,
                    ty
                )

                model_drawer.penup()

    model_drawer.pensize(
        1
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    for cell in display_cells:

        x, y = positions[
            cell.id
        ]

        screen_x = (
            MODEL_LEFT + x
        )

        # ------------------------------------------------
        # Color by surprise
        # ------------------------------------------------

        if cell.surprise > 12:

            node_color = "#ff3355"

        elif cell.surprise > 5:

            node_color = "#ffcc33"

        else:

            node_color = "#66ccff"

        # ------------------------------------------------
        # Size by importance
        # ------------------------------------------------

        size = int(
            max(
                5,
                min(
                    16,
                    4
                    + cell.visits * 0.6
                )
            )
        )

        model_drawer.goto(
            screen_x,
            y
        )

        model_drawer.dot(
            size,
            node_color
        )

    # --------------------------------------------------------
    # Border
    # --------------------------------------------------------

    model_drawer.color(
        "#252936"
    )

    model_drawer.goto(
        -10,
        -325
    )

    model_drawer.pendown()

    model_drawer.goto(
        570,
        -325
    )

    model_drawer.goto(
        570,
        325
    )

    model_drawer.goto(
        -10,
        325
    )

    model_drawer.goto(
        -10,
        -325
    )

    model_drawer.penup()


# ============================================================
# 12. Draw Statistics
# ============================================================

def write_text(
    x,
    y,
    text,
    size=12,
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
        align="left",
        font=(
            "Arial",
            size,
            "normal"
        )
    )


def draw_statistics(
    episode,
    step
):

    text_drawer.clear()

    stats = (
        shared_hippocampus
        .graph_statistics()
    )

    write_text(
        -550,
        315,
        "WORLD",
        16,
        "#ffffff"
    )

    write_text(
        30,
        315,
        "INTERNAL WORLD MODEL",
        16,
        "#ffffff"
    )

    write_text(
        -550,
        -345,
        f"Episode: {episode + 1}/{MAX_EPISODES}"
    )

    write_text(
        -550,
        -365,
        f"Step: {step}/{STEPS_PER_EPISODE}"
    )

    write_text(
        30,
        -345,
        f"Events: {stats['nodes']}"
    )

    write_text(
        30,
        -365,
        f"Transitions: {stats['edges']}"
    )

    write_text(
        30,
        -385,
        f"Mean surprise: "
        f"{shared_model.mean_error():.3f}"
    )

    write_text(
        30,
        -405,
        f"Total edge weight: "
        f"{stats['weight']:.2f}"
    )

    # --------------------------------------------------------
    # Agents scores
    # --------------------------------------------------------

    for i, agent in enumerate(
        agents
    ):

        write_text(
            -550,
            290 - i * 20,
            f"Agent {i}  "
            f"Score={agent['score']}",
            11,
            agent_colors[i]
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    write_text(
        30,
        290,
        "BLUE = stable event",
        10,
        "#66ccff"
    )

    write_text(
        30,
        272,
        "YELLOW = surprising",
        10,
        "#ffcc33"
    )

    write_text(
        30,
        254,
        "RED = highly surprising",
        10,
        "#ff3355"
    )


# ============================================================
# 13. Detect high-level causal patterns
# ============================================================

def inspect_model():

    print()
    print("=" * 70)
    print("INTERNAL WORLD MODEL")
    print("=" * 70)

    stats = (
        shared_hippocampus
        .graph_statistics()
    )

    print(
        f"Events       : {stats['nodes']}"
    )

    print(
        f"Transitions  : {stats['edges']}"
    )

    print(
        f"Edge weight  : {stats['weight']:.2f}"
    )

    print(
        f"Mean surprise: "
        f"{shared_model.mean_error():.3f}"
    )

    print()

    ranked = sorted(
        shared_hippocampus.cells,
        key=lambda c: c.importance(),
        reverse=True
    )

    for cell in ranked[:10]:

        print(
            f"Event {cell.id:3d} "
            f"action={ACTION_NAMES[cell.action]:5s} "
            f"visits={cell.visits:3d} "
            f"surprise={cell.surprise:7.3f} "
            f"energy={cell.energy:5.2f}"
        )

        for action, targets in cell.links.items():

            strongest = sorted(
                targets.items(),
                key=lambda kv: kv[1],
                reverse=True
            )[:3]

            for target_id, weight in strongest:

                print(
                    f"    "
                    f"--{ACTION_NAMES[action]}--> "
                    f"Event {target_id} "
                    f"[{weight:.2f}]"
                )

    print("=" * 70)


# ============================================================
# 14. Episode lifecycle
# ============================================================

current_step = 0
episode = 0


def reset_episode():

    global bridge_active
    global orbs

    bridge_active = False

    # Reset world
    orbs = [
        np.array(
            [-470.0, -30.0]
        ),
        np.array(
            [-200.0, 60.0]
        ),
        np.array(
            [-20.0, 80.0]
        )
    ]

    moving_platform["time"] = 0.0

    draw_world()
    draw_orbs()

    for agent in agents:

        agent["turtle"].goto(
            -520,
            0
        )

        agent["x"] = -520.0
        agent["h"] = -40.0
        agent["vh"] = 0.0
        agent["is_grounded"] = True
        agent["score"] = 0

    # Important:
    # world model persists,
    # episodic temporal pointers do not.
    shared_model.reset_agents()


# ============================================================
# 15. Main loop
# ============================================================

def run_lifecycle():

    global current_step
    global episode

    # --------------------------------------------------------
    # Normal interaction
    # --------------------------------------------------------

    if current_step < STEPS_PER_EPISODE:

        moving_platform["time"] += 0.06

        mx = (
            moving_platform["center_x"]
            + 30
            * np.sin(
                moving_platform["time"]
            )
        )

        moving_drawer.clear()

        moving_drawer.color(
            "#00ff7f"
        )

        moving_drawer.goto(
            mx - moving_platform["width"] / 2,
            moving_platform["y"]
        )

        moving_drawer.pendown()

        moving_drawer.goto(
            mx + moving_platform["width"] / 2,
            moving_platform["y"]
        )

        moving_drawer.penup()

        # ----------------------------------------------------
        # Each agent acts.
        # ----------------------------------------------------

        for agent in agents:

            agent_id = agent["id"]

            t = agent["turtle"]

            state = get_local_sensors(
                agent["x"],
                agent["h"],
                t.heading(),
                agent["is_grounded"]
            )

            # ------------------------------------------------
            # Action selection
            #
            # Still exploratory, but surprise-biased.
            # ------------------------------------------------

            action = random.choice(
                ACTIONS
            )

            if (
                shared_hippocampus.cells
                and random.random() < 0.65
            ):

                action_scores = {}

                for candidate in ACTIONS:

                    (
                        expected,
                        predicted_surprise,
                        matched_cell
                    ) = shared_model.predict(
                        state,
                        candidate
                    )

                    if (
                        predicted_surprise
                        is None
                    ):

                        # Unknown actions are
                        # slightly attractive.
                        score = 2.0

                    else:

                        # Exploration pressure.
                        score = (
                            predicted_surprise
                            + random.random()
                            * 0.5
                        )

                    action_scores[
                        candidate
                    ] = score

                action = max(
                    action_scores,
                    key=action_scores.get
                )

            # ------------------------------------------------
            # Prediction BEFORE action.
            # ------------------------------------------------

            (
                expected_change,
                predicted_surprise,
                predicted_cell
            ) = shared_model.predict(
                state,
                action
            )

            if expected_change is None:

                expected_change = np.zeros(
                    len(state)
                )

            # ------------------------------------------------
            # Execute action.
            # ------------------------------------------------

            triggered = physics_update(
                agent,
                action
            )

            # ------------------------------------------------
            # Observe result.
            # ------------------------------------------------

            next_state = get_local_sensors(
                agent["x"],
                agent["h"],
                t.heading(),
                agent["is_grounded"]
            )

            actual_change = (
                next_state - state
            )

            # ------------------------------------------------
            # Prediction error.
            # ------------------------------------------------

            prediction_error = np.linalg.norm(
                expected_change
                - actual_change
            )

            if triggered:

                prediction_error += 8.0

            # ------------------------------------------------
            # Store actual event.
            # ------------------------------------------------

            shared_model.learn(
                agent_id,
                state,
                action,
                expected_change,
                actual_change,
                prediction_error
            )

        # ----------------------------------------------------
        # Visualize.
        # ----------------------------------------------------

        draw_internal_model()

        draw_statistics(
            episode,
            current_step
        )

        current_step += 1

        screen.update()

        screen.ontimer(
            run_lifecycle,
            35
        )

        return

    # --------------------------------------------------------
    # Episode finished
    # --------------------------------------------------------

    print(
        f"\nEpisode {episode + 1} finished."
    )

    # --------------------------------------------------------
    # Sleep
    # --------------------------------------------------------

    shared_hippocampus.counterfactual_replay(
        REPLAY_COUNT
    )

    # --------------------------------------------------------
    # Metabolism
    # --------------------------------------------------------

    shared_hippocampus.metabolize()

    # --------------------------------------------------------
    # Structural pruning
    # --------------------------------------------------------

    shared_hippocampus.prune()

    # --------------------------------------------------------
    # Print internal state.
    # --------------------------------------------------------

    inspect_model()

    episode += 1

    if episode >= MAX_EPISODES:

        print()
        print(
            "Simulation complete."
        )

        print(
            "The final object above is the "
            "learned internal transition graph."
        )

        draw_internal_model()

        draw_statistics(
            episode - 1,
            STEPS_PER_EPISODE
        )

        screen.update()

        # Leave the window open so
        # the final model can be inspected.
        return

    # --------------------------------------------------------
    # Reset environment,
    # preserve world model.
    # --------------------------------------------------------

    reset_episode()

    current_step = 0

    screen.ontimer(
        run_lifecycle,
        150
    )


# ============================================================
# 16. Start
# ============================================================

reset_episode()

screen.ontimer(
    run_lifecycle,
    100
)

turtle.done()
