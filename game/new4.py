import turtle
import random
import numpy as np

# =====================================================
# 1. 出来事記憶セル (Event Cell: 状況・行動・結果の因果を刻む)
# =====================================================

class EventCell:
    def __init__(self, cell_id, context, action, expected, actual, surprise):
        self.id = cell_id
        self.context = np.array(context, dtype=float)     # 実行直前の局所身体・世界状況
        self.action = action                              # 無名の行動 (0, 1, 2, 3)
        self.expected = np.array(expected, dtype=float)   # 予測した変化
        self.actual = np.array(actual, dtype=float)       # 実際に起きた変化
        
        self.surprise = float(surprise)                   # 予測誤差（驚き）
        self.energy = 1.0                                 # セルの生命力・重要度
        self.visits = 1
        
        # リンク構造: {next_action: {next_cell_id: strength}}
        self.links = {}

    def similarity(self, context, action):
        if self.action != action:
            return 0.0
        d = np.linalg.norm(self.context - context)
        return np.exp(-d * 0.1)


class Hippocampus:
    def __init__(self, threshold=0.75):
        self.cells = []
        self.max_id = 0
        self.threshold = threshold

    def encode(self, context, action, expected, actual, surprise):
        best = None
        max_score = -1.0
        for c in self.cells:
            s = c.similarity(context, action)
            if s > max_score:
                max_score = s
                best = c
                
        if max_score > self.threshold and best is not None:
            best.visits += 1
            best.energy = min(2.0, best.energy + 0.1)
            best.surprise = 0.8 * best.surprise + 0.2 * surprise
            best.actual = 0.7 * best.actual + 0.3 * actual
            return best, False
            
        cell = EventCell(self.max_id, context, action, expected, actual, surprise)
        self.max_id += 1
        self.cells.append(cell)
        return cell, True

    def counterfactual_replay(self, n=150):
        """
        夜の睡眠フェーズ：反実仮想シミュレーション
        実際には起きていない可能性や、過去の経験の組み合わせを脳内で試す
        """
        if len(self.cells) < 2:
            return
        for _ in range(n):
            c1 = random.choice(self.cells)
            # 仮想的に別のアクションを選んだ場合のシナリオを強化
            alt_action = random.choice([0, 1, 2, 3])
            if alt_action in c1.links:
                for next_id in c1.links[alt_action]:
                    c1.links[alt_action][next_id] = min(2.5, c1.links[alt_action][next_id] * 1.02)
                    target = [cell for cell in self.cells if cell.id == next_id]
                    if target:
                        target[0].energy = min(2.0, target[0].energy + 0.01)

    def prune(self, radius=12.0):
        if len(self.cells) < 15:
            return
        retained = []
        removed_ids = set()
        sorted_cells = sorted(self.cells, key=lambda c: (c.visits, c.energy, c.surprise), reverse=True)
        
        for c in sorted_cells:
            if c.id in removed_ids:
                continue
            retained.append(c)
            for other in sorted_cells:
                if other.id != c.id and other.id not in removed_ids:
                    if np.linalg.norm(c.context - other.context) < radius and c.action == other.action:
                        removed_ids.add(other.id)
                        
        self.cells = retained

    def metabolize(self):
        self.cells = [c for c in self.cells if c.energy > 0.25]
        for c in self.cells:
            c.energy -= 0.01


class WorldAndBodyModel:
    def __init__(self, hippocampus):
        self.hippo = hippocampus
        self.base_lr = 0.3

    def learn(self, c1, action, expected, actual, surprise, current_epoch, max_epochs):
        c2, _ = self.hippo.encode(c1, action, expected, actual, surprise)
        
        if action not in c1.links if isinstance(c1, EventCell) else False:
            pass # リンク更新の処理
        
        # 実際のエピソード間リンクの構築
        # （簡易的に最新のセル同士を接続）
        if len(self.hippo.cells) >= 2:
            prev_cell = self.hippo.cells[-2]
            curr_cell = self.hippo.cells[-1]
            if action not in prev_cell.links:
                prev_cell.links[action] = {}
            prev_cell.links[action][curr_cell.id] = 0.4

    def predict(self, context, action):
        if not self.hippo.cells:
            return None, None
        scores = [c.similarity(context, action) for c in self.hippo.cells]
        best_idx = np.argmax(scores)
        if scores[best_idx] < 0.35:
            return None, None
        
        best_c = self.hippo.cells[best_idx]
        return best_c.expected, best_c.surprise


# =====================================================
# 2. 外部環境（物理シミュレータ）
# =====================================================

screen = turtle.Screen()
screen.setup(width=600, height=600)
screen.bgcolor("#0b0b12")
screen.title("Embodied Self-Organizing Agent: World & Body Model")
screen.tracer(0)

world_drawer = turtle.Turtle()
world_drawer.hideturtle()
world_drawer.penup()
world_drawer.speed(0)
world_drawer.pensize(3)

static_platforms = [
    [-180, -110, -50],  
    [110,   180,  40]   
]

bridge_active = False
switch_pos = np.array([-40.0, -50.0])

def draw_world():
    world_drawer.clear()
    world_drawer.color("#333344")
    world_drawer.penup()
    world_drawer.goto(-180, -180)
    world_drawer.pendown()
    for _ in range(4):
        world_drawer.forward(360)
        world_drawer.left(90)

    world_drawer.color("#00ffea")
    for p in static_platforms:
        world_drawer.penup()
        world_drawer.goto(p[0], p[2])
        world_drawer.pendown()
        world_drawer.goto(p[1], p[2])

    switch_drawer = turtle.Turtle()
    switch_drawer.hideturtle()
    switch_drawer.penup()
    switch_drawer.goto(switch_pos[0], switch_pos[1])
    switch_drawer.color("#ff00ff" if bridge_active else "#555555")
    switch_drawer.shape("square")
    switch_drawer.stamp()

    if bridge_active:
        world_drawer.color("#ffaa00")
        world_drawer.penup()
        world_drawer.goto(-90, -50)
        world_drawer.pendown()
        world_drawer.goto(0, -50)

draw_world()

moving_platform = {
    "center_x": 0.0,
    "y": 10.0,
    "width": 50.0,
    "time": 0.0
}

moving_drawer = turtle.Turtle()
moving_drawer.hideturtle()
moving_drawer.penup()
moving_drawer.speed(0)
moving_drawer.color("#00ff7f")

landscape_drawer = turtle.Turtle()
landscape_drawer.hideturtle()
landscape_drawer.penup()
landscape_drawer.speed(0)

shared_hippocampus = Hippocampus(threshold=0.75)
shared_model = WorldAndBodyModel(shared_hippocampus)

colors = ["#00ffff", "#00ff7f", "#ffa500"]
agents = []

for i in range(3):
    t = turtle.Turtle()
    t.shape("turtle")
    t.color(colors[i])
    t.penup()
    t.speed(0)
    t.goto(-150, 0)
    
    agents.append({
        "turtle": t,
        "x": -150.0,
        "y": 0.0,
        "h": -50.0,
        "vh": 0.0,
        "is_grounded": True,
        "hippo": shared_hippocampus,
        "model": shared_model,
        "color": colors[i]
    })


# =====================================================
# 3. 局所感覚と身体モデル
# =====================================================

def get_local_sensors(x, h, heading, is_grounded):
    rad = np.radians(heading)
    front_x = x + 20 * np.cos(rad)
    
    has_floor = 0.0
    for p in static_platforms:
        if p[0] <= front_x <= p[1] and abs(h - p[2]) < 12:
            has_floor = 1.0
    if bridge_active and -90 <= front_x <= 0 and abs(h - (-50)) < 12:
        has_floor = 1.0
    mx = moving_platform["center_x"] + 30 * np.sin(moving_platform["time"])
    if (mx - 25) <= front_x <= (mx + 25) and abs(h - moving_platform["y"]) < 14:
        has_floor = 1.0

    switch_dist = np.linalg.norm(np.array([x, h]) - switch_pos)
    near_switch = 1.0 if switch_dist < 20 else 0.0

    return np.array([
        has_floor,
        1.0 if abs(front_x) < 175 else 0.0,
        1.0 if is_grounded else 0.0,
        near_switch,
        h / 50.0
    ], dtype=float)


def physics_update(agent_data, action):
    global bridge_active
    t = agent_data["turtle"]
    x, y, h, vh = agent_data["x"], agent_data["y"], agent_data["h"], agent_data["vh"]
    is_grounded = agent_data["is_grounded"]
    
    triggered = False

    if action == 0:
        t.forward(6)
    elif action == 1:
        t.left(30)
    elif action == 2:
        t.right(30)
    elif action == 3 and is_grounded:
        vh = 6.5
        is_grounded = False

    if not is_grounded:
        vh -= 0.6
        h += vh

    new_pos = t.pos()
    x, y = new_pos[0], new_pos[1]

    if np.linalg.norm(np.array([x, h]) - switch_pos) < 18:
        bridge_active = not bridge_active
        draw_world()
        triggered = True

    target_h = None
    for p in static_platforms:
        if p[0] <= x <= p[1] and abs(h - p[2]) < 12:
            target_h = p[2]

    if bridge_active and -90 <= x <= 0 and abs(h - (-50)) < 12:
        target_h = -50

    mx = moving_platform["center_x"] + 30 * np.sin(moving_platform["time"])
    if (mx - 25) <= x <= (mx + 25) and abs(h - moving_platform["y"]) < 14:
        target_h = moving_platform["y"]

    if abs(x) > 175:
        t.backward(15)
        t.left(random.randint(90, 180))
        x, y = t.pos()
        target_h = -40
        is_grounded = True
        vh = 0.0

    if target_h is not None and h <= target_h + 4:
        h = target_h
        is_grounded = True
        vh = 0.0
    elif target_h is None and h < -170:
        t.goto(-150, 0)
        x, y = -150, 0
        h = -40
        is_grounded = True
        vh = 0.0
    elif target_h is None:
        is_grounded = False

    agent_data["x"] = x
    agent_data["y"] = y
    agent_data["h"] = h
    agent_data["vh"] = vh
    agent_data["is_grounded"] = is_grounded

    return triggered


# =====================================================
# 4. メインライフサイクル (昼の探索 ⇄ 夜の睡眠)
# =====================================================

steps_per_episode = 160
current_step = 0
episode = 0
max_episodes = 15

def run_lifecycle():
    global current_step, episode
    
    if current_step < steps_per_episode:
        moving_platform["time"] += 0.05
        mx = moving_platform["center_x"] + 30 * np.sin(moving_platform["time"])
        
        moving_drawer.clear()
        moving_drawer.penup()
        moving_drawer.goto(mx - moving_platform["width"]/2, moving_platform["y"])
        moving_drawer.pendown()
        moving_drawer.goto(mx + moving_platform["width"]/2, moving_platform["y"])

        for agent_data in agents:
            t = agent_data["turtle"]
            hippo = agent_data["hippo"]
            model = agent_data["model"]
            
            # 1. 現在の感覚状態を取得
            state = get_local_sensors(agent_data["x"], agent_data["h"], t.heading(), agent_data["is_grounded"])
            
            # 2. 好奇心（未知の探索）と既知のトレードオフによる行動選択
            action = random.choice([0, 1, 2, 3])
            if hippo.cells and random.random() < 0.6:
                # 予測誤差（驚き）が高い状態をもたらしそうな行動を優先的に探索する
                action_surprises = {a: 0.0 for a in range(4)}
                for a in range(4):
                    _, pred_surprise = model.predict(state, a)
                    if pred_surprise is not None:
                        action_surprises[a] = pred_surprise
                # 好奇心価値が一番高そうな行動を選ぶ（または確率的選択）
                action = max(action_surprises, key=action_surprises.get)

            expected_change, _ = model.predict(state, action)
            if expected_change is None:
                expected_change = np.zeros(5)

            # 3. 身体行動の実行
            triggered = physics_update(agent_data, action)
            
            # 4. 次の状態と実際の変化量を観測
            next_state = get_local_sensors(agent_data["x"], agent_data["h"], t.heading(), agent_data["is_grounded"])
            actual_change = next_state - state
            
            # 5. 予測誤差（驚き）の計算
            pred_error = np.linalg.norm(expected_change - actual_change)
            if triggered:
                pred_error += 20.0 # 因果発動のスパイク
            
            # 6. 出来事セルとしての学習・記憶
            model.learn(state, action, expected_change, actual_change, pred_error, episode, max_episodes)
            
        landscape_drawer.clear()
        for c in shared_hippocampus.cells:
            landscape_drawer.goto(c.context[0] * 50, c.context[4] * 50)
            landscape_drawer.color("#ff00ff" if c.surprise > 4.0 else "#ffffff")
            size = max(2, min(5, int(c.visits * 0.3 + 2)))
            landscape_drawer.dot(size)

        current_step += 1
        screen.update()
        screen.ontimer(run_lifecycle, 25)
        
    else:
        print(f"--- Episode {episode+1} Finished (Sleep & Counterfactual Simulation) ---")
        for agent_data in agents:
            agent_data["turtle"].clear()
            agent_data["turtle"].goto(-150, 0)
            agent_data["x"] = -150.0
            agent_data["y"] = 0.0
            agent_data["h"] = -50.0
            agent_data["is_grounded"] = True
            agent_data["vh"] = 0.0
            
        # 夜の睡眠：反実仮想シミュレーションと代謝
        shared_hippocampus.counterfactual_replay(250)
        shared_hippocampus.prune(radius=12.0)
        shared_hippocampus.metabolize()
        
        current_step = 0
        episode += 1
        if episode < max_episodes:
            screen.ontimer(run_lifecycle, 100)
        else:
            print("Simulation Complete. Autonomous Embodied World Model Formed.")
            screen.bye()

screen.ontimer(run_lifecycle, 100)
turtle.done()
