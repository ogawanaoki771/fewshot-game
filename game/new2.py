import turtle
import random
import numpy as np

# =====================================================
# 1. 記憶・脳モデル (好奇心・共有海馬・予測誤差・リプレイ)
# =====================================================

class MemoryCell:
    def __init__(self, cell_id, state):
        self.id = cell_id
        self.state = np.array(state, dtype=float) # [x, y]
        
        self.velocity = np.zeros(2)
        self.direction = 0.0
        self.context = np.zeros(3)
        
        self.prediction_accuracy = 1.0
        self.age = 0
        self.stability = 1.0
        
        self.links = {}  # {action: {next_id: strength}}
        self.visits = 1
        self.energy = 1.0

    def similarity(self, state):
        d = np.linalg.norm(self.state - state)
        return np.exp(-d * 0.08) # 空間が狭いため影響範囲を少し調整


class Hippocampus:
    def __init__(self, threshold=0.75):
        self.cells = []
        self.max_id = 0
        self.threshold = threshold

    def encode(self, state):
        best = None
        max_score = -1.0
        for c in self.cells:
            s = c.similarity(state)
            if s > max_score:
                max_score = s
                best = c
                
        if max_score > self.threshold and best is not None:
            best.visits += 1
            best.energy = min(2.0, best.energy + 0.12)
            return best, False
            
        cell = MemoryCell(self.max_id, state)
        self.max_id += 1
        self.cells.append(cell)
        return cell, True

    def replay(self, n=100):
        if len(self.cells) < 2:
            return
        for _ in range(n):
            c = random.choice(self.cells)
            for action in c.links:
                for next_id in c.links[action]:
                    c.links[action][next_id] = min(2.5, c.links[action][next_id] * 1.01)
                    target = [cell for cell in self.cells if cell.id == next_id]
                    if target:
                        target[0].energy = min(2.0, target[0].energy + 0.005)

    def prune_spatial_density(self, radius=12.0): # 空間縮小に合わせて半径を調整
        if len(self.cells) < 10:
            return
        retained = []
        removed_ids = set()
        sorted_cells = sorted(self.cells, key=lambda c: (c.visits, c.energy), reverse=True)
        
        for c in sorted_cells:
            if c.id in removed_ids:
                continue
            retained.append(c)
            for other in sorted_cells:
                if other.id != c.id and other.id not in removed_ids:
                    if np.linalg.norm(c.state - other.state) < radius:
                        removed_ids.add(other.id)
                        
        self.cells = retained

    def metabolize(self):
        self.cells = [c for c in self.cells if c.energy > 0.2]
        for c in self.cells:
            c.energy -= 0.012


class Cortex:
    def __init__(self, hippocampus):
        self.hippo = hippocampus
        self.base_lr = 0.3

    def learn(self, s1, action, s2, prediction_error, current_epoch, max_epochs):
        c1, _ = self.hippo.encode(s1)
        c2, _ = self.hippo.encode(s2)
        if action not in c1.links:
            c1.links[action] = {}
        if c2.id not in c1.links[action]:
            c1.links[action][c2.id] = 0.25
        else:
            c1.links[action][c2.id] = min(2.2, c1.links[action][c2.id] + 0.2)
            
        decay_factor = 1.0 - (current_epoch / (max_epochs * 1.2))
        effective_base_lr = self.base_lr * max(0.4, decay_factor)
        
        adaptive_lr = min(0.85, effective_base_lr + prediction_error * 0.04)
        for a in c1.links:
            for next_id in c1.links[a]:
                c1.links[a][next_id] += adaptive_lr * 0.01

    def predict(self, state, action):
        if not self.hippo.cells:
            return None
        scores = [c.similarity(state) for c in self.hippo.cells]
        best_idx = np.argmax(scores)
        if scores[best_idx] < 0.4 or action not in self.hippo.cells[best_idx].links:
            return None
        
        best_c = self.hippo.cells[best_idx]
        candidates = list(best_c.links[action].keys())
        if not candidates:
            return None
        weights = np.array([best_c.links[action][x] for x in candidates])
        if weights.sum() == 0:
            return None
        weights = weights / weights.sum()
        next_id = np.random.choice(candidates, p=weights)
        target = [c for c in self.hippo.cells if c.id == next_id]
        return target[0].state if target else None


class Curiosity:
    def __init__(self):
        self.history = []

    def evaluate(self, error):
        curiosity_val = np.tanh(error * 0.1)
        self.history.append(curiosity_val)
        return curiosity_val


# =====================================================
# 2. 画面・コンパクトな迷路＆開閉扉（スイッチ）設定
# =====================================================

screen = turtle.Screen()
screen.setup(width=600, height=600)
screen.bgcolor("#0b0b12")
screen.title("Compact Athletic World Model: Dynamic Doors & Tight Spaces")
screen.tracer(0)

wall_drawer = turtle.Turtle()
wall_drawer.hideturtle()
wall_drawer.penup()
wall_drawer.speed(0)
wall_drawer.color("#ffffff")
wall_drawer.pensize(3)

# 2つのトンネルの開閉状態を管理するフラグ（コンパクト化に伴い2箇所に簡略化）
tunnel_states = [True, True]

def draw_maze():
    wall_drawer.clear()
    # 外枠（コンパクトに縮小：±180範囲）
    wall_drawer.penup()
    wall_drawer.goto(-180, -180)
    wall_drawer.pendown()
    for _ in range(4):
        wall_drawer.forward(360)
        wall_drawer.left(90)
        
    # 中央の壁
    wall_drawer.penup()
    wall_drawer.goto(0, -180)
    wall_drawer.setheading(90)
    wall_drawer.pendown()
    wall_drawer.forward(360)

    # 開いているトンネル部分を消去（背景色で上書き）
    eraser = turtle.Turtle()
    eraser.hideturtle()
    eraser.penup()
    eraser.speed(0)
    eraser.color("#0b0b12")
    eraser.pensize(5)
    
    tunnels = [
        (-100, -30, tunnel_states[0]), # トンネル1
        (30, 100,    tunnel_states[1])  # トンネル2
    ]
    
    for y_start, y_end, is_open in tunnels:
        if is_open:
            eraser.penup()
            eraser.goto(0, y_start)
            eraser.setheading(90)
            eraser.pendown()
            eraser.forward(y_end - y_start)

draw_maze()

# スイッチゲート（触れると対応する扉の開閉が反転する）
gates = [
    {"pos": np.array([-90.0, 120.0]), "target_tunnel": 0},
    {"pos": np.array([90.0, -120.0]), "target_tunnel": 1}
]
gate_drawers = []
for g in gates:
    gd = turtle.Turtle()
    gd.hideturtle()
    gd.penup()
    gd.speed(0)
    gd.color("#00ffea")
    gd.shape("square")
    gate_drawers.append(gd)

# 動的障害物（狭い空間に合わせてサイズ・移動量を調整）
obstacles = [
    {"pos": np.array([-60.0, 60.0]), "vel": np.array([1.8, 1.4])},
    {"pos": np.array([60.0, -60.0]), "vel": np.array([-1.8, -1.4])}
]
obs_drawers = []
for _ in obstacles:
    od = turtle.Turtle()
    od.hideturtle()
    od.penup()
    od.speed(0)
    od.color("#ff4d4d")
    od.shape("square")
    obs_drawers.append(od)

landscape_drawer = turtle.Turtle()
landscape_drawer.hideturtle()
landscape_drawer.penup()
landscape_drawer.speed(0)

# 共有海馬
shared_hippocampus = Hippocampus(threshold=0.75)
shared_cortex = Cortex(shared_hippocampus)

# 複数エージェント（3体）
colors = ["#00ffff", "#00ff7f", "#ffa500"]
start_positions = [(-100, 0), (-100, -30), (-100, 30)]
agents = []

for i in range(3):
    t = turtle.Turtle()
    t.shape("turtle")
    t.color(colors[i])
    t.penup()
    t.speed(0)
    t.goto(start_positions[i])
    
    agents.append({
        "turtle": t,
        "hippo": shared_hippocampus,
        "cortex": shared_cortex,
        "curiosity": Curiosity(),
        "color": colors[i]
    })


# =====================================================
# 3. 衝突・センサー・ゲート（扉の開閉）判定
# =====================================================

def check_collision(x, y):
    if abs(x) > 175 or abs(y) > 175:
        return True
    
    if -4 <= x <= 4:
        in_t1 = (-100 <= y <= -30) and tunnel_states[0]
        in_t2 = (30 <= y <= 100) and tunnel_states[1]
        
        if not (in_t1 or in_t2):
            return True
            
    for obs in obstacles:
        if np.linalg.norm(np.array([x, y]) - obs["pos"]) < 20:
            return True
    return False

def check_gates(x, y):
    global tunnel_states
    for i, g in enumerate(gates):
        if np.linalg.norm(np.array([x, y]) - g["pos"]) < 20:
            tid = g["target_tunnel"]
            tunnel_states[tid] = not tunnel_states[tid]
            gate_drawers[i].color("#ff00ff" if tunnel_states[tid] else "#555555")
            draw_maze()
            return True
    return False

def get_obstacle_sensors(x, y, heading):
    sensor_dist = 25.0
    angles = [heading + 35, heading, heading - 35]
    readings = []
    for ang in angles:
        rad = np.radians(ang)
        sx = x + sensor_dist * np.cos(rad)
        sy = y + sensor_dist * np.sin(rad)
        readings.append(1.0 if check_collision(sx, sy) else 0.0)
    return readings

def draw_memory_landscape():
    landscape_drawer.clear()
    for c in shared_hippocampus.cells:
        landscape_drawer.goto(c.state[0], c.state[1])
        landscape_drawer.color("#ffffff")
        size = max(2, min(4, int(c.visits * 0.4 + 2)))
        landscape_drawer.dot(size)


# =====================================================
# 4. メインライフサイクル
# =====================================================

steps_per_episode = 140
current_step = 0
episode = 0
max_episodes = 15

def run_lifecycle():
    global current_step, episode
    
    if current_step < steps_per_episode:
        for i, g in enumerate(gates):
            gate_drawers[i].clear()
            gate_drawers[i].goto(g["pos"][0], g["pos"][1])
            gate_drawers[i].stamp()

        for i, obs in enumerate(obstacles):
            obs["pos"] += obs["vel"]
            if abs(obs["pos"][0]) > 150:
                obs["vel"][0] = -obs["vel"][0]
            if abs(obs["pos"][1]) > 150:
                obs["vel"][1] = -obs["vel"][1]
            
            obs_drawers[i].clear()
            obs_drawers[i].goto(obs["pos"][0], obs["pos"][1])
            obs_drawers[i].stamp()
            
        for agent_data in agents:
            t = agent_data["turtle"]
            hippo = agent_data["hippo"]
            cortex = agent_data["cortex"]
            curiosity = agent_data["curiosity"]
            
            pos = t.pos()
            state = [pos[0], pos[1]]
            heading = t.heading()
            
            memory_action = random.choice([0, 1, 2])
            if hippo.cells:
                scores = [c.similarity(state) for c in hippo.cells]
                best_idx = np.argmax(scores)
                best_c = hippo.cells[best_idx]
                if best_c.links:
                    memory_action = random.choice(list(best_c.links.keys()))

            if random.random() < 0.4:
                action = random.choice([0, 1, 2])
            else:
                action = memory_action
            
            predicted_next = cortex.predict(state, action)
            
            t.pendown()
            t.forward(5) # 狭い空間に合わせて歩幅を微調整
            t.penup()
            
            if action == 1:
                t.left(35)
            elif action == 2:
                t.right(35)
                
            new_pos = t.pos()
            
            if check_collision(new_pos[0], new_pos[1]):
                t.backward(10)
                t.left(random.randint(80, 160))
                new_pos = t.pos()
                
            check_gates(new_pos[0], new_pos[1])

            next_state = [new_pos[0], new_pos[1]]
            
            if predicted_next is not None:
                pred_error = np.linalg.norm(predicted_next - np.array(next_state))
            else:
                pred_error = 10.0
            
            curiosity.evaluate(pred_error)
            cortex.learn(state, action, next_state, pred_error, episode, max_episodes)
            
        draw_memory_landscape()
        current_step += 1
        screen.update()
        screen.ontimer(run_lifecycle, 20)
        
    else:
        print(f"--- Episode {episode+1} Finished (Replay & Sleep) ---")
        for agent_data in agents:
            agent_data["turtle"].clear()
            
        global tunnel_states
        tunnel_states = [random.choice([True, False]) for _ in range(2)]
        draw_maze()
                
        shared_hippocampus.replay(200)
        shared_hippocampus.prune_spatial_density(radius=12.0)
        shared_hippocampus.metabolize()
        
        current_step = 0
        episode += 1
        if episode < max_episodes:
            screen.ontimer(run_lifecycle, 100)
        else:
            print("Simulation Complete. Compact Athletic World Model Formed.")
            screen.bye()

screen.ontimer(run_lifecycle, 100)
turtle.done()
