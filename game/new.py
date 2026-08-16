import turtle
import random
import numpy as np

# =====================================================
# 1. 記憶・脳モデル (軌跡記憶・共有海馬・夢リプレイ)
# =====================================================

class MemoryCell:
    def __init__(self, cell_id, state):
        self.id = cell_id
        self.state = np.array(state, dtype=float) # [x, y]
        self.links = {}  # {action: {next_id: strength}}
        self.visits = 1
        self.energy = 1.0

    def similarity(self, state):
        d = np.linalg.norm(self.state - state)
        return np.exp(-d * 0.08)


class Hippocampus:
    def __init__(self, threshold=0.75):
        self.cells = []
        self.max_id = 0
        self.threshold = threshold
        self.trajectory_history = [] # エージェントが辿った軌跡（セルのID列）のバッファ

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

    def record_step(self, cell_id):
        """エージェントの移動軌跡を記録"""
        if not self.trajectory_history or self.trajectory_history[-1] != cell_id:
            self.trajectory_history.append(cell_id)
            # 軌跡が長すぎたら古いものを整理（最新300ステップを保持）
            if len(self.trajectory_history) > 300:
                self.trajectory_history.pop(0)

    def dream_and_replay(self, n=50):
        """
        【夢の再構成フェーズ】
        単なるリンク強化だけでなく、過去の軌跡の連なり（パス）を再生し、
        離れたセル同士を直接結ぶ「因果のショートカット（抽象化）」を創る。
        """
        if len(self.cells) < 4 or len(self.trajectory_history) < 5:
            return

        for _ in range(n):
            # 軌跡の中からランダムに連続する一連のサブパス（例: A → B → C → D）をサンプリング
            if len(self.trajectory_history) >= 4:
                start_idx = random.randint(0, len(self.trajectory_history) - 4)
                sub_path = self.trajectory_history[start_idx : start_idx + random.randint(3, min(6, len(self.trajectory_history) - start_idx))]
                
                if len(sub_path) >= 3:
                    # 夢による因果のショートカット：
                    # パスの最初（A）と最後（Z）を直接繋ぐ架空のリンク（ショートカット）を補強する
                    origin_id = sub_path[0]
                    destination_id = sub_path[-1]
                    
                    origin_cell = next((c for c in self.cells if c.id == origin_id), None)
                    dest_cell = next((c for c in self.cells if c.id == destination_id), None)
                    
                    if origin_cell and dest_cell and origin_id != destination_id:
                        # 仮想的な「直通アクション（疑似アクション 99）」として直接リンクを作成・強化
                        shortcut_action = 99 
                        if shortcut_action not in origin_cell.links:
                            origin_cell.links[shortcut_action] = {}
                        
                        current_strength = origin_cell.links[shortcut_action].get(destination_id, 0.1)
                        origin_cell.links[shortcut_action][destination_id] = min(3.0, current_strength + 0.35)
                        
                        # 経由したセルのエネルギーを底上げ
                        origin_cell.energy = min(2.0, origin_cell.energy + 0.05)
                        dest_cell.energy = min(2.0, dest_cell.energy + 0.05)

            # 通常のリンク強化リプレイも併用
            c = random.choice(self.cells)
            for action in c.links:
                for next_id in c.links[action]:
                    c.links[action][next_id] = min(2.5, c.links[action][next_id] * 1.015)

    def prune_spatial_density(self, radius=12.0):
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
# 2. 画面・迷路＆開閉扉の設定
# =====================================================

screen = turtle.Screen()
screen.setup(width=600, height=600)
screen.bgcolor("#0b0b12")
screen.title("Athletic World Model: Dream Trajectory Replay & Shortcut Abstraction")
screen.tracer(0)

wall_drawer = turtle.Turtle()
wall_drawer.hideturtle()
wall_drawer.penup()
wall_drawer.speed(0)
wall_drawer.color("#ffffff")
wall_drawer.pensize(3)

tunnel_states = [True, True]

def draw_maze():
    wall_drawer.clear()
    wall_drawer.penup()
    wall_drawer.goto(-180, -180)
    wall_drawer.pendown()
    for _ in range(4):
        wall_drawer.forward(360)
        wall_drawer.left(90)
        
    wall_drawer.penup()
    wall_drawer.goto(0, -180)
    wall_drawer.setheading(90)
    wall_drawer.pendown()
    wall_drawer.forward(360)

    eraser = turtle.Turtle()
    eraser.hideturtle()
    eraser.penup()
    eraser.speed(0)
    eraser.color("#0b0b12")
    eraser.pensize(5)
    
    tunnels = [
        (-100, -30, tunnel_states[0]),
        (30, 100,    tunnel_states[1])
    ]
    
    for y_start, y_end, is_open in tunnels:
        if is_open:
            eraser.penup()
            eraser.goto(0, y_start)
            eraser.setheading(90)
            eraser.pendown()
            eraser.forward(y_end - y_start)

draw_maze()

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

shared_hippocampus = Hippocampus(threshold=0.75)
shared_cortex = Cortex(shared_hippocampus)

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
# 3. 衝突・センサー・ゲート判定
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
            
            # 海馬へのエンコードと軌跡の記録
            current_cell, _ = hippo.encode(state)
            hippo.record_step(current_cell.id)
            
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
            t.forward(5)
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
        print(f"--- Episode {episode+1} Finished (Dream & Trajectory Replay) ---")
        for agent_data in agents:
            agent_data["turtle"].clear()
            
        global tunnel_states
        tunnel_states = [random.choice([True, False]) for _ in range(2)]
        draw_maze()
                
        # 睡眠中の「夢（軌跡の再構成によるショートカット生成）」
        shared_hippocampus.dream_and_replay(80)
        shared_hippocampus.prune_spatial_density(radius=12.0)
        shared_hippocampus.metabolize()
        
        current_step = 0
        episode += 1
        if episode < max_episodes:
            screen.ontimer(run_lifecycle, 100)
        else:
            print("Simulation Complete. Dream-Abstracted World Model Formed.")
            screen.bye()

screen.ontimer(run_lifecycle, 100)
turtle.done()
