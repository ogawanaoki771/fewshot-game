import turtle
import random
import numpy as np

# =====================================================
# 1. 記憶・脳モデル (「場所」から「経験の断片」への進化)
# =====================================================

class MemoryCell:
    def __init__(self, cell_id, state, action, pred_error):
        self.id = cell_id
        self.state = np.array(state, dtype=float) # [x, y, h, vx, vy, vh]
        self.action = action                      # この経験を引き起こした、あるいは結ばれたアクション
        
        self.prediction_accuracy = 1.0
        self.surprise = float(pred_error)         # 予測誤差（驚き）
        self.age = 0
        self.stability = 1.0
        
        self.links = {}  # {action: {next_id: strength}}
        self.visits = 1
        self.energy = 1.0

    def similarity(self, state):
        # 位置(0,1)と高さ(2)、速度などを考慮した状態類似度
        d = np.linalg.norm(self.state[:3] - state[:3])
        return np.exp(-d * 0.05)


class Hippocampus:
    def __init__(self, threshold=0.75):
        self.cells = []
        self.max_id = 0
        self.threshold = threshold

    def encode(self, state, action, pred_error):
        best = None
        max_score = -1.0
        for c in self.cells:
            s = c.similarity(state)
            if s > max_score:
                max_score = s
                best = c
                
        if max_score > self.threshold and best is not None:
            best.visits += 1
            best.energy = min(2.0, best.energy + 0.1)
            best.surprise = 0.8 * best.surprise + 0.2 * pred_error
            return best, False
            
        cell = MemoryCell(self.max_id, state, action, pred_error)
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

    def prune_spatial_density(self, radius=15.0):
        if len(self.cells) < 10:
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
                    if np.linalg.norm(c.state[:3] - other.state[:3]) < radius:
                        removed_ids.add(other.id)
                        
        self.cells = retained

    def metabolize(self):
        self.cells = [c for c in self.cells if c.energy > 0.2]
        for c in self.cells:
            c.energy -= 0.01


class Cortex:
    def __init__(self, hippocampus):
        self.hippo = hippocampus
        self.base_lr = 0.3

    def learn(self, s1, action, s2, pred_error, current_epoch, max_epochs):
        c1, _ = self.hippo.encode(s1, action, pred_error)
        c2, _ = self.hippo.encode(s2, action, pred_error)
        
        if action not in c1.links:
            c1.links[action] = {}
        if c2.id not in c1.links[action]:
            c1.links[action][c2.id] = 0.3
        else:
            c1.links[action][c2.id] = min(2.5, c1.links[action][c2.id] + 0.2)
            
        decay_factor = 1.0 - (current_epoch / (max_epochs * 1.2))
        effective_base_lr = self.base_lr * max(0.4, decay_factor)
        adaptive_lr = min(0.85, effective_base_lr + pred_error * 0.03)
        
        for a in c1.links:
            for next_id in c1.links[a]:
                c1.links[a][next_id] += adaptive_lr * 0.01

    def predict(self, state, action):
        if not self.hippo.cells:
            return None
        scores = [c.similarity(state) for c in self.hippo.cells]
        best_idx = np.argmax(scores)
        if scores[best_idx] < 0.35 or action not in self.hippo.cells[best_idx].links:
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
        val = np.tanh(error * 0.08)
        self.history.append(val)
        return val


# =====================================================
# 2. 画面・アスレチック環境（足場・高低差・穴・重力）設定
# =====================================================

screen = turtle.Screen()
screen.setup(width=600, height=600)
screen.bgcolor("#0b0b12")
screen.title("Athletic World Model: Gravity, Gaps & Body Discovery")
screen.tracer(0)

world_drawer = turtle.Turtle()
world_drawer.hideturtle()
world_drawer.penup()
world_drawer.speed(0)
world_drawer.pensize(3)

# アスレチックステージの定義（高低差とギャップ・穴）
# 各プラットフォーム: [x_start, x_end, height_y]
platforms = [
    [-180, -90, -50],   # 開始足場
    [-70,  20,   0],    # 中央低めの足場 (間にギャップ)
    [40,   130,  50],   # 高めの足場 (さらにギャップ)
    [140,  180,  0]     # ゴール方向の足場
]

def draw_athletic_world():
    world_drawer.clear()
    
    # 外枠
    world_drawer.color("#333344")
    world_drawer.penup()
    world_drawer.goto(-180, -180)
    world_drawer.pendown()
    for _ in range(4):
        world_drawer.forward(360)
        world_drawer.left(90)

    # プラットフォーム（足場）の描画
    world_drawer.color("#00ffea")
    for p in platforms:
        world_drawer.penup()
        world_drawer.goto(p[0], p[2])
        world_drawer.pendown()
        world_drawer.goto(p[1], p[2])
        
        # 支柱の描画
        world_drawer.penup()
        world_drawer.goto(p[0], p[2])
        world_drawer.pendown()
        world_drawer.goto(p[0], -180)
        world_drawer.penup()
        world_drawer.goto(p[1], p[2])
        world_drawer.pendown()
        world_drawer.goto(p[1], -180)

draw_athletic_world()

landscape_drawer = turtle.Turtle()
landscape_drawer.hideturtle()
landscape_drawer.penup()
landscape_drawer.speed(0)

shared_hippocampus = Hippocampus(threshold=0.75)
shared_cortex = Cortex(shared_hippocampus)

# 複数エージェント (3体)
colors = ["#00ffff", "#00ff7f", "#ffa500"]
agents = []

for i in range(3):
    t = turtle.Turtle()
    t.shape("turtle")
    t.color(colors[i])
    t.penup()
    t.speed(0)
    # 初期位置は最初の足場の上
    t.goto(-150, -40)
    
    agents.append({
        "turtle": t,
        "h": 0.0,            # 高さ方向の位置オフセット
        "vh": 0.0,           # 垂直速度 (重力・ジャンプ用)
        "is_grounded": True, # 接地フラグ
        "hippo": shared_hippocampus,
        "cortex": shared_cortex,
        "curiosity": Curiosity(),
        "color": colors[i]
    })


# =====================================================
# 3. 物理演算・判定（重力・足場判定・衝突）
# =====================================================

def get_current_platform(x, current_y):
    """ 現在のx座標と高さにおいて、足場の上にいるか判定する """
    for p in platforms:
        if p[0] <= x <= p[1]:
            # 足場のY座標付近にいるか
            if abs(current_y - p[2]) < 12:
                return p[2]
    return None

def check_physics_and_collision(x, y, h):
    """ 壁との衝突や落下判定を行う """
    if abs(x) > 175:
        return True, y
    
    # 足場の上空にいるかどうか
    ground_y = get_current_platform(x, y)
    if ground_y is not None:
        # 足場に着地
        return False, ground_y
    else:
        # 足場がない場所（穴・ギャップ） -> 落下する
        return False, y


# =====================================================
# 4. メインライフサイクル（身体運動と学習）
# =====================================================

steps_per_episode = 160
current_step = 0
episode = 0
max_episodes = 15

def run_lifecycle():
    global current_step, episode
    
    if current_step < steps_per_episode:
        for agent_data in agents:
            t = agent_data["turtle"]
            hippo = agent_data["hippo"]
            cortex = agent_data["cortex"]
            curiosity = agent_data["curiosity"]
            
            pos = t.pos()
            x, y = pos[0], pos[1]
            h = agent_data["h"]
            vh = agent_data["vh"]
            
            state = [x, y, h, t.heading(), vh, 1.0 if agent_data["is_grounded"] else 0.0]
            
            # 行動選択: [0: 前進, 1: 左旋回, 2: 右旋回, 3: ジャンプ]
            # 最初はランダム、徐々に内部モデルの予測・記憶に基づく
            action = random.choice([0, 1, 2, 3])
            if hippo.cells and random.random() < 0.5:
                scores = [c.similarity(state) for c in hippo.cells]
                best_idx = np.argmax(scores)
                best_c = hippo.cells[best_idx]
                if best_c.links:
                    action = random.choice(list(best_c.links.keys()))

            predicted_next = cortex.predict(state, action)
            
            # 身体アクションの実行
            t.pendown()
            if action == 0:
                t.forward(6)
            elif action == 1:
                t.left(30)
            elif action == 2:
                t.right(30)
            elif action == 3 and agent_data["is_grounded"]:
                # ジャンプ実行（垂直上向きの初速を与える）
                vh = 6.5
                agent_data["is_grounded"] = False
            t.penup()
            
            # 重力・垂直方向の物理演算
            if not agent_data["is_grounded"]:
                vh -= 0.6 # 重力加速度
                y += vh
            
            new_pos = t.pos()
            nx, ny = new_pos[0], new_pos[1]
            
            # 足場・境界判定
            is_wall, target_y = check_physics_and_collision(nx, ny, y)
            
            if is_wall:
                t.backward(12)
                t.left(random.randint(90, 180))
                nx, ny = t.pos()
                target_y = get_current_platform(nx, ny) or ny
                agent_data["is_grounded"] = True
                vh = 0.0
            
            if target_y is not None and ny <= target_y:
                # 着地
                ny = target_y
                t.goto(nx, ny)
                agent_data["is_grounded"] = True
                vh = 0.0
            elif target_y is None and ny < -170:
                # 底まで落下した場合のリセット（安全のため最初の足場に戻す）
                t.goto(-150, -40)
                nx, ny = -150, -40
                agent_data["is_grounded"] = True
                vh = 0.0

            agent_data["h"] = ny
            agent_data["vh"] = vh
            
            next_state = [nx, ny, ny, t.heading(), vh, 1.0 if agent_data["is_grounded"] else 0.0]
            
            if predicted_next is not None:
                pred_error = np.linalg.norm(predicted_next[:3] - np.array(next_state[:3]))
            else:
                pred_error = 12.0
            
            curiosity.evaluate(pred_error)
            cortex.learn(state, action, next_state, pred_error, episode, max_episodes)
            
        # 記憶のランドスケープを描画
        landscape_drawer.clear()
        for c in shared_hippocampus.cells:
            landscape_drawer.goto(c.state[0], c.state[1])
            landscape_drawer.color("#ff00ff" if c.surprise > 3.0 else "#ffffff")
            size = max(2, min(5, int(c.visits * 0.3 + 2)))
            landscape_drawer.dot(size)

        current_step += 1
        screen.update()
        screen.ontimer(run_lifecycle, 25)
        
    else:
        print(f"--- Episode {episode+1} Finished (Sleep & Consolidation) ---")
        for agent_data in agents:
            agent_data["turtle"].clear()
            agent_data["turtle"].goto(-150, -40)
            agent_data["is_grounded"] = True
            agent_data["vh"] = 0.0
            
        shared_hippocampus.replay(200)
        shared_hippocampus.prune_spatial_density(radius=15.0)
        shared_hippocampus.metabolize()
        
        current_step = 0
        episode += 1
        if episode < max_episodes:
            screen.ontimer(run_lifecycle, 100)
        else:
            print("Simulation Complete. Athletic Body & World Model Formed.")
            screen.bye()

screen.ontimer(run_lifecycle, 100)
turtle.done()
