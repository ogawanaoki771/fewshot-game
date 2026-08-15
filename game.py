import turtle
import random
from dataclasses import dataclass

# =========================================================
# 基本設定
# =========================================================

GRID_SIZE = 9
CELL_SIZE = 60

SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700

MAX_HP = 5

UP = (0, 1)
DOWN = (0, -1)
LEFT = (-1, 0)
RIGHT = (1, 0)

ACTIONS = [UP, DOWN, LEFT, RIGHT]


# =========================================================
# GameState
# =========================================================

@dataclass
class GameState:
    turn: int = 0
    gate_timer: int = 0
    game_over: bool = False
    victory: bool = False

    @property
    def gate_open(self):
        return self.gate_timer > 0


# =========================================================
# Grid
# =========================================================

class Grid:
    def __init__(self, size):
        self.size = size

    def inside(self, pos):
        x, y = pos

        half = self.size // 2

        return (
            -half <= x <= half
            and
            -half <= y <= half
        )


# =========================================================
# Game Objects
# =========================================================

class Trap:
    def __init__(self, pos):
        self.pos = pos


class Treasure:
    def __init__(self, pos):
        self.pos = pos
        self.active = True


class Gate:
    def __init__(self, pos):
        self.pos = pos


class Goal:
    def __init__(self, pos):
        self.pos = pos


# =========================================================
# Cell
# =========================================================

class Cell:
    def __init__(self, name, pos, color):
        self.name = name
        self.pos = pos
        self.color = color

        self.hp = MAX_HP

        self.turtle = turtle.Turtle()
        self.turtle.shape("circle")
        self.turtle.color(color)
        self.turtle.penup()
        self.turtle.speed(0)

        self.update_visual()

    # -----------------------------------------------------
    # 移動可能位置
    # -----------------------------------------------------

    def valid_actions(self, game):

        actions = []

        for dx, dy in ACTIONS:

            new_pos = (
                self.pos[0] + dx,
                self.pos[1] + dy
            )

            if not game.grid.inside(new_pos):
                continue

            # ゲートが閉じている場合
            if (
                new_pos == game.gate.pos
                and
                not game.state.gate_open
            ):
                continue

            actions.append((dx, dy))

        return actions

    # -----------------------------------------------------
    # 行動決定
    # -----------------------------------------------------

    def decide(self, game):
        raise NotImplementedError

    # -----------------------------------------------------
    # 移動
    # -----------------------------------------------------

    def move(self, action, game):

        dx, dy = action

        new_pos = (
            self.pos[0] + dx,
            self.pos[1] + dy
        )

        if new_pos not in self.valid_positions(game):
            return

        self.pos = new_pos

    def valid_positions(self, game):

        positions = []

        for action in self.valid_actions(game):

            dx, dy = action

            positions.append(
                (
                    self.pos[0] + dx,
                    self.pos[1] + dy
                )
            )

        return positions

    # -----------------------------------------------------
    # ダメージ
    # -----------------------------------------------------

    def damage(self):

        self.hp -= 1

        print(
            f"{self.name}: DAMAGE! HP={self.hp}"
        )

        if self.hp <= 0:
            print(
                f"{self.name}: DEAD"
            )

    # -----------------------------------------------------
    # 回復
    # -----------------------------------------------------

    def heal(self):

        self.hp = min(
            self.hp + 1,
            MAX_HP
        )

        print(
            f"{self.name}: HEAL! HP={self.hp}"
        )

    # -----------------------------------------------------
    # 描画
    # -----------------------------------------------------

    def update_visual(self):

        x, y = self.pos

        self.turtle.goto(
            x * CELL_SIZE,
            y * CELL_SIZE
        )


# =========================================================
# AlgorithmCell
# =========================================================

class AlgorithmCell(Cell):

    def __init__(self, pos):
        super().__init__(
            "Algorithm",
            pos,
            "blue"
        )

    def decide(self, game):

        valid = self.valid_actions(game)

        if not valid:
            return None

        goal = game.goal.pos

        best_action = None
        best_distance = float("inf")

        for dx, dy in valid:

            new_pos = (
                self.pos[0] + dx,
                self.pos[1] + dy
            )

            distance = abs(
                new_pos[0] - goal[0]
            ) + abs(
                new_pos[1] - goal[1]
            )

            if distance < best_distance:

                best_distance = distance
                best_action = (dx, dy)

        return best_action


# =========================================================
# RandomCell
# =========================================================

class RandomCell(Cell):

    def __init__(self, pos):
        super().__init__(
            "Random",
            pos,
            "green"
        )

    def decide(self, game):

        valid = self.valid_actions(game)

        if not valid:
            return None

        return random.choice(valid)


# =========================================================
# YourAICell
# =========================================================

class YourAICell(Cell):

    def __init__(self, pos):
        super().__init__(
            "YourAI",
            pos,
            "red"
        )

    def decide(self, game):

        # ---------------------------------------------
        # 今は仮AI
        #
        # ここを将来自作AIに置き換える
        # ---------------------------------------------

        valid = self.valid_actions(game)

        if not valid:
            return None

        goal = game.goal.pos

        # 現在は単純にゴールへの距離を見る
        best_action = None
        best_score = float("inf")

        for action in valid:

            dx, dy = action

            new_pos = (
                self.pos[0] + dx,
                self.pos[1] + dy
            )

            distance = (
                abs(new_pos[0] - goal[0])
                +
                abs(new_pos[1] - goal[1])
            )

            # 罠を避ける
            trap_penalty = 0

            if any(
                trap.pos == new_pos
                for trap in game.traps
            ):
                trap_penalty = 10

            score = distance + trap_penalty

            if score < best_score:

                best_score = score
                best_action = action

        return best_action


# =========================================================
# Game
# =========================================================

class Game:

    def __init__(self):

        self.screen = turtle.Screen()

        self.screen.setup(
            SCREEN_WIDTH,
            SCREEN_HEIGHT
        )

        self.screen.title(
            "Three Cell Game"
        )

        self.screen.bgcolor("white")

        self.grid = Grid(GRID_SIZE)

        self.state = GameState()

        # ---------------------------------------------
        # ゲームオブジェクト
        # ---------------------------------------------

        self.goal = Goal(
            (4, 4)
        )

        self.gate = Gate(
            (2, 3)
        )

        self.traps = [
            Trap((-2, 0)),
            Trap((0, 1)),
            Trap((1, 2)),
            Trap((2, 1)),
        ]

        self.treasures = [
            Treasure((-3, -2)),
            Treasure((1, -3)),
        ]

        # ---------------------------------------------
        # Cell
        # ---------------------------------------------

        self.algorithm_cell = AlgorithmCell(
            (-4, -4)
        )

        self.random_cell = RandomCell(
            (-3, -4)
        )

        self.your_ai_cell = YourAICell(
            (-2, -4)
        )

        self.cells = [
            self.algorithm_cell,
            self.random_cell,
            self.your_ai_cell
        ]

        # ---------------------------------------------
        # 描画
        # ---------------------------------------------

        self.draw_board()
        self.draw_objects()

        self.info = turtle.Turtle()
        self.info.hideturtle()
        self.info.penup()

        self.info.goto(
            -SCREEN_WIDTH // 2 + 20,
            SCREEN_HEIGHT // 2 - 40
        )

        # ゲーム開始
        self.update()


    # =====================================================
    # 座標変換
    # =====================================================

    def screen_position(self, pos):

        x, y = pos

        return (
            x * CELL_SIZE,
            y * CELL_SIZE
        )


    # =====================================================
    # ボード
    # =====================================================

    def draw_board(self):

        drawer = turtle.Turtle()

        drawer.hideturtle()
        drawer.speed(0)
        drawer.penup()
        drawer.color("gray")

        half = GRID_SIZE // 2

        # 縦線
        for x in range(
            -half,
            half + 2
        ):

            sx = x * CELL_SIZE

            drawer.goto(
                sx,
                -half * CELL_SIZE
            )

            drawer.pendown()

            drawer.goto(
                sx,
                half * CELL_SIZE
            )

            drawer.penup()

        # 横線
        for y in range(
            -half,
            half + 2
        ):

            sy = y * CELL_SIZE

            drawer.goto(
                -half * CELL_SIZE,
                sy
            )

            drawer.pendown()

            drawer.goto(
                half * CELL_SIZE,
                sy
            )

            drawer.penup()


    # =====================================================
    # オブジェクト描画
    # =====================================================

    def draw_objects(self):

        self.object_turtles = []

        # ---------------------------------------------
        # Trap
        # ---------------------------------------------

        for trap in self.traps:

            t = turtle.Turtle()

            t.shape("square")
            t.shapesize(0.5)
            t.color("black")
            t.penup()

            t.goto(
                *self.screen_position(
                    trap.pos
                )
            )

            self.object_turtles.append(t)

        # ---------------------------------------------
        # Treasure
        # ---------------------------------------------

        for treasure in self.treasures:

            t = turtle.Turtle()

            t.shape("circle")
            t.color("gold")
            t.penup()

            t.goto(
                *self.screen_position(
                    treasure.pos
                )
            )

            self.object_turtles.append(t)

        # ---------------------------------------------
        # Gate
        # ---------------------------------------------

        self.gate_turtle = turtle.Turtle()

        self.gate_turtle.shape(
            "square"
        )

        self.gate_turtle.shapesize(
            0.8
        )

        self.gate_turtle.penup()

        self.update_gate_visual()

        # ---------------------------------------------
        # Goal
        # ---------------------------------------------

        self.goal_turtle = turtle.Turtle()

        self.goal_turtle.shape(
            "triangle"
        )

        self.goal_turtle.color(
            "purple"
        )

        self.goal_turtle.penup()

        self.goal_turtle.goto(
            *self.screen_position(
                self.goal.pos
            )
        )


    # =====================================================
    # Gate表示
    # =====================================================

    def update_gate_visual(self):

        self.gate_turtle.goto(
            *self.screen_position(
                self.gate.pos
            )
        )

        if self.state.gate_open:

            self.gate_turtle.color(
                "green"
            )

        else:

            self.gate_turtle.color(
                "gray"
            )


    # =====================================================
    # イベント処理
    # =====================================================

    def process_cell(self, cell):

        # ---------------------------------------------
        # Trap
        # ---------------------------------------------

        for trap in self.traps:

            if cell.pos == trap.pos:

                cell.damage()


        # ---------------------------------------------
        # Treasure
        # ---------------------------------------------

        for treasure in self.treasures:

            if (
                treasure.active
                and
                cell.pos == treasure.pos
            ):

                treasure.active = False

                cell.heal()

        # ---------------------------------------------
        # Gate
        # ---------------------------------------------

        if cell.pos == self.gate.pos:

            self.state.gate_timer = 3

            print(
                f"{cell.name}: GATE OPEN"
            )


        # ---------------------------------------------
        # Goal
        # ---------------------------------------------

        if cell.pos == self.goal.pos:

            self.state.game_over = True
            self.state.victory = True

            print(
                f"{cell.name}: GOAL!"
            )


    # =====================================================
    # ゲーム更新
    # =====================================================

    def update(self):

        if self.state.game_over:

            self.show_game_over()

            return

        self.state.turn += 1

        print(
            f"\n===== TURN {self.state.turn} ====="
        )

        # ---------------------------------------------
        # 各Cellを行動
        # ---------------------------------------------

        for cell in self.cells:

            action = cell.decide(self)

            if action is not None:

                cell.move(
                    action,
                    self
                )

                cell.update_visual()

                self.process_cell(
                    cell
                )

        # ---------------------------------------------
        # Gate timer
        # ---------------------------------------------

        if self.state.gate_timer > 0:

            self.state.gate_timer -= 1

        self.update_gate_visual()

        # ---------------------------------------------
        # 情報表示
        # ---------------------------------------------

        self.update_info()

        # ---------------------------------------------
        # 次のターン
        # ---------------------------------------------

        if not self.state.game_over:

            self.screen.ontimer(
                self.update,
                500
            )


    # =====================================================
    # UI
    # =====================================================

    def update_info(self):

        self.info.clear()

        text = (
            f"Turn: {self.state.turn}    "
            f"Gate: "
            f"{'OPEN' if self.state.gate_open else 'CLOSED'}\n"
            f"Blue Algorithm HP: "
            f"{self.algorithm_cell.hp}\n"
            f"Green Random HP: "
            f"{self.random_cell.hp}\n"
            f"Red YourAI HP: "
            f"{self.your_ai_cell.hp}"
        )

        self.info.write(
            text,
            font=(
                "Arial",
                14,
                "normal"
            )
        )


    # =====================================================
    # Game Over
    # =====================================================

    def show_game_over(self):

        message = turtle.Turtle()

        message.hideturtle()
        message.penup()

        message.goto(
            0,
            0
        )

        if self.state.victory:

            message.write(
                "GOAL!",
                align="center",
                font=(
                    "Arial",
                    36,
                    "bold"
                )
            )

        else:

            message.write(
                "GAME OVER",
                align="center",
                font=(
                    "Arial",
                    36,
                    "bold"
                )
            )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    game = Game()

    turtle.mainloop()
