"""
MAZE // PATHFINDING LAB  (Python + pygame version)
====================================================
2D procedural maze with a shortest-path mouse agent -- now merged with
the "instinct walk" (no-lookahead, smell + random) brain into a single
game, plus maze persistence and a proper in-game UI.

Run:
    pip install pygame
    python maze_game.py

Controls:
    N        - generate a BRAND NEW maze (new walls) and remember it
    R        - RESET the current maze (same walls, mouse/cheese/timer
               back to the start) -- does NOT change the layout
    A        - start / pause auto-play
    1/2/3/4  - switch "brain": 1=A* 2=Dijkstra 3=BFS 4=Instinct walk
    V        - toggle visited / heatmap overlay
    [ / ]    - (fallback) nudge the instinct slider by 10%
    ESC / close window - quit

    All of the above are also clickable buttons in the sidebar, and the
    instinct ratio can be dragged directly with the mouse on its slider.

Maze memory:
    The maze layout (walls only, not mouse/cheese progress) is saved to
    "maze_save.json" next to this script every time a NEW maze is
    generated. On startup the game loads that file if present, so
    closing and reopening the game always shows the SAME maze until you
    press "N" (or the button) to generate + save a new one. "R" / Reset
    intentionally does NOT touch this file -- it only resets progress
    on the maze you already have.

Brains (selectable, swappable mid-game):
    - A* / Dijkstra / BFS: on every tick the mouse re-plans a full
      shortest path to the nearest remaining cheese and takes one step
      of it (see original algorithmic description below).
    - Instinct walk: no graph search at all. The mouse only "smells"
      the Manhattan distance to the nearest cheese and, most of the
      time (slider), steps toward whichever legal neighbor smells
      closest; the rest of the time it wanders to a random legal
      neighbor. It won't immediately backtrack unless stuck in a dead
      end. The slider (0%-100%) controls how much it trusts its nose
      vs. wanders randomly.

Rules implemented (per spec):
    - 30 x 30 grid, each cell 16x16 cm (informational, shown in UI).
    - Maze generated as a perfect maze (randomized DFS / recursive
      backtracker) -> guaranteed solvable, no unreachable areas,
      naturally produces dead ends. Path width = 1 cell.
    - Mouse moves 1 cell at a time, 4 directions only (up/down/left/right),
      cannot pass through walls or leave the grid. Movement only happens
      through auto-play (no manual stepping).
    - Cheese is placed at Start and Goal; mouse always targets whichever
      remaining cheese is nearest (by the active brain's own notion of
      "nearest"). Auto-play automatically PAUSES the instant the mouse
      eats any cheese (including the one at Start, at the very beginning).
    - The 3-minute clock starts the FIRST time you press A (auto-play),
      not when the maze is generated / reset.
    - WIN: mouse reaches the cheese at Goal.
    - LOSE: total auto-play time exceeds 3 minutes (180 seconds).
"""

import os
import sys
import time
import heapq
import json
import random
import pygame

# ============================================================================
# Fonts (bundled Thai-capable TTFs so Thai text never shows as tofu boxes)
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _font_candidates(bold):
    name = "Waree-Bold.ttf" if bold else "Waree.ttf"
    return [
        os.path.join(SCRIPT_DIR, "assets", "fonts", name),  # preferred layout
        os.path.join(SCRIPT_DIR, name),                       # dropped next to script
    ]


def load_thai_font(size, bold=False):
    """Load the bundled Thai font; fall back to a system font if missing."""
    for path in _font_candidates(bold):
        try:
            if os.path.exists(path):
                return pygame.font.Font(path, size)
        except Exception:
            pass
    for name in ("tahoma", "leelawadeeui", "angsananew", "notosansthai", "arial"):
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f is not None:
                return f
        except Exception:
            continue
    return pygame.font.SysFont(None, size, bold=bold)


# ============================================================================
# MODULE: CONFIG
# ============================================================================
GRID_SIZE = 30            # 30 x 30 cells
CELL_CM = 16               # real-world size per cell (informational only)
CELL_PX = 20                # on-screen pixel size per cell
MAZE_PX = GRID_SIZE * CELL_PX
SIDEBAR_PX = 380
WINDOW_W = MAZE_PX + SIDEBAR_PX
WINDOW_H = 700

TIME_LIMIT_SECONDS = 3 * 60   # 3 minutes -> lose condition
COMPUTATION_MINIMUM = 5        # informational only (not a fail condition anymore)

SAVE_FILE = os.path.join(SCRIPT_DIR, "maze_save.json")

COLOR_BG = (10, 20, 36)
COLOR_PANEL = (15, 27, 46)
COLOR_LINE = (58, 85, 128)
COLOR_GRID_FAINT = (30, 45, 70)
COLOR_TEXT = (230, 237, 247)
COLOR_MUTED = (124, 140, 166)
COLOR_CYAN = (76, 211, 224)
COLOR_AMBER = (245, 166, 35)
COLOR_RED = (240, 85, 77)
COLOR_GREEN = (95, 217, 138)

# Selectable "brains" -- order also maps to keys 1/2/3/4.
ALGO_OPTIONS = [
    ("astar", "A*"),
    ("dijkstra", "Dijkstra"),
    ("bfs", "BFS"),
    ("instinct", "สัญชาตญาณ"),
]
PATHFINDING_ALGOS = {"astar", "dijkstra", "bfs"}

# Default instinct ratio: chance the mouse trusts its nose vs wanders.
DEFAULT_INSTINCT_STRENGTH = 0.7


# ============================================================================
# MODULE: Maze persistence ("จำแมพ")
# The maze layout is saved every time a brand-new one is generated, and
# loaded back on startup so the SAME maze reopens until "N" is pressed.
# ============================================================================
def load_maze_from_file():
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cells = data.get("cells")
        if cells and len(cells) == GRID_SIZE and len(cells[0]) == GRID_SIZE:
            return cells
    except Exception:
        pass
    return None


def save_maze_to_file(cells):
    try:
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump({"cells": cells}, f)
    except Exception:
        pass  # persistence is a nice-to-have; never crash the game over it


# ============================================================================
# MODULE: MazeGenerator
# Builds a perfect maze (spanning tree) via randomized depth-first search.
# A perfect maze guarantees exactly one path between any two cells
# (=> solvable, no unreachable areas) and naturally creates dead ends.
# ============================================================================
class MazeGenerator:
    DIRS = [
        ("top", -1, 0, "bottom"),
        ("right", 0, 1, "left"),
        ("bottom", 1, 0, "top"),
        ("left", 0, -1, "right"),
    ]

    @staticmethod
    def generate(size):
        cells = [[{"top": True, "right": True, "bottom": True, "left": True,
                    "visited": False} for _ in range(size)] for _ in range(size)]

        stack = [(0, 0)]
        cells[0][0]["visited"] = True

        while stack:
            r, c = stack[-1]
            unvisited = []
            for key, dr, dc, opposite in MazeGenerator.DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and not cells[nr][nc]["visited"]:
                    unvisited.append((nr, nc, key, opposite))

            if unvisited:
                nr, nc, key, opposite = random.choice(unvisited)
                cells[r][c][key] = False
                cells[nr][nc][opposite] = False
                cells[nr][nc]["visited"] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        return cells

    @staticmethod
    def neighbors(cells, r, c):
        size = len(cells)
        cell = cells[r][c]
        out = []
        if not cell["top"] and r > 0:
            out.append((r - 1, c))
        if not cell["bottom"] and r < size - 1:
            out.append((r + 1, c))
        if not cell["left"] and c > 0:
            out.append((r, c - 1))
        if not cell["right"] and c < size - 1:
            out.append((r, c + 1))
        return out


# ============================================================================
# MODULE: Pathfinding (strategy pattern - swap algorithms freely)
# Every algorithm returns a dict:
#   { path: [(r,c), ...], visited: [(r,c), ...], nodes_explored: int,
#     distance: int (-1 if not found), time_ms: float }
# ============================================================================
class Pathfinding:

    @staticmethod
    def _reconstruct(prev, start, goal):
        path = [goal]
        cur = goal
        while cur != start:
            cur = prev.get(cur)
            if cur is None:
                return []
            path.append(cur)
        path.reverse()
        return path

    @staticmethod
    def bfs(cells, start, goal):
        t0 = time.perf_counter()
        from collections import deque
        visited = {start}
        prev = {}
        queue = deque([start])
        order = []
        found = False

        while queue:
            cur = queue.popleft()
            order.append(cur)
            if cur == goal:
                found = True
                break
            for n in MazeGenerator.neighbors(cells, *cur):
                if n not in visited:
                    visited.add(n)
                    prev[n] = cur
                    queue.append(n)

        path = Pathfinding._reconstruct(prev, start, goal) if found else []
        distance = len(path) - 1 if path else -1
        return {"path": path, "visited": order, "nodes_explored": len(order),
                "distance": distance, "time_ms": (time.perf_counter() - t0) * 1000}

    @staticmethod
    def dijkstra(cells, start, goal):
        t0 = time.perf_counter()
        dist = {start: 0}
        prev = {}
        visited = set()
        order = []
        pq = [(0, start)]
        found = False

        while pq:
            d, cur = heapq.heappop(pq)
            if cur in visited:
                continue
            visited.add(cur)
            order.append(cur)
            if cur == goal:
                found = True
                break
            for n in MazeGenerator.neighbors(cells, *cur):
                nd = d + 1
                if nd < dist.get(n, float("inf")):
                    dist[n] = nd
                    prev[n] = cur
                    heapq.heappush(pq, (nd, n))

        path = Pathfinding._reconstruct(prev, start, goal) if found else []
        distance = len(path) - 1 if path else -1
        return {"path": path, "visited": order, "nodes_explored": len(order),
                "distance": distance, "time_ms": (time.perf_counter() - t0) * 1000}

    @staticmethod
    def astar(cells, start, goal):
        t0 = time.perf_counter()

        def h(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])  # Manhattan

        g = {start: 0}
        prev = {}
        visited = set()
        order = []
        open_heap = [(h(start), start)]
        found = False

        while open_heap:
            f, cur = heapq.heappop(open_heap)
            if cur in visited:
                continue
            visited.add(cur)
            order.append(cur)
            if cur == goal:
                found = True
                break
            for n in MazeGenerator.neighbors(cells, *cur):
                ng = g[cur] + 1
                if ng < g.get(n, float("inf")):
                    g[n] = ng
                    prev[n] = cur
                    heapq.heappush(open_heap, (ng + h(n), n))

        path = Pathfinding._reconstruct(prev, start, goal) if found else []
        distance = len(path) - 1 if path else -1
        return {"path": path, "visited": order, "nodes_explored": len(order),
                "distance": distance, "time_ms": (time.perf_counter() - t0) * 1000}

    @staticmethod
    def run(name, cells, start, goal):
        if name == "bfs":
            return Pathfinding.bfs(cells, start, goal)
        if name == "dijkstra":
            return Pathfinding.dijkstra(cells, start, goal)
        return Pathfinding.astar(cells, start, goal)


# ============================================================================
# MODULE: GameManager
# Owns maze state, mouse position, cheese list, timers, and BOTH brains
# (pathfinding + instinct walk) so switching mid-game is instant.
# ============================================================================
class GameManager:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (GRID_SIZE - 1, GRID_SIZE - 1)

        loaded = load_maze_from_file()
        if loaded is not None:
            self.cells = loaded
        else:
            self.cells = MazeGenerator.generate(GRID_SIZE)
            save_maze_to_file(self.cells)

        self._reset_state()

    # ---- maze-level actions -------------------------------------------
    def new_maze(self):
        """Generate a BRAND NEW maze layout and remember it on disk."""
        self.cells = MazeGenerator.generate(GRID_SIZE)
        save_maze_to_file(self.cells)
        self._reset_state()

    def reset_current(self):
        """Reset mouse/cheese/timer/stats but KEEP the current walls."""
        self._reset_state()

    def _reset_state(self):
        self.mouse = self.start
        self.cheeses = [self.start, self.goal]
        self.status = "playing"     # playing | win | lose
        self.lose_reason = ""
        self.start_time = None
        self.end_time = None
        self.timer_started = False

        # pathfinding-brain bookkeeping
        self.computations = 0
        self.last_result = None

        # instinct-brain bookkeeping
        self.prev_cell = None
        self.steps = 0
        self.backtracks = 0
        self.last_move_was_instinct = None
        self.last_sensed_distance = None
        self.visit_counts = {}

        self._collect_cheese_at_mouse()

    # ---- shared helpers -------------------------------------------------
    def start_timer_if_needed(self):
        """Start the 3-minute clock the first time auto-play is switched on."""
        if not self.timer_started:
            self.timer_started = True
            self.start_time = time.time()

    def _freeze_timer(self):
        if self.timer_started and self.end_time is None:
            self.end_time = time.time()

    def _collect_cheese_at_mouse(self):
        self.cheeses = [ch for ch in self.cheeses if ch != self.mouse]

    def elapsed_seconds(self):
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def _check_time_limit(self):
        if not self.timer_started or self.status != "playing":
            return False
        if self.elapsed_seconds() > TIME_LIMIT_SECONDS:
            self.status = "lose"
            self.lose_reason = f"หมดเวลา! เดินเกิน {TIME_LIMIT_SECONDS // 60} นาที — หนู \"ตาย\""
            self._freeze_timer()
            return True
        return False

    # ---- dispatch ---------------------------------------------------------
    def step(self, algo_name, instinct_strength):
        if algo_name == "instinct":
            return self._step_instinct(instinct_strength)
        return self._step_pathfinding(algo_name)

    # ---- brain 1: shortest-path (A* / Dijkstra / BFS) ----------------------
    def _nearest_cheese(self, algo_name):
        best = None
        for ch in self.cheeses:
            result = Pathfinding.run(algo_name, self.cells, self.mouse, ch)
            if result["path"] and (best is None or result["distance"] < best[1]["distance"]):
                best = (ch, result)
        return best

    def _step_pathfinding(self, algo_name):
        if self.status != "playing":
            return {"ok": False}

        if self._check_time_limit():
            return {"ok": False}

        if not self.cheeses:
            self.status = "win"
            self._freeze_timer()
            return {"ok": False}

        best = self._nearest_cheese(algo_name)
        self.computations += 1
        self.last_result = best[1] if best else {
            "nodes_explored": 0, "distance": -1, "time_ms": 0.0, "path": [], "visited": []
        }

        if self._check_time_limit():
            return {"ok": False, "result": self.last_result}

        if not best:
            self.status = "lose"
            self.lose_reason = "ไม่สามารถหาเส้นทางไปชีสได้ (unreachable)"
            self._freeze_timer()
            return {"ok": False, "result": self.last_result}

        path = best[1]["path"]
        cheese_count_before = len(self.cheeses)

        if len(path) < 2:
            self._collect_cheese_at_mouse()
            ate = len(self.cheeses) < cheese_count_before
            return {"ok": True, "result": self.last_result, "moved": False, "ate_cheese": ate}

        next_pos = path[1]
        legal_moves = MazeGenerator.neighbors(self.cells, *self.mouse)
        if next_pos not in legal_moves:
            self.status = "lose"
            self.lose_reason = "หนูพยายามเดินทะลุกำแพง (illegal move)"
            self._freeze_timer()
            return {"ok": False, "result": self.last_result}

        self.mouse = next_pos
        self._collect_cheese_at_mouse()
        ate = len(self.cheeses) < cheese_count_before

        if not self.cheeses:
            self.status = "win"
            self._freeze_timer()

        return {"ok": True, "result": self.last_result, "moved": True, "ate_cheese": ate}

    # ---- brain 2: instinct walk (smell + random, no lookahead) -------------
    def _nearest_cheese_and_distance(self):
        best = None
        for ch in self.cheeses:
            d = abs(ch[0] - self.mouse[0]) + abs(ch[1] - self.mouse[1])
            if best is None or d < best[1]:
                best = (ch, d)
        return best

    def _step_instinct(self, instinct_strength):
        if self.status != "playing":
            return {"ok": False}

        if self._check_time_limit():
            return {"ok": False}

        if not self.cheeses:
            self.status = "win"
            self._freeze_timer()
            return {"ok": False}

        target_pos, sensed_distance = self._nearest_cheese_and_distance()
        self.last_sensed_distance = sensed_distance

        neighbors = MazeGenerator.neighbors(self.cells, *self.mouse)
        self.steps += 1

        candidates = [n for n in neighbors if n != self.prev_cell]
        if not candidates:
            candidates = neighbors
            self.backtracks += 1

        def dist_to_target(pos):
            return abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])

        if random.random() < instinct_strength:
            min_d = min(dist_to_target(c) for c in candidates)
            best_candidates = [c for c in candidates if dist_to_target(c) == min_d]
            next_pos = random.choice(best_candidates)
            self.last_move_was_instinct = True
        else:
            next_pos = random.choice(candidates)
            self.last_move_was_instinct = False

        self.prev_cell = self.mouse
        self.mouse = next_pos
        self.visit_counts[next_pos] = self.visit_counts.get(next_pos, 0) + 1

        cheese_before = len(self.cheeses)
        self._collect_cheese_at_mouse()
        ate = len(self.cheeses) < cheese_before

        if not self.cheeses:
            self.status = "win"
            self._freeze_timer()

        return {"ok": True, "ate_cheese": ate}


# ============================================================================
# MODULE: small UI widgets (buttons + a draggable slider)
# ============================================================================
class Button:
    def __init__(self, rect, label, font):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font

    def draw(self, screen, active=False):
        bg = COLOR_CYAN if active else COLOR_PANEL
        border = COLOR_CYAN if active else COLOR_LINE
        text_color = COLOR_BG if active else COLOR_TEXT
        pygame.draw.rect(screen, bg, self.rect, border_radius=6)
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=6)
        surf = self.font.render(self.label, True, text_color)
        screen.blit(surf, surf.get_rect(center=self.rect.center))

    def contains(self, pos):
        return self.rect.collidepoint(pos)


class Slider:
    """Horizontal 0.0-1.0 slider, click-and-drag with the mouse."""

    def __init__(self, value=DEFAULT_INSTINCT_STRENGTH):
        self.rect = pygame.Rect(0, 0, 10, 10)  # updated every frame by the renderer
        self.value = value
        self.dragging = False

    def knob_pos(self):
        x = self.rect.x + int(self.value * self.rect.width)
        return x, self.rect.centery

    def handle_mousedown(self, pos):
        kx, ky = self.knob_pos()
        knob_hit = (pos[0] - kx) ** 2 + (pos[1] - ky) ** 2 <= 14 ** 2
        track_hit = self.rect.inflate(0, 16).collidepoint(pos)
        if knob_hit or track_hit:
            self.dragging = True
            self._update_from_x(pos[0])
            return True
        return False

    def handle_mousemotion(self, pos):
        if self.dragging:
            self._update_from_x(pos[0])

    def handle_mouseup(self):
        self.dragging = False

    def nudge(self, delta):
        self.value = round(max(0.0, min(1.0, self.value + delta)), 2)

    def _update_from_x(self, x):
        if self.rect.width <= 0:
            return
        pct = (x - self.rect.x) / self.rect.width
        self.value = round(max(0.0, min(1.0, pct)), 2)

    def draw(self, screen, font, label):
        label_surf = font.render(label, True, COLOR_TEXT)
        screen.blit(label_surf, (self.rect.x, self.rect.y - label_surf.get_height() - 6))

        pygame.draw.rect(screen, COLOR_LINE, self.rect, border_radius=4)
        filled_w = int(self.value * self.rect.width)
        if filled_w > 0:
            pygame.draw.rect(screen, COLOR_CYAN, (self.rect.x, self.rect.y, filled_w, self.rect.height),
                              border_radius=4)
        kx, ky = self.knob_pos()
        pygame.draw.circle(screen, COLOR_AMBER, (kx, ky), 9)
        pygame.draw.circle(screen, COLOR_BG, (kx, ky), 9, 2)


# ============================================================================
# MODULE: Renderer
# ============================================================================
class Renderer:
    def __init__(self, screen, font, font_small, font_big):
        self.screen = screen
        self.font = font
        self.font_small = font_small
        self.font_big = font_big

        self.algo_buttons = {}     # name -> Button (rebuilt every frame at fixed layout spots)
        self.control_buttons = {}  # name -> Button
        self.slider = Slider()

    def draw(self, game, algo_name, show_visited, auto_playing):
        self.screen.fill(COLOR_BG)
        self._draw_maze(game, show_visited, algo_name)
        self._draw_sidebar(game, algo_name, show_visited, auto_playing)
        pygame.display.flip()

    def _cell_rect(self, r, c):
        return pygame.Rect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX)

    def _draw_maze(self, game, show_visited, algo_name):
        screen = self.screen
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(screen, COLOR_GRID_FAINT, (i * CELL_PX, 0), (i * CELL_PX, MAZE_PX))
            pygame.draw.line(screen, COLOR_GRID_FAINT, (0, i * CELL_PX), (MAZE_PX, i * CELL_PX))

        if show_visited:
            if algo_name in PATHFINDING_ALGOS and game.last_result and game.last_result.get("visited"):
                overlay = pygame.Surface((MAZE_PX, MAZE_PX), pygame.SRCALPHA)
                for (r, c) in game.last_result["visited"]:
                    pygame.draw.rect(overlay, (245, 166, 35, 45),
                                      (c * CELL_PX + 1, r * CELL_PX + 1, CELL_PX - 2, CELL_PX - 2))
                screen.blit(overlay, (0, 0))
            elif algo_name == "instinct" and game.visit_counts:
                max_count = max(game.visit_counts.values())
                overlay = pygame.Surface((MAZE_PX, MAZE_PX), pygame.SRCALPHA)
                for (r, c), count in game.visit_counts.items():
                    alpha = int(20 + 160 * (count / max_count))
                    pygame.draw.rect(overlay, (245, 166, 35, min(alpha, 180)),
                                      (c * CELL_PX + 1, r * CELL_PX + 1, CELL_PX - 2, CELL_PX - 2))
                screen.blit(overlay, (0, 0))

        if algo_name in PATHFINDING_ALGOS and game.last_result and len(game.last_result.get("path", [])) > 1:
            pts = [(c * CELL_PX + CELL_PX // 2, r * CELL_PX + CELL_PX // 2)
                   for (r, c) in game.last_result["path"]]
            path_surface = pygame.Surface((MAZE_PX, MAZE_PX), pygame.SRCALPHA)
            pygame.draw.lines(path_surface, (76, 211, 224, 140), False, pts, max(2, CELL_PX // 4))
            screen.blit(path_surface, (0, 0))

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = game.cells[r][c]
                x, y = c * CELL_PX, r * CELL_PX
                if cell["top"]:
                    pygame.draw.line(screen, COLOR_LINE, (x, y), (x + CELL_PX, y), 2)
                if cell["left"]:
                    pygame.draw.line(screen, COLOR_LINE, (x, y), (x, y + CELL_PX), 2)
                if cell["bottom"]:
                    pygame.draw.line(screen, COLOR_LINE, (x, y + CELL_PX), (x + CELL_PX, y + CELL_PX), 2)
                if cell["right"]:
                    pygame.draw.line(screen, COLOR_LINE, (x + CELL_PX, y), (x + CELL_PX, y + CELL_PX), 2)

        pygame.draw.rect(screen, (90, 127, 176), (0, 0, MAZE_PX, MAZE_PX), 3)

        self._marker(game.start, COLOR_GREEN)
        self._marker(game.goal, COLOR_RED)

        for ch in game.cheeses:
            rect = self._cell_rect(*ch)
            pygame.draw.circle(screen, COLOR_AMBER, rect.center, CELL_PX // 3)

        rect = self._cell_rect(*game.mouse)
        pygame.draw.circle(screen, COLOR_CYAN, rect.center, CELL_PX // 2 - 2)

    def _marker(self, pos, color):
        rect = self._cell_rect(*pos)
        s = pygame.Surface((CELL_PX, CELL_PX), pygame.SRCALPHA)
        s.fill((*color, 55))
        self.screen.blit(s, rect.topleft)
        pygame.draw.rect(self.screen, color, rect, 2)

    def _draw_sidebar(self, game, algo_name, show_visited, auto_playing):
        screen = self.screen
        x0 = MAZE_PX
        pad = 18
        usable_w = SIDEBAR_PX - 2 * pad
        pygame.draw.rect(screen, COLOR_PANEL, (x0, 0, SIDEBAR_PX, WINDOW_H))
        pygame.draw.line(screen, COLOR_LINE, (x0, 0), (x0, WINDOW_H), 2)

        y = 20

        def text(s, font=None, color=COLOR_TEXT):
            nonlocal y
            surf = (font or self.font).render(s, True, color)
            screen.blit(surf, (x0 + pad, y))
            y += surf.get_height() + 6

        def two_col_buttons(row_defs, font):
            """row_defs: list of (name, label). Builds Button objects, draws
            them, returns {name: Button}. Advances y by one row."""
            nonlocal y
            gap = 10
            n = len(row_defs)
            w = (usable_w - gap * (n - 1)) // n
            h = 32
            out = {}
            for i, (name, label) in enumerate(row_defs):
                rect = (x0 + pad + i * (w + gap), y, w, h)
                btn = Button(rect, label, font)
                out[name] = btn
            y += h + 8
            return out

        text("MAZE // PATHFINDING LAB", self.font_big, COLOR_CYAN)
        text("30x30 cells, 16x16 cm each", self.font_small, COLOR_MUTED)
        y += 6

        status_colors = {"playing": COLOR_CYAN, "win": COLOR_GREEN, "lose": COLOR_RED}
        status_labels = {"playing": "กำลังเล่น", "win": "ชนะ !", "lose": "แพ้"}
        text(f"สถานะ: {status_labels[game.status]}", self.font, status_colors[game.status])

        elapsed = game.elapsed_seconds()
        remaining = max(0, TIME_LIMIT_SECONDS - elapsed)
        mm, ss = divmod(int(remaining), 60)
        time_color = COLOR_RED if remaining < 30 else (COLOR_AMBER if remaining < 60 else COLOR_TEXT)
        text(f"เวลาที่เหลือ: {mm:02d}:{ss:02d} / 03:00", self.font, time_color)

        bar_w = usable_w
        pct = max(0.0, min(1.0, remaining / TIME_LIMIT_SECONDS))
        pygame.draw.rect(screen, COLOR_LINE, (x0 + pad, y, bar_w, 8))
        bar_color = COLOR_RED if pct < 0.16 else (COLOR_AMBER if pct < 0.33 else COLOR_CYAN)
        pygame.draw.rect(screen, bar_color, (x0 + pad, y, int(bar_w * pct), 8))
        y += 20

        # ---- Algorithm selection (clickable buttons + 1/2/3/4) ----
        text("เลือกอัลกอริทึม (สมอง):", self.font_small, COLOR_MUTED)
        row1 = two_col_buttons([("astar", "1: A*"), ("dijkstra", "2: Dijkstra")], self.font_small)
        row2 = two_col_buttons([("bfs", "3: BFS"), ("instinct", "4: สัญชาตญาณ")], self.font_small)
        self.algo_buttons = {**row1, **row2}
        for name, btn in self.algo_buttons.items():
            btn.draw(screen, active=(name == algo_name))
        y += 4

        # ---- Instinct ratio slider (replaces the old [ / ] keys) ----
        slider_h = 14
        self.slider.rect = pygame.Rect(x0 + pad, y + 20, usable_w, slider_h)
        active_note = "" if algo_name == "instinct" else "  (ใช้กับโหมดสัญชาตญาณ)"
        self.slider.draw(screen, self.font_small,
                          f"ตามกลิ่น {int(self.slider.value*100)}%  /  เดินมั่ว {int((1-self.slider.value)*100)}%{active_note}")
        y += 20 + slider_h + 20

        # ---- Control buttons: new maze / reset / auto-play / visited ----
        crow1 = two_col_buttons([("new", "แมพใหม่ (N)"), ("reset", "รีเซ็ต (R)")], self.font_small)
        crow2 = two_col_buttons([("auto", ("หยุด" if auto_playing else "เริ่ม") + " อัตโนมัติ (A)"),
                                  ("visited", ("ซ่อน" if show_visited else "แสดง") + " overlay (V)")],
                                 self.font_small)
        self.control_buttons = {**crow1, **crow2}
        self.control_buttons["new"].draw(screen)
        self.control_buttons["reset"].draw(screen)
        self.control_buttons["auto"].draw(screen, active=auto_playing)
        self.control_buttons["visited"].draw(screen, active=show_visited)
        y += 4

        text("แมพจะถูกจำไว้ที่ maze_save.json — ปิดเปิดใหม่ได้แมพเดิม", self.font_small, COLOR_MUTED)
        y += 4

        # ---- Stats (depends on selected brain) ----
        if algo_name in PATHFINDING_ALGOS:
            text(f"Computation ใช้ไป: {game.computations}")
            r = game.last_result
            if r:
                text(f"Nodes สำรวจล่าสุด: {r['nodes_explored']}")
                dist_txt = f"{r['distance']} ช่อง" if r["distance"] >= 0 else "ไม่พบ"
                text(f"ระยะทางที่พบ: {dist_txt}")
                text(f"เวลาในการคำนวณ: {r['time_ms']:.2f} ms")
            else:
                text("Nodes สำรวจล่าสุด: -")
                text("ระยะทางที่พบ: -")
                text("เวลาในการคำนวณ: -")
        else:
            text(f"ก้าวที่เดินไปแล้ว: {game.steps}")
            text(f"เลี้ยวกลับ (dead end): {game.backtracks}")
            d_txt = game.last_sensed_distance if game.last_sensed_distance is not None else "-"
            text(f"ระยะที่ 'ได้กลิ่น' ตอนนี้: {d_txt}")
            if game.last_move_was_instinct is not None:
                move_txt = "เดินตามกลิ่น" if game.last_move_was_instinct else "เดินมั่ว (สุ่ม)"
                text(f"ก้าวล่าสุด: {move_txt}", self.font_small, COLOR_MUTED)

        text(f"ชีสที่เหลือ: {len(game.cheeses)}")
        y += 6

        if game.status == "win":
            if algo_name == "instinct":
                text(f"ผลลัพธ์: หนูหาทางเจอชีสเอง! ใช้ {game.steps} ก้าว", self.font, COLOR_GREEN)
            else:
                text("ผลลัพธ์: หนูถึงชีสที่ Goal สำเร็จ!", self.font, COLOR_GREEN)
        elif game.status == "lose":
            text(f"ผลลัพธ์: {game.lose_reason}", self.font, COLOR_RED)

        y += 8
        text("N = แมพใหม่ (จำไว้อัตโนมัติ) · R = รีเซ็ตแมพเดิม", self.font_small, COLOR_MUTED)
        text("A = เริ่ม/หยุดเดิน (นาฬิกาเริ่มตอนกด A ครั้งแรก)", self.font_small, COLOR_MUTED)
        text("ESC = ออก", self.font_small, COLOR_MUTED)


# ============================================================================
# MAIN
# ============================================================================
def main():
    pygame.init()
    pygame.display.set_caption("Maze // Pathfinding Lab")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    font = load_thai_font(18)
    font_small = load_thai_font(14)
    font_big = load_thai_font(22, bold=True)

    game = GameManager()
    renderer = Renderer(screen, font, font_small, font_big)

    algo_name = "astar"
    show_visited = True
    auto_playing = False
    auto_interval_ms = 120          # step pace for pathfinding brains
    STEPS_PER_FRAME_INSTINCT = 10   # instinct steps are cheap, run several/frame
    last_step_time = pygame.time.get_ticks()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    game.new_maze()
                    auto_playing = False
                elif event.key == pygame.K_r:
                    game.reset_current()
                    auto_playing = False
                elif event.key == pygame.K_a:
                    auto_playing = not auto_playing
                    if auto_playing:
                        game.start_timer_if_needed()
                        last_step_time = pygame.time.get_ticks()
                elif event.key == pygame.K_v:
                    show_visited = not show_visited
                elif event.key == pygame.K_1:
                    algo_name = "astar"
                elif event.key == pygame.K_2:
                    algo_name = "dijkstra"
                elif event.key == pygame.K_3:
                    algo_name = "bfs"
                elif event.key == pygame.K_4:
                    algo_name = "instinct"
                elif event.key == pygame.K_LEFTBRACKET:
                    renderer.slider.nudge(-0.1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    renderer.slider.nudge(0.1)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                clicked_algo = False
                for name, btn in renderer.algo_buttons.items():
                    if btn.contains(pos):
                        algo_name = name
                        clicked_algo = True
                if not clicked_algo:
                    for name, btn in renderer.control_buttons.items():
                        if btn.contains(pos):
                            if name == "new":
                                game.new_maze()
                                auto_playing = False
                            elif name == "reset":
                                game.reset_current()
                                auto_playing = False
                            elif name == "auto":
                                auto_playing = not auto_playing
                                if auto_playing:
                                    game.start_timer_if_needed()
                                    last_step_time = pygame.time.get_ticks()
                            elif name == "visited":
                                show_visited = not show_visited
                    renderer.slider.handle_mousedown(pos)

            elif event.type == pygame.MOUSEMOTION:
                renderer.slider.handle_mousemotion(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                renderer.slider.handle_mouseup()

        if game.status == "playing":
            game._check_time_limit()

        if auto_playing and game.status == "playing":
            if algo_name == "instinct":
                for _ in range(STEPS_PER_FRAME_INSTINCT):
                    if game.status != "playing":
                        break
                    result = game.step(algo_name, renderer.slider.value)
                    if result.get("ate_cheese"):
                        auto_playing = False
                        break
            else:
                now = pygame.time.get_ticks()
                if now - last_step_time >= auto_interval_ms:
                    result = game.step(algo_name, renderer.slider.value)
                    last_step_time = now
                    if result.get("ate_cheese"):
                        auto_playing = False

        renderer.draw(game, algo_name, show_visited, auto_playing)
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
