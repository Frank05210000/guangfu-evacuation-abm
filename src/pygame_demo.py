"""
Pygame Evac Demo with Terrain/Flood (class-based)
- Masks downsampled to 200x200 for computation; render scaled to 700x700
- River impassable; mountain passable with slowdown; shelters fixed from masks
- Flood surges every interval (flood_interval) until flood_steps; agents keep moving
- UI (mouse): Start/Pause, Reset Agents (resets flood+agents), Instant Run, Flood toggle, Regenerate
- Sliders: Population, Deadline, dt, Adult speed, Slow speed, Slow share, Flood interval, Flood steps
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from PIL import Image

WORLD_SIZE = 1000
VIEW_SIZE = 700
WIN_W, WIN_H = 1040, 760
TARGET_GRID = 200

COLOR_BG = (245, 245, 245)
COLOR_CANVAS = (230, 230, 230)
COLOR_SHELTER = (44, 160, 44)
COLOR_AGENT = (119, 119, 119)
COLOR_ARRIVED = (0, 170, 0)
COLOR_FAILED = (200, 50, 50)
COLOR_MOUNTAIN = (80, 140, 80)
COLOR_RIVER = (60, 150, 220)

MOUNTAIN_SLOW = 0.45


def load_image(path: Path, size: int) -> Optional[pygame.Surface]:
    if not path.exists():
        return None
    try:
        im = Image.open(path).convert("RGBA").resize((size, size))
        mode = im.mode
        data = im.tobytes()
        return pygame.image.fromstring(data, im.size, mode)
    except Exception:
        return None


def downsample_mask(mask: np.ndarray, target: int = TARGET_GRID) -> np.ndarray:
    H, W = mask.shape
    if H == target and W == target:
        return mask.astype(bool)
    bh = H // target
    bw = W // target
    mask = mask[: bh * target, : bw * target]
    reshaped = mask.reshape(target, bh, target, bw)
    return reshaped.any(axis=3).any(axis=1)


def dilate(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(steps):
        shifted = np.zeros_like(out)
        shifted[1:] |= out[:-1]
        shifted[:-1] |= out[1:]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out |= shifted
    return out


def components(walkable: np.ndarray) -> np.ndarray:
    comp = -np.ones_like(walkable, dtype=int)
    cid = 0
    H, W = walkable.shape
    for i in range(H):
        for j in range(W):
            if not walkable[i, j] or comp[i, j] != -1:
                continue
            stack = [(i, j)]
            comp[i, j] = cid
            while stack:
                ci, cj = stack.pop()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < H and 0 <= nj < W and walkable[ni, nj] and comp[ni, nj] == -1:
                        comp[ni, nj] = cid
                        stack.append((ni, nj))
            cid += 1
    return comp


def merge_shelters(mask: np.ndarray, block: np.ndarray) -> List[Tuple[float, float]]:
    if mask is None:
        return []
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    shelters = []
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            coords = []
            while stack:
                ci, cj = stack.pop()
                coords.append((ci, cj))
                for di, dj in dirs:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < H and 0 <= nj < W and mask[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        stack.append((ni, nj))
            arr = np.array(coords, dtype=float)
            ci, cj = arr.mean(axis=0)
            x = (cj + 0.5) / W * WORLD_SIZE
            y = (ci + 0.5) / H * WORLD_SIZE
            if block[int(ci), int(cj)]:
                found = False
                for rad in range(1, 10):
                    for di in range(-rad, rad + 1):
                        for dj in range(-rad, rad + 1):
                            ni, nj = int(ci) + di, int(cj) + dj
                            if 0 <= ni < H and 0 <= nj < W and not block[ni, nj]:
                                x = (nj + 0.5) / W * WORLD_SIZE
                                y = (ni + 0.5) / H * WORLD_SIZE
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            shelters.append((x, y))
    return shelters


def mask_to_surface(mask: np.ndarray, color: Tuple[int, int, int], alpha: int = 120, scale: int = VIEW_SIZE) -> pygame.Surface:
    H, W = mask.shape
    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    arr_rgb = np.zeros((W, H, 3), dtype=np.uint8)
    arr_rgb[mask.T] = color
    pygame.surfarray.blit_array(surf, arr_rgb)
    alpha_arr = pygame.surfarray.pixels_alpha(surf)
    alpha_arr[:, :] = 0
    alpha_arr[mask.T] = alpha
    del alpha_arr
    return pygame.transform.scale(surf, (scale, scale))


def world_to_screen(x: float, y: float) -> Tuple[int, int]:
    scale = VIEW_SIZE / WORLD_SIZE
    return int(x * scale), int(y * scale)


@dataclass
class Params:
    population: int = 1000
    deadline: float = 200.0
    dt: float = 0.1
    adult_speed: float = 2.5
    slow_speed: float = 1.3
    slow_share: float = 0.35
    flood_enable: bool = True
    flood_interval: float = 3.8
    flood_steps: int = 48


@dataclass
class Agent:
    x: float
    y: float
    speed: float
    target: int
    arrived: bool = False
    failed: bool = False
    arrive_t: float = 0.0


class World:
    def __init__(self, params: Params, base: Path):
        self.params = params
        self.base = base
        self._load_masks()
        self._init_world()

    def _load_masks(self):
        data_dir = self.base / "data"
        npz_path = data_dir / "masks_1000_with_main_river.npz"
        river_main = data_dir / "main_river_mask_1000.npy"
        shelter_ext = data_dir / "shelter_mask_center_1000_v2.npy"
        masks: Dict[str, np.ndarray] = {}
        if npz_path.exists():
            z = np.load(npz_path)
            for k in z.files:
                masks[k] = z[k].astype(bool)
        if river_main.exists():
            masks["river_mask"] = np.load(river_main).astype(bool)
        if shelter_ext.exists():
            masks["shelter_mask_center_merged"] = np.load(shelter_ext).astype(bool)

        river = masks.get("main_river_mask", masks.get("river_mask", np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=bool)))
        mountain = masks.get("mountain_mask_corrected", masks.get("mountain_mask", np.zeros_like(river)))
        spawnable = masks.get("spawnable_no_river_mask", masks.get("spawnable_mask", np.ones_like(river)))
        pop_allowed = masks.get("population_allowed_mask", np.ones_like(river))
        shelter_mask = masks.get("shelter_center_mask", masks.get("shelter_mask_center_merged", masks.get("shelter_mask_center_from_redcircles", None)))

        self.river = downsample_mask(river)
        self.mountain = downsample_mask(mountain)
        self.spawnable = downsample_mask(spawnable)
        self.pop_allowed = downsample_mask(pop_allowed)
        self.shelter_mask = downsample_mask(shelter_mask) if shelter_mask is not None else None

        self.spawnable = self.spawnable & (~self.river) & (~self.mountain)
        self.pop_allowed = self.pop_allowed & (~self.river) & (~self.mountain)

    def _init_world(self):
        self.walkable = ~self.river
        self.shelters = merge_shelters(self.shelter_mask, self.river)
        if not self.shelters:
            self.shelters = [(random.uniform(20, WORLD_SIZE - 20), random.uniform(20, WORLD_SIZE - 20)) for _ in range(10)]
        self.comp = components(self.walkable)
        self.shelter_comp = [
            self.comp[int(s[1] / WORLD_SIZE * self.comp.shape[0]), int(s[0] / WORLD_SIZE * self.comp.shape[1])]
            for s in self.shelters
        ]
        self.flood_mask = self.river.copy()
        self.flood_steps_done = 0
        self.flood_accum = 0.0
        self.agents: List[Agent] = self.sample_agents(self.params.population)
        self.time_s = 0.0

    def sample_agents(self, n: int) -> List[Agent]:
        allowed = (self.spawnable & self.pop_allowed & self.walkable)
        ys, xs = np.where(allowed)
        coords = list(zip(ys, xs))
        random.shuffle(coords)
        agents: List[Agent] = []
        H, W = allowed.shape
        allowed_comp = set(self.shelter_comp)
        while len(agents) < n:
            if coords:
                i, j = coords.pop()
                x = (j + 0.5) / W * WORLD_SIZE
                y = (i + 0.5) / H * WORLD_SIZE
            else:
                x = random.uniform(0, WORLD_SIZE)
                y = random.uniform(0, WORLD_SIZE)
            comp_id = self.comp[int(y / WORLD_SIZE * H), int(x / WORLD_SIZE * W)]
            if comp_id not in allowed_comp:
                continue
            tidx, dmin = -1, 1e9
            for idx, (sx, sy) in enumerate(self.shelters):
                if self.shelter_comp[idx] != comp_id:
                    continue
                d = math.hypot(sx - x, sy - y)
                if d < dmin:
                    dmin = d
                    tidx = idx
            if tidx == -1:
                continue
            is_slow = random.random() < self.params.slow_share
            base_v = self.params.slow_speed if is_slow else self.params.adult_speed
            v = max(0.1, random.gauss(base_v, base_v * 0.08))
            agents.append(Agent(x, y, v, tidx))
        return agents

    def reset_agents(self):
        self.flood_mask = self.river.copy()
        self.flood_steps_done = 0
        self.flood_accum = 0.0
        self.walkable = ~self.flood_mask
        self.comp = components(self.walkable)
        self.shelter_comp = [
            self.comp[int(s[1] / WORLD_SIZE * self.comp.shape[0]), int(s[0] / WORLD_SIZE * self.comp.shape[1])]
            for s in self.shelters
        ]
        self.agents = self.sample_agents(self.params.population)
        self.time_s = 0.0

    def expand_flood(self):
        if self.flood_steps_done >= self.params.flood_steps:
            return
        new_mask = dilate(self.flood_mask, 1)
        new_mask = new_mask & (~self.mountain)
        self.flood_mask = new_mask
        self.flood_steps_done += 1
        self.walkable = ~self.flood_mask
        self.comp = components(self.walkable)
        self.shelter_comp = [
            self.comp[int(s[1] / WORLD_SIZE * self.comp.shape[0]), int(s[0] / WORLD_SIZE * self.comp.shape[1])]
            for s in self.shelters
        ]

    def step_agents(self, dt: float):
        H, W = self.flood_mask.shape
        for a in self.agents:
            if a.arrived or a.failed:
                continue
            sx, sy = self.shelters[a.target]
            dx = sx - a.x
            dy = sy - a.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                a.arrived = True
                a.arrive_t = self.time_s
                continue
            speed = a.speed
            mi = int(a.y / WORLD_SIZE * self.mountain.shape[0])
            mj = int(a.x / WORLD_SIZE * self.mountain.shape[1])
            if self.mountain[mi, mj]:
                speed *= MOUNTAIN_SLOW
            step = speed * dt
            if step >= dist:
                a.x, a.y = sx, sy
                a.arrived = True
                a.arrive_t = self.time_s + dist / speed
            else:
                a.x += dx / dist * step
                a.y += dy / dist * step
            ri = int(a.y / WORLD_SIZE * H)
            rj = int(a.x / WORLD_SIZE * W)
            if self.flood_mask[ri, rj]:
                a.failed = True
        self.time_s += dt
        if self.time_s >= self.params.deadline:
            for a in self.agents:
                if not (a.arrived or a.failed):
                    a.failed = True

    def stats(self):
        total = len(self.agents)
        arrived = sum(a.arrived for a in self.agents)
        failed = sum(a.failed and not a.arrived for a in self.agents)
        moving = total - arrived - failed
        times = [a.arrive_t for a in self.agents if a.arrived]
        mean_t = sum(times) / len(times) if times else float("nan")
        p90 = np.quantile(times, 0.9) if times else float("nan")
        return {
            "total": total,
            "arrived": arrived,
            "failed": failed,
            "moving": moving,
            "mean_t": mean_t,
            "p90": p90,
        }


FONT = None


class Button:
    def __init__(self, rect, text, callback, toggle=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.toggle = toggle
        self.active = False

    def draw(self, surf):
        color = (70, 130, 180) if not self.active else (30, 160, 90)
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        pygame.draw.rect(surf, (40, 40, 40), self.rect, 2, border_radius=6)
        txt = FONT.render(self.text, True, (255, 255, 255))
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle(self, pos):
        if self.rect.collidepoint(pos):
            if self.toggle:
                self.active = not self.active
            self.callback()
            return True
        return False


class Slider:
    def __init__(self, rect, label, minv, maxv, step, value_ref: str, fmt="{:.1f}"):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.minv = minv
        self.maxv = maxv
        self.step = step
        self.value_ref = value_ref
        self.fmt = fmt
        self.dragging = False

    def _val_to_x(self, v):
        return self.rect.x + int((v - self.minv) / (self.maxv - self.minv) * self.rect.w)

    def _x_to_val(self, x):
        r = (x - self.rect.x) / self.rect.w
        v = self.minv + r * (self.maxv - self.minv)
        v = round(v / self.step) * self.step
        return min(self.maxv, max(self.minv, v))

    def draw(self, surf, params: Params):
        v = getattr(params, self.value_ref)
        pygame.draw.rect(surf, (200, 200, 200), self.rect, 2)
        handle_x = self._val_to_x(v)
        pygame.draw.circle(surf, (60, 120, 200), (handle_x, self.rect.centery), 8)
        lbl = FONT.render(f"{self.label}: {self.fmt.format(v)}", True, (30, 30, 30))
        surf.blit(lbl, (self.rect.x, self.rect.y - 18))

    def handle(self, event, params: Params):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
            return True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if self.dragging and event.type == pygame.MOUSEMOTION:
            v = self._x_to_val(event.pos[0])
            setattr(params, self.value_ref, v)
            return True
        return False


class App:
    def __init__(self):
        global FONT
        pygame.init()
        FONT = pygame.font.SysFont("Arial", 16)
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Evac with Terrain/Flood")
        self.clock = pygame.time.Clock()
        self.base = Path(__file__).resolve().parent.parent

        self.params = Params()
        self.world = World(self.params, self.base)

        data_dir = self.base / "data"
        self.bg = load_image(data_dir / "background.jpeg", VIEW_SIZE) or load_image(self.base / "background.jpeg", VIEW_SIZE)
        if self.bg is None:
            self.bg = load_image(data_dir / "image.png", VIEW_SIZE) or load_image(self.base / "image.png", VIEW_SIZE)

        self.river_surf = self.build_river_surf()
        self.mountain_surf = mask_to_surface(self.world.mountain, COLOR_MOUNTAIN, 90, VIEW_SIZE)
        self.buttons: List[Button] = []
        self.sliders: List[Slider] = []
        self._setup_ui()

    def build_river_surf(self):
        return mask_to_surface(self.world.flood_mask, COLOR_RIVER, 180, VIEW_SIZE)

    def regenerate(self):
        self.world = World(self.params, self.base)
        self.river_surf = self.build_river_surf()
        self.mountain_surf = mask_to_surface(self.world.mountain, COLOR_MOUNTAIN, 90, VIEW_SIZE)
        self.buttons[0].active = False

    def reset_agents(self):
        self.world.reset_agents()
        self.river_surf = self.build_river_surf()
        self.buttons[0].active = False

    def instant_run(self):
        dt = self.params.dt
        while True:
            if self.params.flood_enable and self.world.time_s < self.params.deadline:
                self.world.flood_accum += dt
                expand_limit = 200
                count = 0
                while (
                    self.world.flood_accum >= self.params.flood_interval
                    and self.world.flood_steps_done < self.params.flood_steps
                    and count < expand_limit
                ):
                    self.world.expand_flood()
                    self.world.flood_accum -= self.params.flood_interval
                    self.river_surf = self.build_river_surf()
                    count += 1
            self.world.step_agents(dt)
            if self.world.time_s >= self.params.deadline or all(a.arrived or a.failed for a in self.world.agents):
                break
        self.river_surf = self.build_river_surf()

    def _setup_ui(self):
        self.buttons = [
            Button((820, 30, 200, 32), "Start / Pause", lambda: None, toggle=True),
            Button((820, 70, 200, 32), "Reset Agents", self.reset_agents),
            Button((820, 110, 200, 32), "Instant Run", self.instant_run),
            Button((820, 150, 200, 32), "Flood toggle", lambda: setattr(self.params, "flood_enable", not self.params.flood_enable), toggle=True),
            Button((820, 190, 200, 32), "Regenerate", self.regenerate),
        ]
        self.sliders = [
            Slider((820, 240, 200, 16), "Population", 100, 3000, 50, "population", "{:.0f}"),
            Slider((820, 270, 200, 16), "Deadline T", 2, 200, 2, "deadline", "{:.0f}"),
            Slider((820, 300, 200, 16), "dt", 0.02, 0.5, 0.01, "dt"),
            Slider((820, 330, 200, 16), "Adult speed", 0.5, 10.0, 0.1, "adult_speed", "{:.1f}"),
            Slider((820, 360, 200, 16), "Slow speed", 0.2, 5.0, 0.1, "slow_speed", "{:.1f}"),
            Slider((820, 390, 200, 16), "Slow share", 0.0, 1.0, 0.05, "slow_share", "{:.2f}"),
            Slider((820, 420, 200, 16), "Flood interval", 0.05, 5.0, 0.05, "flood_interval", "{:.2f}"),
            Slider((820, 450, 200, 16), "Flood steps", 1, 200, 1, "flood_steps", "{:.0f}"),
        ]

    def handle_event(self, event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            for b in self.buttons:
                if b.handle(event.pos):
                    return
        for s in self.sliders:
            if s.handle(event, self.params):
                return

    def update(self):
        running = self.buttons[0].active
        if running:
            if self.params.flood_enable and self.world.time_s < self.params.deadline:
                self.world.flood_accum += self.params.dt
                expand_limit = 20
                count = 0
                while (
                    self.world.flood_accum >= self.params.flood_interval
                    and self.world.flood_steps_done < self.params.flood_steps
                    and count < expand_limit
                ):
                    self.world.expand_flood()
                    self.world.flood_accum -= self.params.flood_interval
                    self.river_surf = self.build_river_surf()
                    count += 1
            self.world.step_agents(self.params.dt)
            if self.world.time_s >= self.params.deadline:
                for a in self.world.agents:
                    if not (a.arrived or a.failed):
                        a.failed = True
                self.buttons[0].active = False

    def render(self):
        self.screen.fill(COLOR_BG)
        if self.bg:
            self.screen.blit(self.bg, (20, 20))
        else:
            pygame.draw.rect(self.screen, COLOR_CANVAS, (20, 20, VIEW_SIZE, VIEW_SIZE))
        self.screen.blit(self.mountain_surf, (20, 20))
        self.screen.blit(self.river_surf, (20, 20))
        for sx, sy in self.world.shelters:
            px, py = world_to_screen(sx, sy)
            pygame.draw.rect(self.screen, COLOR_SHELTER, pygame.Rect(20 + px - 6, 20 + py - 6, 12, 12))
        for a in self.world.agents:
            px, py = world_to_screen(a.x, a.y)
            color = COLOR_ARRIVED if a.arrived else COLOR_FAILED if a.failed else COLOR_AGENT
            pygame.draw.circle(self.screen, color, (20 + px, 20 + py), 2)
        pygame.draw.rect(self.screen, (255, 255, 255), (800, 0, WIN_W - 800, WIN_H))
        for b in self.buttons:
            b.draw(self.screen)
        for s in self.sliders:
            s.draw(self.screen, self.params)

        st = self.world.stats()
        info = [
            f"Time: {self.world.time_s:.1f}s / T={self.params.deadline:.0f}s",
            f"Arrived: {st['arrived']} ({st['arrived']/st['total']*100:4.1f}%)",
            f"Failed: {st['failed']} ({st['failed']/st['total']*100:4.1f}%)",
            f"Moving: {st['moving']}",
            f"Mean t: {st['mean_t']:.2f}  P90: {st['p90']:.2f}",
            f"Flood: {self.world.flood_steps_done}/{self.params.flood_steps}",
        ]
        for i, line in enumerate(info):
            txt = FONT.render(line, True, (20, 20, 20))
            self.screen.blit(txt, (820, 520 + i * 20))
        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                self.handle_event(event)
            self.update()
            self.render()
            self.clock.tick(60)


if __name__ == "__main__":
    App().run()
