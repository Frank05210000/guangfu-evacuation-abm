"""
Pygame evacuation ABM with mouse UI (rebuilt from notes):
 - Masks are downsampled to 200x200 for computation, rendering scaled to full view.
 - River blocked; mountains passable with slowdown.
 - Fixed shelters from mask centroids; merged by 4-neighbor.
 - Flood surges every flood_interval (uses flood_time slider), repeats until flood_steps reached; runs alongside agent movement.
 - UI: Start/Pause, Reset Agents (resets flood), Instant Run, Flood toggle; sliders for population, clusters, sigma, min dist, deadline, dt, speeds, slow share, flood interval, flood steps.
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
WIN_W, WIN_H = 1100, 820
VIEW_SIZE = 800
TARGET_GRID = 200  # computation grid
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


@dataclass
class Params:
    n_agents: int = 1000
    clusters: int = 4
    cluster_sigma: float = 90.0
    cluster_min_dist: float = 80.0
    deadline: float = 30.0
    dt: float = 0.1
    adult_speed: float = 2.5
    slow_speed: float = 1.3
    slow_share: float = 0.35
    flood_enable: bool = True
    flood_time: float = 0.5   # interval between surges (seconds)
    flood_steps: int = 200
    flood_interval: float = 0.5  # legacy/unused


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
    def __init__(self, params: Params):
        self.params = params
        self._load_masks()
        self._init_world()

    def _load_masks(self):
        base = Path(__file__).resolve().parent.parent
        npz_path = base / "data/masks_1000_user_corrected.npz"
        river_main = base / "data/main_river_mask_1000.npy"
        shelter_ext = base / "data/shelter_mask_center_1000_v2.npy"
        masks: Dict[str, np.ndarray] = {}
        if npz_path.exists():
            z = np.load(npz_path)
            for k in z.files:
                masks[k] = z[k].astype(bool)
        if river_main.exists():
            masks["river_mask"] = np.load(river_main).astype(bool)
        if shelter_ext.exists():
            masks["shelter_mask_center_merged"] = np.load(shelter_ext).astype(bool)

        river = masks.get("river_mask", np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=bool))
        mountain = masks.get("mountain_mask_corrected", masks.get("mountain_mask", np.zeros_like(river)))
        spawnable = masks.get("spawnable_no_river_mask", masks.get("spawnable_mask", np.ones_like(river)))
        pop_allowed = masks.get("population_allowed_mask", np.ones_like(river))
        shelter_mask = masks.get("shelter_mask_center_merged", masks.get("shelter_mask_center_from_redcircles", None))

        # downsample
        self.river = downsample_mask(river)
        self.mountain = downsample_mask(mountain)
        self.spawnable = downsample_mask(spawnable)
        self.pop_allowed = downsample_mask(pop_allowed)
        self.shelter_mask = downsample_mask(shelter_mask) if shelter_mask is not None else None

        # avoid spawn on river/mountain
        self.spawnable = self.spawnable & (~self.river) & (~self.mountain)
        self.pop_allowed = self.pop_allowed & (~self.river) & (~self.mountain)

    def _init_world(self):
        self.walkable = ~self.river
        self.shelters = merge_shelters(self.shelter_mask, self.river) if self.shelter_mask is not None else []
        self.comp = components(self.walkable)
        self.shelter_comp = [
            self.comp[int(s[1] / WORLD_SIZE * self.comp.shape[0]), int(s[0] / WORLD_SIZE * self.comp.shape[1])]
            for s in self.shelters
        ]
        self.flood_mask = self.river.copy()
        self.flood_steps_done = 0
        self.flood_accum = 0.0
        self.flood_interval_current = max(0.01, self.params.flood_time)
        self.agents: List[Agent] = self.sample_agents(self.params.n_agents)
        self.time_s = 0.0

    def reset(self, params: Params):
        self.params = params
        self._load_masks()
        self._init_world()

    def expand_river(self):
        if self.flood_steps_done >= self.params.flood_steps:
            return
        self.flood_mask = dilate(self.flood_mask, 1)
        self.walkable = ~self.flood_mask
        self.comp = components(self.walkable)
        self.shelter_comp = [
            self.comp[int(s[1] / WORLD_SIZE * self.comp.shape[0]), int(s[0] / WORLD_SIZE * self.comp.shape[1])]
            for s in self.shelters
        ]
        self.flood_steps_done += 1
        self.flood_interval_current = max(0.01, self.params.flood_time)

    def sample_agents(self, n: int) -> List[Agent]:
        allowed = (self.spawnable & self.pop_allowed & self.walkable)
        ys, xs = np.where(allowed)
        coords = list(zip(ys, xs))
        random.shuffle(coords)
        agents: List[Agent] = []
        H, W = allowed.shape
        while len(agents) < n:
            if coords:
                i, j = coords.pop()
                x = (j + 0.5) / W * WORLD_SIZE
                y = (i + 0.5) / H * WORLD_SIZE
            else:
                x = random.uniform(0, WORLD_SIZE)
                y = random.uniform(0, WORLD_SIZE)
            comp = self.comp[int(y / WORLD_SIZE * H), int(x / WORLD_SIZE * W)]
            dmin = 1e9
            tidx = -1
            for idx, (sx, sy) in enumerate(self.shelters):
                if self.shelter_comp[idx] != comp:
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

    def step_agent(self, a: Agent, dt: float):
        if a.arrived or a.failed:
            return
        sx, sy = self.shelters[a.target]
        dx = sx - a.x
        dy = sy - a.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            a.arrived = True
            a.arrive_t += dt
            return
        speed = a.speed
        i = int(a.y / WORLD_SIZE * self.mountain.shape[0])
        j = int(a.x / WORLD_SIZE * self.mountain.shape[1])
        if self.mountain[i, j]:
            speed *= MOUNTAIN_SLOW
        step = speed * dt
        nx = a.x + dx / dist * step
        ny = a.y + dy / dist * step
        ri = int(ny / WORLD_SIZE * self.flood_mask.shape[0])
        rj = int(nx / WORLD_SIZE * self.flood_mask.shape[1])
        if self.flood_mask[ri, rj]:
            a.failed = True
            return
        a.x, a.y = nx, ny
        a.arrive_t += dt
        if a.arrive_t >= self.params.deadline:
            a.failed = True
        if math.hypot(sx - a.x, sy - a.y) < 1.0:
            a.arrived = True

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


def world_to_screen(x: float, y: float) -> Tuple[int, int]:
    scale = VIEW_SIZE / WORLD_SIZE
    return int(x * scale), int(y * scale)


def main():
    global FONT
    pygame.init()
    FONT = pygame.font.SysFont("Arial", 16)
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Pygame Evac ABM (GUI)")
    clock = pygame.time.Clock()

    bg = load_image(Path(__file__).resolve().parent.parent / "image.png", VIEW_SIZE)

    params = Params()
    world = World(params)
    river_surf = mask_to_surface(world.flood_mask, (60, 150, 220), 200, VIEW_SIZE)
    mountain_surf = mask_to_surface(world.mountain, (80, 140, 80), 90, VIEW_SIZE)

    running = False

    buttons: List[Button] = []
    sliders: List[Slider] = []

    def regenerate():
        nonlocal world, river_surf, mountain_surf, running
        world = World(params)
        river_surf = mask_to_surface(world.flood_mask, (60, 150, 220), 200, VIEW_SIZE)
        mountain_surf = mask_to_surface(world.mountain, (80, 140, 80), 90, VIEW_SIZE)
        running = False

    def start_pause():
        nonlocal running
        running = not running

    def reset_agents():
        nonlocal world, running, river_surf
        world.flood_mask = world.river.copy()
        world.walkable = ~world.flood_mask
        world.comp = components(world.walkable)
        world.shelter_comp = [
            world.comp[int(s[1] / WORLD_SIZE * world.comp.shape[0]), int(s[0] / WORLD_SIZE * world.comp.shape[1])]
            for s in world.shelters
        ]
        world.flood_steps_done = 0
        world.flood_accum = 0.0
        world.flood_interval_current = max(0.01, params.flood_time)
        world.agents = world.sample_agents(params.n_agents)
        world.time_s = 0.0
        river_surf = mask_to_surface(world.flood_mask, (60, 150, 220), 200, VIEW_SIZE)
        running = False

    def instant_run():
        nonlocal world, river_surf
        dt = params.dt
        while True:
            if params.flood_enable:
                world.flood_accum += dt
                expand_limit = 200
                count = 0
                while world.flood_accum >= world.flood_interval_current and world.flood_steps_done < params.flood_steps and count < expand_limit:
                    world.expand_river()
                    world.flood_accum -= world.flood_interval_current
                    river_surf = mask_to_surface(world.flood_mask, (60, 150, 220), 200, VIEW_SIZE)
                    count += 1
            for a in world.agents:
                world.step_agent(a, dt)
            world.time_s += dt
            if world.time_s >= params.deadline or all(a.arrived or a.failed for a in world.agents):
                if world.time_s >= params.deadline:
                    for a in world.agents:
                        if not (a.arrived or a.failed):
                            a.failed = True
                break

    buttons.append(Button((840, 40, 220, 32), "Start / Pause", start_pause, toggle=True))
    buttons.append(Button((840, 80, 220, 32), "Reset Agents", reset_agents))
    buttons.append(Button((840, 120, 220, 32), "Instant Run", instant_run))

    sliders.append(Slider((840, 210, 220, 16), "Population", 200, 3000, 50, "n_agents", "{:.0f}"))
    sliders.append(Slider((840, 240, 220, 16), "Clusters", 1, 10, 1, "clusters", "{:.0f}"))
    sliders.append(Slider((840, 270, 220, 16), "Cluster sigma", 20, 250, 5, "cluster_sigma"))
    sliders.append(Slider((840, 300, 220, 16), "Cluster min dist", 20, 500, 10, "cluster_min_dist", "{:.0f}"))
    sliders.append(Slider((840, 330, 220, 16), "Deadline T", 2, 120, 1, "deadline", "{:.0f}"))
    sliders.append(Slider((840, 360, 220, 16), "dt", 0.02, 0.5, 0.01, "dt"))
    sliders.append(Slider((840, 390, 220, 16), "Adult speed", 0.5, 5.0, 0.1, "adult_speed", "{:.1f}"))
    sliders.append(Slider((840, 420, 220, 16), "Slow speed", 0.2, 3.0, 0.1, "slow_speed", "{:.1f}"))
    sliders.append(Slider((840, 450, 220, 16), "Slow share", 0.0, 1.0, 0.05, "slow_share"))
    sliders.append(Slider((840, 480, 220, 16), "Flood interval (s)", 0.05, 5.0, 0.05, "flood_time", "{:.2f}"))
    sliders.append(Slider((840, 510, 220, 16), "Flood steps", 1, 200, 1, "flood_steps", "{:.0f}"))

    flood_toggle = Button((840, 580, 220, 32), "Flood toggle", lambda: setattr(params, "flood_enable", not params.flood_enable), toggle=True)
    buttons.append(flood_toggle)

    def handle_ui_event(event):
        nonlocal world, river_surf, mountain_surf, running
        if event.type == pygame.MOUSEBUTTONDOWN:
            for b in buttons:
                if b.handle(event.pos):
                    return True
        for s in sliders:
            if s.handle(event, params):
                return True
        return False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if handle_ui_event(event):
                if event.type == pygame.MOUSEBUTTONUP:
                    regenerate()

        if running:
            if params.flood_enable:
                world.flood_accum += params.dt
                expand_limit = 20
                count = 0
                while world.flood_accum >= world.flood_interval_current and world.flood_steps_done < params.flood_steps and count < expand_limit:
                    world.expand_river()
                    world.flood_accum -= world.flood_interval_current
                    river_surf = mask_to_surface(world.flood_mask, (60, 150, 220), 200, VIEW_SIZE)
                    count += 1
            for a in world.agents:
                world.step_agent(a, params.dt)
            world.time_s += params.dt
            if world.time_s >= params.deadline:
                for a in world.agents:
                    if not (a.arrived or a.failed):
                        a.failed = True
                running = False

        screen.fill((245, 245, 245))
        if bg:
            screen.blit(bg, (20, 20))
        else:
            pygame.draw.rect(screen, (230, 230, 230), (20, 20, VIEW_SIZE, VIEW_SIZE))
        screen.blit(mountain_surf, (20, 20))
        screen.blit(river_surf, (20, 20))

        for sx, sy in world.shelters:
            px, py = world_to_screen(sx, sy)
            pygame.draw.rect(screen, (44, 160, 44), pygame.Rect(20 + px - 5, 20 + py - 5, 10, 10))
        for a in world.agents:
            px, py = world_to_screen(a.x, a.y)
            color = (0, 160, 0) if a.arrived else (200, 50, 50) if a.failed else (90, 90, 90)
            pygame.draw.circle(screen, color, (20 + px, 20 + py), 2)

        pygame.draw.rect(screen, (255, 255, 255), (820, 0, WIN_W - 820, WIN_H))
        for b in buttons:
            b.draw(screen)
        flood_toggle.active = params.flood_enable
        for s in sliders:
            s.draw(screen, params)

        st = world.stats()
        info = [
            f"Time: {world.time_s:.1f}s / T={params.deadline:.1f}s",
            f"Arrived: {st['arrived']} ({st['arrived']/st['total']*100:4.1f}%)",
            f"Failed: {st['failed']} ({st['failed']/st['total']*100:4.1f}%)",
            f"Moving: {st['moving']}",
            f"Mean t: {st['mean_t']:.2f}  P90: {st['p90']:.2f}",
            f"Flood: {world.flood_steps_done}/{params.flood_steps}",
            f"Interval now: {world.flood_interval_current:.3f}s",
        ]
        for idx, line in enumerate(info):
            txt = FONT.render(line, True, (20, 20, 20))
            screen.blit(txt, (840, 630 + idx * 20))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
