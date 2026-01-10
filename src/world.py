from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import WORLD_SIZE, MOUNTAIN_SLOW
from utils import downsample_mask, components, merge_shelters, dilate


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
