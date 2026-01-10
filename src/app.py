from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pygame

from config import (
    WIN_W,
    WIN_H,
    VIEW_SIZE,
    COLOR_BG,
    COLOR_CANVAS,
    COLOR_MOUNTAIN,
    COLOR_SHELTER,
    COLOR_AGENT,
    COLOR_ARRIVED,
    COLOR_FAILED,
)
from world import Params, World
from ui import Button, Slider
from utils import load_image, mask_to_surface, world_to_screen


class App:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 16)
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
            Button((820, 30, 200, 32), "Start / Pause", lambda: None, toggle=True, font=self.font),
            Button((820, 70, 200, 32), "Reset Agents", self.reset_agents, font=self.font),
            Button((820, 110, 200, 32), "Instant Run", self.instant_run, font=self.font),
            Button((820, 150, 200, 32), "Flood toggle", lambda: setattr(self.params, "flood_enable", not self.params.flood_enable), toggle=True, font=self.font),
            Button((820, 190, 200, 32), "Regenerate", self.regenerate, font=self.font),
        ]
        self.sliders = [
            Slider((820, 240, 200, 16), "Population", 100, 3000, 50, "population", "{:.0f}", self.font),
            Slider((820, 270, 200, 16), "Deadline T", 2, 200, 2, "deadline", "{:.0f}", self.font),
            Slider((820, 300, 200, 16), "dt", 0.02, 0.5, 0.01, "dt", "{:.2f}", self.font),
            Slider((820, 330, 200, 16), "Adult speed", 0.5, 10.0, 0.1, "adult_speed", "{:.1f}", self.font),
            Slider((820, 360, 200, 16), "Slow speed", 0.2, 5.0, 0.1, "slow_speed", "{:.1f}", self.font),
            Slider((820, 390, 200, 16), "Slow share", 0.0, 1.0, 0.05, "slow_share", "{:.2f}", self.font),
            Slider((820, 420, 200, 16), "Flood interval", 0.05, 5.0, 0.05, "flood_interval", "{:.2f}", self.font),
            Slider((820, 450, 200, 16), "Flood steps", 1, 200, 1, "flood_steps", "{:.0f}", self.font),
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
            txt = self.font.render(line, True, (20, 20, 20))
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
    from config import COLOR_RIVER  # imported late to avoid circular
    App().run()
