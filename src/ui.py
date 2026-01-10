from __future__ import annotations

import pygame
from typing import Callable


class Button:
    def __init__(self, rect, text: str, callback: Callable, toggle: bool = False, font=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.toggle = toggle
        self.active = False
        self.font = font

    def draw(self, surf):
        color = (70, 130, 180) if not self.active else (30, 160, 90)
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        pygame.draw.rect(surf, (40, 40, 40), self.rect, 2, border_radius=6)
        if self.font:
            txt = self.font.render(self.text, True, (255, 255, 255))
            surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle(self, pos) -> bool:
        if self.rect.collidepoint(pos):
            if self.toggle:
                self.active = not self.active
            self.callback()
            return True
        return False


class Slider:
    def __init__(self, rect, label: str, minv: float, maxv: float, step: float, value_ref: str, fmt: str = "{:.1f}", font=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.minv = minv
        self.maxv = maxv
        self.step = step
        self.value_ref = value_ref
        self.fmt = fmt
        self.dragging = False
        self.font = font

    def _val_to_x(self, v):
        return self.rect.x + int((v - self.minv) / (self.maxv - self.minv) * self.rect.w)

    def _x_to_val(self, x):
        r = (x - self.rect.x) / self.rect.w
        v = self.minv + r * (self.maxv - self.minv)
        v = round(v / self.step) * self.step
        return min(self.maxv, max(self.minv, v))

    def draw(self, surf, params):
        v = getattr(params, self.value_ref)
        pygame.draw.rect(surf, (200, 200, 200), self.rect, 2)
        handle_x = self._val_to_x(v)
        pygame.draw.circle(surf, (60, 120, 200), (handle_x, self.rect.centery), 8)
        if self.font:
            lbl = self.font.render(f"{self.label}: {self.fmt.format(v)}", True, (30, 30, 30))
            surf.blit(lbl, (self.rect.x, self.rect.y - 18))

    def handle(self, event, params) -> bool:
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
