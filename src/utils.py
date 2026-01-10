from __future__ import annotations

import pygame
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
from pathlib import Path

from config import WORLD_SIZE, VIEW_SIZE, TARGET_GRID


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
