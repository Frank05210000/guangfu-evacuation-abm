from pathlib import Path

# World / view
WORLD_SIZE = 1000
VIEW_SIZE = 700
WIN_W, WIN_H = 1040, 760
TARGET_GRID = 200  # computation grid (downsampled)

# Colors
COLOR_BG = (245, 245, 245)
COLOR_CANVAS = (230, 230, 230)
COLOR_SHELTER = (44, 160, 44)
COLOR_AGENT = (119, 119, 119)
COLOR_ARRIVED = (0, 170, 0)
COLOR_FAILED = (200, 50, 50)
COLOR_MOUNTAIN = (80, 140, 80)
COLOR_RIVER = (60, 150, 220)

MOUNTAIN_SLOW = 0.45

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
