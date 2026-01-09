# GUI Evac ABM Demo (rebuilt)

- 1000x1000 world, render 700x700, background `image.png` if present.
- Masks downsampled to 200x200 for computation; render scaled up.
- River: `data/main_river_mask_1000.npy` if present, else `masks_1000_user_corrected.npz:river_mask`; impassable.
- Mountain: `mountain_mask_corrected` (passable, 0.45x speed).
- Population masks: `spawnable_no_river_mask` (or spawnable/pop_allowed), minus river/mountain.
- Shelters: `shelter_mask_center_merged` (or `shelter_mask_center_1000_v2.npy`), merged by 4-neighbor; centroid shown; routing nudges to nearest walkable if blocked.
- UI (Pygame, mouse): Start/Pause, Reset Agents (resets flood), Instant Run, Flood toggle; sliders for Population, Clusters, Cluster sigma, Cluster min dist, Deadline, dt, Adult/Slow speed, Slow share, Flood interval (s), Flood steps.
- Movement: straight toward shelter; mountains slow; river blocks; segment avoids blocked cells; deadline marks remaining as failed.
- Flood: interval = Flood interval slider (default 0.5s), steps default 200 (1-200). Each interval expands one ring until steps reached; runs alongside agent movement (per-frame cap to keep smooth). Instant Run uses higher cap. Interval is fixed (no shrink).
- Defaults: population 1000, deadline 30s, adult 2.5, slow 1.3, flood interval 0.5s, steps 200.

Run:
```
cd 基礎架構
pip install pygame pillow numpy
python pygame_demo.py
```
