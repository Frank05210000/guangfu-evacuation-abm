# GUI Evac with Terrain/Flood（現版）

## 功能
- 地形遮罩：河道不可通行；山區可通行但減速。遮罩計算 downsample 至 200x200，渲染放大至 700x700；背景 `data/background.jpeg`（或 `image.png`）若存在會顯示。
- 固定避難所：從遮罩合併（4 鄰域）取得質心，避難所位置固定（綠色）。
- 人口：使用 spawnable/pop_allowed 遮罩（扣除河/山）且需與避難所同連通分量，最近避難所配對。成人/慢速速度可調。
- 洪水：依 Flood interval 反覆暴漲，每次一圈，直到 Flood steps；暴漲同時代理人持續移動。
- 控制（滑鼠）：Start/Pause、Reset Agents（重置洪水與代理人）、Instant Run、Flood toggle、Regenerate（重載遮罩/重建世界）。
- 滑條：Population、Deadline、dt、Adult speed、Slow speed、Slow share、Flood interval、Flood steps。
- 指標：Arrived/Failed/Moving、Mean/P90、Flood 進度。

## 執行
```bash
cd 基礎架構
pip install pygame pillow numpy
python src/app.py
```
