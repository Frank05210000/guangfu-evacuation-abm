# 花蓮光復洪水避難模擬（Terrain/Flood 版）

本專案是一個 Pygame 的互動模擬，使用既有的遮罩資料（河流、山區、人口生成區、避難所）與簡化洪水暴漲規則，觀察代理人撤離到固定避難所的過程。

## 目前狀態（與原始 spec 的差異）
- 使用遮罩決定地形與避難所（固定位置），而非隨機避難所。
- 河流不可通行；山區可通行但減速。洪水會以固定間隔向外膨脹（不覆蓋山區）。
- 下採樣：所有遮罩計算在 200x200 上，渲染放大至 700x700。
- 代理人：從遮罩允許區生成，配對同連通分量最近避難所，直線移動；無容量、無道路網。
- UI：Start/Pause、Reset Agents（重置洪水與人口）、Instant Run、Flood toggle、Regenerate；滑條包含人口、期限、dt、速度、慢速比例、洪水間隔/步數。

## 專案結構
```
基礎架構/
├── src/app.py                 # 主程式入口（類別化，滑鼠 UI）
├── src/world.py               # 模擬核心（遮罩載入、洪水、代理人）
├── src/utils.py               # 輔助函式（遮罩處理、繪圖轉換）
├── src/ui.py                  # Button / Slider UI 元件
├── src/config.py              # 常數與顏色設定
├── data/
│   ├── masks_1000_with_main_river.npz  # river/mountain/spawnable/shelter 遮罩
│   └── background.jpeg                 # 背景圖（可缺）
├── docs/
│   ├── README_gui_demo.md     # 現版功能與執行說明
│   ├── spec.md                # 需求筆記（原始 spec）
│   └── assets/預期UI.png       # 介面示意
└── __pycache__/               # 執行產生，可忽略
```

## 執行
```bash
cd 基礎架構
pip install pygame pillow numpy
python src/app.py
```

## 介面與控制摘要
- 按鈕：Start/Pause、Reset Agents、Instant Run、Flood toggle、Regenerate
- 滑條：Population、Deadline、dt、Adult speed、Slow speed、Slow share、Flood interval、Flood steps
- 指標：Arrived/Failed/Moving、Mean/P90、Flood 進度

## 待補/限制
- 避難所為遮罩固定，無容量設定；無道路網距離、無隨機避難所模式。
- 洪水為簡化「固定間隔一圈」模型，僅做阻擋；山區不會被淹。
- 若需回到原始 spec（隨機避難所、無地形/洪水），可另開分支或再提供遮罩/需求。***
