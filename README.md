# ⚡ Ireland Energy Transition — Decision Intelligence Case Study  
### From System Volatility → Insight → Strategy → Simulation  

![Banner](https://user-images.githubusercontent.com/your-banner-image.png)

> **Author:** [Rijo Mathew John](https://www.linkedin.com/in/rijomathewjohn)  
> **Degree:** MSc Data Analytics | Decision Intelligence & Operations Analytics  
> **Live Demo:** 🎯 [Launch Interactive App](https://ireland-energy-simulator-v5fozxllkk9lnjxamurzzw.streamlit.app/?wind=0&demand=0&smooth=0)  
> **Built With:** Python • Streamlit • Plotly • scikit-learn • ENTSO-E • Meteostat

---

## 🧭 1. Project Overview

Ireland’s electricity grid is in constant tension — balancing **Affordability**, **Reliability**, and **Sustainability**.

This project builds a **Decision Intelligence framework** that connects:
- Hourly **generation & demand** (ENTSO-E)
- **Weather impacts** (Meteostat)
- **Market price behavior**
- **Scenario simulations** for energy policy & operations

It’s not just visualization — it’s a working model that **tests decisions before implementing them**.

---

## 🌍 2. Problem Statement

> Ireland’s grid shows high **price sensitivity** to renewable volatility.  
> When wind generation drops, stress hours spike and prices surge disproportionately.

This project explores:
- How weather and renewables shape market volatility  
- How system stress relates to cost and reliability  
- What strategies can stabilize the grid without raising costs

---

## 🎯 3. Goal

To design a **data-driven decision support tool** that quantifies the trade-offs between:
- 💶 **Price stability**  
- ⚙️ **Grid reliability**  
- 🌱 **Renewable consistency**

And to simulate how operational policies — like increasing wind, smoothing renewables, or managing demand — affect these KPIs in real time.

---

## 🧩 4. Methodology

### In Simple Terms
1. Combine hourly **generation, demand, weather, and price** data for one week (Oct 18–25 2025).  
2. Find patterns — where do **prices jump** and **stress hours** occur?  
3. Train a model that links price to real system variables (renewable share, load, volatility).  
4. Build **sliders** to simulate “what-if” scenarios — e.g., *What if wind output grows 20%?*  
5. Observe how that shifts the grid’s performance metrics.

### In Technical Terms
- **Data Sources:**  
  - ENTSO-E Transparency Platform — Load, Generation, Market Prices  
  - Meteostat — Weather (wind, temperature, sunshine)
- **Processing:**  
  - Python (Pandas, NumPy) with hourly normalization  
  - Star-schema “mart” of load, generation, renewables, weather, price
- **Modeling:**  
  - Ridge Regression → `price ~ ren_share + stress + load_scaled (+ weather)`  
  - Auto-handling of NaN/Inf + feature scaling (scikit-learn Pipeline)
- **Simulation Engine:**  
  - Dynamic parameter adjustment via Streamlit sliders  
  - KPI comparison (avg price, stress %, RSD of renewables)  
  - Real-time Plotly dashboards

---

## 🔬 5. Exploratory Insights

| Insight | Observation |
|:--|:--|
| 💨 Wind volatility | High variability (RSD > 0.25) directly drives price spikes |
| ⚙️ Stress hours | When generation < demand, stress > 10 % → price surges |
| 🌅 Peak hours | 17:00 – 21:00 hrs remain high-stress even on stable days |
| 💶 Price-renewable link | Prices fall almost linearly with renewable % > 60 % |

**Gap Identified:**  
The grid is *too reactive* — a small renewable dip triggers large market swings.  
→ Ireland needs more **resilience, stability, and foresight**.

---

## 🚀 6. Call to Action (Strategic Recommendations)

| Action | Impact |
|:--|:--|
| **Increase Wind Penetration (+20%)** | Reduces avg price / stress hours |
| **Stabilize Renewables (–25% RSD)** | Smoother supply curve, better reliability |
| **Prepare for Demand Growth (+10%)** | Requires flexible storage / responsive generation |
| **Hybrid Strategy** | Combines cost reduction + resilience gain |

---

## 🧠 7. Decision Intelligence Simulator

> 🎯 Try it yourself: [**Launch Interactive App →**](https://ireland-energy-simulator-v5fozxllkk9lnjxamurzzw.streamlit.app/?wind=0&demand=0&smooth=0)

**Interactive Controls**
- Adjust `Wind`, `Demand`, and `Stability` sliders  
- Observe real-time KPI shifts and price trajectories  

**KPIs**
- 💶 Average Price  
- ⚙️ Stress Hours  
- 🌱 Renewable Stability (RSD)

**Technically**
```text
Model: Ridge Regression (scikit-learn)
Features: ren_share, stress, load_scaled, wind_speed, sunshine_fraction
Data Window: Oct 18–25 2025 (hourly)

---

## 📈 8. Key Results

| Metric                    | Baseline        | Simulated (Hybrid) | Δ Change           |
| :------------------------ | :-------------- | :----------------- | :----------------- |
| Average Price (€/MWh)     | ↓ From live app | ↓                  | Cost reduction     |
| Stress Hours (%)          | ↓               | ↓                  | Higher reliability |
| Renewable Stability (RSD) | ↓               | ↓                  | Smoother operation |

📊 The **hybrid scenario (Wind + Stability)** gave the **best multi-objective balance**.

---

## 🧱 9. Architecture Overview

```text
ENTSO-E Data  ─┬─> Load  ─┐
                ├─> Generation  ─┐
Meteostat Data ─┘               │
                                 ├─> Data Mart (hourly)
                                 │
                                 ├─> EDA + KPI Engine
                                 │
                                 └─> Simulation (Ridge Model)
                                         ↓
                                Streamlit Decision Layer
```

---

## 🧰 10. Tech Stack

| Layer           | Tools Used                      |
| :-------------- | :------------------------------ |
| Data Processing | Python (pandas, numpy)          |
| Modeling        | scikit-learn (Ridge Regression) |
| Visualization   | Plotly Express + Graph Objects  |
| App Layer       | Streamlit (Dark Theme UI)       |
| Data Sources    | ENTSO-E • Meteostat             |
| Deployment      | Streamlit Cloud + GitHub        |

---

## 🧾 11. Repository Structure

```
📂 Ireland-Energy-Simulator
│
├── app.py                           # Streamlit Case Study + Simulator
├── data/
│   └── processed/
│       └── mart_ie_hourly_system_kpis.csv
├── requirements.txt
├── .streamlit/
│   └── config.toml                  # Dark theme setup
├── README.md                        # This file
└── runtime.txt                      # Python version
```

---

## ✨ 12. What’s Unique (“My Uniqueness”)

> **Decision Intelligence for Energy Balance** — not just a dashboard.

This project fuses **operational data analytics** with **strategic simulation**, quantifying trade-offs between:

* Cost 🪙
* Reliability ⚙️
* Sustainability 🌱

It converts static energy reporting into a **decision-making tool** — a glimpse of how future smart grids will be managed.

---

## 🧑‍💻 13. How to Run Locally

```bash
# clone the repository
git clone https://github.com/rijomj008-create/Ireland-Energy-Simulator.git
cd Ireland-Energy-Simulator

# install dependencies
pip install -r requirements.txt

# run locally
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501)

---

## 🎓 14. About the Author

**Rijo Mathew John**
📍 Dublin, Ireland
🎓 MSc Data Analytics — Dublin Business School
💼 Decision Intelligence | Operations Analytics | Energy Systems
📧 [rijomj008@gmail.com](mailto:rijomj008@gmail.com)
🔗 [LinkedIn →](https://www.linkedin.com/in/rijomathewjohn)

---

## 🏁 15. License

MIT License — You’re welcome to reuse with attribution.

---

### ⭐ If you found this project useful, please star ⭐ the repository — it helps others discover Decision Intelligence for Energy.

```

---

### ✅ How this aligns with your vision
- **Narrative-first** flow (Problem → Goal → Methodology → Simulator → Findings → CTA).  
- Mix of **layman & analytical tone** for recruiters and domain experts alike.  
- **Polished formatting** with icons, tables, and clickable demo link.  
- **Future-proof** — you can reuse this layout for other Decision Intelligence projects.

Would you like me to include a matching `config.toml` theme section (so the app dark mode matches this README’s aesthetic)?
```
