# FinShield — Agentic AI Fraud Detection System

FinShield ek **multi-agent fraud detection system** hai jo real-time mein transactions ko analyze karke batata hai ki woh fraud hai ya nahi. Isme 4 specialized AI agents hain + 1 LLM Decision Agent jo final verdict deta hai.

**Live Demo:** [finshield.onrender.com](https://finshield.onrender.com)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Tailwind CSS, Framer Motion, Axios |
| Backend | FastAPI, Uvicorn, Python 3.12 |
| LLM | Groq Cloud API (`llama-3.3-70b-versatile`) |
| Framework | LangChain + LangGraph |
| Deployment | Render (Docker) |

---

## Architecture — Kaise Kaam Karta Hai

```
User Transaction
       │
       ▼
┌─────────────────────────────────────────────┐
│           LLM Decision Agent (Orchestrator) │
│                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│   │Behavioral│  │ Temporal │  │   Geo    │ │
│   │  Agent   │  │  Agent   │  │  Agent   │ │
│   │ w=0.30   │  │ w=0.20   │  │ w=0.25   │ │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│        │             │             │        │
│   ┌────┴─────────────┴─────────────┴────┐   │
│   │         Device Agent (w=0.25)       │   │
│   └─────────────────────────────────────┘   │
│                                             │
│   Final: R_composite = Σ(wᵢ · rᵢ) / Σ(wᵢ) │
└─────────────────────────────────────────────┘
       │
       ▼
  ALLOW / REVIEW / BLOCK
```

Har agent apna kaam karta hai, apna risk score (0 to 1) deta hai, phir **LLM Decision Agent** sabke signals ko mila ke final verdict deta hai.

---

## Agent 1: Behavioral Agent (Weight = 0.30)

### Kya karta hai
Customer ki past transaction history se compare karke dekhta hai ki current transaction kitni "unusual" hai.

### Mathematical Formulas

**Z-Score (Standard Score):**
```
Z = (x - μ) / σ
```
- `x` = current transaction amount
- `μ` = customer ki average transaction amount (mean)
- `σ` = standard deviation
- Agar Z > 2 hai toh transaction bohot unusual hai

**Statistical Risk (Sigmoid-smoothed):**
```
risk = σ(|Z| - 1.5) = 1 / (1 + e^(-(|Z| - 1.5)))
```
- Sigmoid function use karte hain taaki risk smoothly 0 se 1 ke beech aaye
- Threshold = 1.5 standard deviations

**CVV Verification Risk:**
```
M (Match)                  → 0.0
N (No Match)               → 0.9
P (Not Processed)          → 0.4
S (Should be present)      → 0.7
U (Uncertified)            → 0.3
Empty/Unknown              → 0.5
```

**Account Age Risk:**
```
risk = σ(-(age - 30) / 20) = 1 / (1 + e^((age - 30) / 20))
```
- Naya account (age=0) → risk = 0.82
- 30 din purana account → risk = 0.50
- 90 din purana → risk = 0.05

**Recent Payment Rejects Risk:**
```
0 rejects → 0.0
1 reject  → 0.4
2+ rejects → 0.8
```

### Flow
1. Customer ki history se mean (μ) aur std deviation (σ) calculate karo
2. Z-Score nikalo → Sigmoid se statistical risk banao
3. CVV, Account Age, Recent Rejects ke enrichment signals nikalo
4. Ye sab LLM ko do → LLM final behavioral risk (0-1) aur label (Low/Medium/High) deta hai

---

## Agent 2: Temporal Agent (Weight = 0.20)

### Kya karta hai
Transaction ka timing check karta hai — kya customer apne usual time pe transaction kar raha hai ya kisi unusual ghante mein.

### Mathematical Formulas

**Circular Hour Distance:**
```
d = min(|h₁ - h₂|, 24 - |h₁ - h₂|)
```
- 24-hour clock circular hai (23:00 aur 01:00 mein sirf 2 ghante ka farak hai, 22 nahi)
- `h₁` = transaction hour, `h₂` = customer ka average active hour

**Exponential Decay Risk:**
```
risk = 1 - e^(-λ · d)
```
- `λ` = 0.3 (decay rate)
- `d` = circular hour distance
- Jitna zyada deviation, utna zyada risk
- d=0 → risk=0, d=6h → risk=0.83, d=12h → risk=0.97

**Hour Frequency Score:**
```
freq = count_at_hour / total_count
```
- Customer kitni baar is hour pe transaction karta hai
- Frequency risk = 1 - freq (kam frequency = zyada risk)

**Composite Temporal Risk:**
```
R_temporal = 0.6 × decay_risk + 0.4 × (1 - freq)
```

### Flow
1. Transaction ka hour nikalo (localHour prefer karo, warna timestamp se)
2. Customer ki past hours ki distribution nikalo
3. Circular distance + Exponential decay risk calculate karo
4. Hour frequency check karo
5. Composite score LLM ko do → LLM final temporal risk deta hai

---

## Agent 3: Geographic (Geo) Agent (Weight = 0.25)

### Kya karta hai
Transaction ki location check karta hai — kya customer apni usual jagah se transaction kar raha hai, kya impossible travel ho raha hai, aur kya countries match kar rahe hain.

### Mathematical Formulas

**Haversine Distance (do points ke beech Earth pe distance):**
```
a = sin²(Δlat/2) + cos(lat₁) · cos(lat₂) · sin²(Δlon/2)
c = 2 · atan2(√a, √(1-a))
distance = R · c     (R = 6371 km)
```

**Sigmoid-smoothed Distance Risk:**
```
risk = σ((d - 100) / 60) = 1 / (1 + e^(-(d - 100) / 60))
```
- 100 km ke andar → low risk
- 100 km ke baad risk rapidly badhta hai

**Impossible Travel Velocity:**
```
velocity = haversine_distance / Δt    (km/h)
```
| Velocity | Risk |
|----------|------|
| > 900 km/h | 0.95 (airplane se bhi fast) |
| > 500 km/h | 0.75 |
| > 200 km/h | 0.50 |
| ≤ 200 km/h | 0.10 |

**Country Mismatch Risk:**
```
Har mismatch = +0.3 risk (max 0.9)
```
- IP Country ≠ Shipping Country
- Shipping Country ≠ Billing Country
- IP Country ≠ Account Country

**Composite Geo Risk:**
```
R_geo = 0.35 × dist_risk + 0.30 × velocity_risk + 0.35 × mismatch_risk
```

### Flow
1. Transaction ki coordinates resolve karo (lat/lon ya city name se)
2. Historical locations se Haversine distance nikalo
3. Last transaction se travel velocity check karo
4. Country mismatches detect karo
5. Composite score LLM ko do → LLM final geo risk deta hai

---

## Agent 4: Device Agent (Weight = 0.25)

### Kya karta hai
Device fingerprinting karta hai — kya device naya hai, kya proxy IP use ho raha hai, kya browser/device type consistent hai customer ke saath.

### Mathematical Formulas

**Shannon Entropy (Device Diversity):**
```
H = -Σ pᵢ · log₂(pᵢ)
```
- `pᵢ` = probability of each device in customer history
- High entropy = bohot saare devices use karta hai (unpredictable)
- Low entropy = consistent device usage

**Device Frequency Ratio:**
```
f_device = n_device / N_total
```
- Current device kitni baar history mein aaya hai

**Proxy IP Risk:**
```
isProxyIP = TRUE  → 0.85
isProxyIP = FALSE → 0.0
```

**Browser/Device Type Consistency:**
```
New browser (never seen)      → 0.4
New device type (never seen)  → 0.3
Known browser/device          → 0.0
```

**Composite Device Risk:**
```
R_device = 0.25 × known_signal
         + 0.15 × entropy_norm
         + 0.15 × (1 - freq)
         + 0.20 × proxy_risk
         + 0.10 × browser_risk
         + 0.15 × device_type_risk
```

### Flow
1. Device ID history mein hai ya nahi check karo
2. Shannon Entropy se device diversity measure karo
3. Device frequency ratio nikalo
4. Proxy IP, browser consistency, device type consistency check karo
5. Composite score LLM ko do → LLM final device risk deta hai

---

## Agent 5: LLM Decision Agent (Final Orchestrator)

### Kya karta hai
Ye sabse important agent hai. Ye baaki 4 agents ke results ko synthesize karke final decision deta hai.

### Flow
1. Pehle Behavioral → Temporal → Geo → Device agents ko sequentially run karta hai
2. Sabke risk scores, labels, aur reasons collect karta hai
3. LLM (`llama-3.3-70b-versatile` via Groq) ko saara data deta hai
4. LLM final verdict deta hai: `ALLOW`, `REVIEW`, ya `BLOCK`

### Weighted Composite Risk (Final Score)
```
R_composite = Σ(wᵢ · rᵢ) / Σ(wᵢ)

= (0.30 × behavioral_risk
 + 0.20 × temporal_risk
 + 0.25 × geo_risk
 + 0.25 × device_risk) / 1.0
```

### Fallback (agar LLM fail ho jaye)
```
Koi bhi agent "High" → BLOCK
Koi bhi agent "Medium" → REVIEW
Sab "Low" → ALLOW
```

---

## Datasets Used

### 1. Indian UPI Fraud Dataset
- **Source:** Kaggle
- **Link:** [https://www.kaggle.com/datasets/dhatribadri/indian-upi-fraud-data](https://www.kaggle.com/datasets/dhatribadri/indian-upi-fraud-data)
- **Description:** Indian UPI transactions ka dataset jisme fraud aur legitimate dono transactions hain
- **Fields:** transactionId, customerId, amount, timestamp, merchant, device, location, isFraud, etc.
- **Use:** Indian transaction patterns ke liye — UPI-based fraud detection

### 2. Microsoft R-Server Fraud Detection Dataset
- **Source:** Microsoft ML Server / Kaggle
- **Link:** [https://www.kaggle.com/datasets/llabhishekll/microsoft-fraud-detection-dataset](https://www.kaggle.com/datasets/llabhishekll/microsoft-fraud-detection-dataset)
- **Description:** International e-commerce transactions with rich features like device type, IP address, browser, CVV result, proxy detection, shipping/billing country, account age, etc.
- **Fields:** transactionId, accountID, amount, transactionDateTime, localHour, deviceType, ipState, ipCountry, isProxyIP, browserType, paymentType, cardType, cvvResult, shippingCountry, billingCountry, accountAge, isRegistered, isFraud, etc.
- **Use:** Rich feature-set ke saath international fraud patterns — device fingerprinting, geo-velocity, country mismatch detection

### Curated Showcase Transactions
Backend mein 5 handpicked transactions hain (mix of Indian + Microsoft dataset) jo frontend mein dropdown mein dikhte hain. Inme dono FRAUD aur LEGITIMATE transactions hain taaki demo mein variety dikhe.

---

## Project Structure

```
FinShield/
├── backend/
│   ├── agents/
│   │   ├── behavioral_agent.py    # Z-score + Sigmoid + CVV/AcctAge analysis
│   │   ├── temporal_agent.py      # Circular distance + Exponential decay
│   │   ├── geo_agent.py           # Haversine + Velocity + Country mismatch
│   │   ├── device_agent.py        # Shannon entropy + Device fingerprinting
│   │   └── decision_agent_llm.py  # LLM Orchestrator (final verdict)
│   ├── tools/
│   │   ├── geo_tool.py            # Haversine, velocity, country mismatch tools
│   │   └── device_tool.py         # Shannon entropy, frequency, proxy tools
│   ├── data/
│   │   ├── curated_transactions.csv
│   │   ├── indian_upi_fraud.csv
│   │   └── microsoft_transactions.csv
│   ├── app.py                     # FastAPI server + static file serving
│   ├── fraud_graph.py             # Pipeline orchestration + CSV loading
│   ├── requirements.txt
│   └── state.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── TransactionPicker.jsx
│   │   │   ├── TransactionCard.jsx
│   │   │   ├── ReasoningCard.jsx
│   │   │   ├── Controls.jsx
│   │   │   ├── Hero.jsx
│   │   │   ├── Navbar.jsx
│   │   │   └── BackgroundEffects.jsx
│   │   └── lib/
│   │       └── api.js
│   ├── package.json
│   └── vite.config.js
├── Dockerfile
├── render.yaml
├── build.sh
└── README.md
```

---

## Local Setup

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`.env` file banao `backend/` mein:
```
GROQ_API_KEY=your_groq_api_key_here
```

Backend start karo:
```bash
uvicorn app:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend `http://localhost:5173` pe chalega aur `/api/*` requests backend ko proxy karega.

### Production (Single Port)
```bash
cd frontend && npm run build && cd ..
cd backend && uvicorn app:app --host 0.0.0.0 --port 8000
```
Ye frontend build ko backend se serve karega — sab kuch ek port pe.

---

## Docker

```bash
docker build -t finshield .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key finshield
```

---

## Render Deployment

1. GitHub repo connect karo
2. Runtime: **Docker**
3. Environment variable set karo: `GROQ_API_KEY`
4. Deploy

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/transaction` | Frontend se transaction analyze karo (pipeline + fallback) |
| POST | `/fraud/check` | Direct pipeline call with full schema |
| GET | `/api/transactions` | Showcase transactions list for picker |
| GET | `/health` | Health check |

---

## Summary Table — Formulas at a Glance

| Agent | Formula | Kya Measure Karta Hai |
|-------|---------|----------------------|
| Behavioral | `Z = (x - μ) / σ` | Transaction amount kitna unusual hai |
| Behavioral | `risk = σ(\|Z\| - 1.5)` | Z-score se smooth risk score |
| Behavioral | `risk = σ(-(age-30)/20)` | Account age se risk |
| Temporal | `d = min(\|h₁-h₂\|, 24-\|h₁-h₂\|)` | Circular hour deviation |
| Temporal | `risk = 1 - e^(-0.3·d)` | Time deviation se exponential risk |
| Temporal | `R = 0.6·decay + 0.4·(1-freq)` | Composite temporal risk |
| Geo | `Haversine(lat₁,lon₁,lat₂,lon₂)` | Do points ke beech Earth pe distance |
| Geo | `risk = σ((d-100)/60)` | Distance se smooth risk |
| Geo | `velocity = distance / Δt` | Impossible travel detection |
| Geo | `R = 0.35·dist + 0.30·vel + 0.35·mismatch` | Composite geo risk |
| Device | `H = -Σ pᵢ·log₂(pᵢ)` | Device diversity (Shannon entropy) |
| Device | `f = n_device / N_total` | Device frequency ratio |
| Device | `R = Σ(wᵢ·signalᵢ)` | Weighted composite device risk |
| Final | `R = Σ(wᵢ·rᵢ) / Σ(wᵢ)` | Weighted ensemble of all agents |

---

Proudly developed by **Group 9**
