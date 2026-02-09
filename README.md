
<div align="left">

# LeakAgent: Multi-Agent Coordination for Water Distribution Networks

**An Intelligent System for Leak Detection, Localization, and Sensor Placement via Multi-Modal LLMs**

[Tianwei Mu](https://github.com/mutianwei521)</sup>

Guangzhou Institute of Industrial Intelligence

[![Paper](https://img.shields.io/badge/Paper-Waiting_for_Publishing-yellow)](https://drive.google.com/file/d/1oZm9E2qCikHOpMeODedFsH2RQZxJ4bw8/view?usp=drive_link)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🔥 News
- **[2026-02-09]** LeakAgent code is released! Now supports advanced sensor placement and boundary optimization.
- **[2026-01-20]** Added support for spectral clustering-based network partitioning.

## 📋 Introduction

**LeakAgent** is a comprehensive multi-agent system designed to revolutionize water distribution network (WDN) management. By leveraging the power of Multi-modal Large Language Models (LLMs), it acts as an intelligent, interactive assistant for hydraulic engineers and researchers.

The system orchestrates a team of specialized agents to execute complex, multi-step tasks across the WDN lifecycle:

- 💧 **Hydraulic Analysis**: Precision simulation of pressure, flow, and head using EPANET.
- 🧩 **Network Partitioning**: Intelligent division of large networks into manageable optimization zones.
- 📡 **Sensor Placement**: Strategic optimization of sensor locations for maximum coverage and resilience.
- 🔍 **Leak Detection & Localization**: Advanced machine learning models to identify and pinpoint leaks in real-time.
- 📊 **Interactive Visualization**: Intuitive web-based interface for topology analysis and result interpretation.

## 🏗️ Architecture

![Overall Network Architecture](paper/scheme.png)
*Figure 1: The architecture of LeakDetection Agent: (a) Total workflow for training; (b) Adaptor layer; (c) LTGFM layers; (d) Inference of LeakDetection Agent.*

## � Quick Start

### Prerequisites
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8–3.11 (Recommended: 3.10)
- **Hardware**: NVIDIA GPU (8GB+ VRAM recommended for training), 16GB+ RAM

### Installation & Run

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/leakagent.git
    cd leakagent
    ```

2.  **Start the system**
    ```bash
    ./start.sh
    ```
    Or manually:
    ```bash
    pip install -r requirements.txt
    python web_chat_app.py
    ```

3.  **Access the interface**
    Open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

## 📊 Interface & Demo

<div align="center">
  <img src="paper/interface.png" width="100%" alt="Interface Preview"/>
</div>
<p align="center"><em>Figure 2: The interactive web interface of LeakAgent.</em></p>

### 🎥 Demo Video
[![Watch Demo](paper/td.png)](https://www.youtube.com/watch?v=GbUHKllEPh0)
*Click the image above to watch the demonstration video.*

## 💡 Recommended Prompts

LeakAgent understands natural language. Here are effective prompts to get you started:

### 1. 🧩 Network Partitioning
> "Please partition the network into 5 zones."
> "Divide the network using spectral clustering algorithm."
> "Perform graph partitioning analysis on this network."

### 2. ⚙️ Optimization
> "Optimize the network partition to minimize cut edges."
> "Optimize sensor placement for maximum detection coverage."
> "Find the optimal configuration for pressure monitoring."

### 3. 📡 Sensor Placement
> "Where should I place 3 pressure sensors? "
> "Recommend the best locations for monitoring sensors."
> "Analyze sensor placement resilience."

### 4. 🔍 Leakage Model
> "Train a leak detection model with 100 epochs."
> "Build a leak identification model using 50 scenarios."
> "Detect leaks based on the current sensor data."

### 5. ℹ️ Network Information
> "Show me the network summary."
> "What is the total pipe length?"
> "Analyze the network structure."
> "How many junctions and reservoirs are there?"

## 📂 Resources

**Original INP Files**:
- [Net1 (Google Drive)](https://drive.google.com/file/d/1oZm9E2qCikHOpMeODedFsH2RQZxJ4bw8/view?usp=drive_link)
- [KY3 (Google Drive)](https://drive.google.com/file/d/10kc7K7lo8v8gk9d-AlB32B9rIbox5bKC/view?usp=drive_link)
- [KY5 (Google Drive)](https://drive.google.com/file/d/17_DD-6aCh1UazxIIz9ifML3Dzi4z5y6p/view?usp=drive_link)

## 📜 Acknowledgments

The authors express deep gratitude to **Dr. Enyang Gao**, General Manager of Deep-Sea 3D Vision Co., Ltd., whose invaluable guidance and mentorship made this research possible. The project serves as a tribute to Dr. Gao’s contributions to the field of computer science and his continued support of our research endeavors.

Benchmarks are derived from the [OpenWaterAnalytics EPANET-Matlab-Toolkit](https://github.com/OpenWaterAnalytics/EPANET-Matlab-Toolkit).

## ⚠️ Attention

The core implementation of LeakAgent is open-sourced in this repository. Please note that the **LTGFM module** is currently proprietary and not included in this release.

## 📝 Citation

As soon as possible
```

## � License

This project is licensed under the [MIT License](LICENSE).
