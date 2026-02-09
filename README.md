# LeakAgent: A Multi-Agent System for Leak Detection and Localization using Multi-modal Large Language Model Coordination in Water Distribution Network

Tianwei Mu, Guangzhou Institute of Industrial Intelligence

## 📋 System Overview

This repository contains a minimal web-based chat interface for interacting with the LeakAgent system.

### 🏗️ Network Architecture
![Overall Network Architecture](paper/scheme.png)
*Figure: The scheme of LeakDetection Agent: (a) Total work flow for training; (b) Adaptor layer; (c) LTGFM layers; (d) Inference of LeakDetection Agent*

---
### 📊 Interface Show
![Result Show](paper/interface.png)
*Figure: The software interface of the LeakAgent.*

---
### 📊 LeakAgent Demo
<p align="center">
  <a href="https://www.youtube.com/watch?v=GbUHKllEPh0">
    <img src="paper/td.png" alt="Demo Video" width="720">
  </a>
</p>
<p align="center"><em>Click the image to watch the demo video.</em></p>

---
## Quick Start
```bash
./start.sh
```

## File Description
- `web_chat_app.py` - Main application
- `agents/` - Agent modules
- `templates/` - HTML templates
- `requirements.txt` - Python dependencies

## access
After startup access: http://localhost:5000

## attention
The core implementation of LeakAgent has been uploaded to a GitHub repository. The repository currently contains the main components of the framework, while the LTGFM module cannot be made publicly available at this stage.
