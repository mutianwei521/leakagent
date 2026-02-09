# LeakAgent: A Multi-Agent System for Leak Detection and Localization using Multi-modal Large Language Model Coordination in Water Distribution Network

Tianwei Mu, Guangzhou Institute of Industrial Intelligence

## 📋 System Overview

This repository contains a minimal web-based chat interface for interacting with the LeakAgent system.

**Paper**
- Waiting for publishing

**Link to original INP files**
- Net1: [Google Drive](https://drive.google.com/file/d/1oZm9E2qCikHOpMeODedFsH2RQZxJ4bw8/view?usp=drive_link)
- KY3: [Google Drive](https://drive.google.com/file/d/10kc7K7lo8v8gk9d-AlB32B9rIbox5bKC/view?usp=drive_link)
- KY5: [Google Drive](https://drive.google.com/file/d/17_DD-6aCh1UazxIIz9ifML3Dzi4z5y6p/view?usp=drive_link)

### 🏗️ Network Architecture
![Overall Network Architecture](paper/scheme.png)
*Figure: The scheme of LeakDetection Agent: (a) Total work flow for training; (b) Adaptor layer; (c) LTGFM layers; (d) Inference of LeakDetection Agent*

---

### 📊 Interface Show
![Result Show](paper/interface.png)
*Figure: The software interface of the LeakAgent.*

---
### 📊 LeakAgent Demo
[![Watch Demo](paper/td.png)](https://www.youtube.com/watch?v=GbUHKllEPh0)
*Click the image to play demo.mp4.*

---

## 📜 Quick Start
```bash
./start.sh
```

## 📜 File Description
- `web_chat_app.py` - Main application
- `agents/` - Agent modules
- `templates/` - HTML templates
- `requirements.txt` - Python dependencies


## 📜 Access
After startup access: http://localhost:5000

---

## 📜 Acknowledgment

The authors express deep gratitude to Dr. Enyang Gao, General Manager of Deep-Sea 3D Vision Co., Ltd., whose invaluable guidance and mentorship made this research possible. The name “2-GAO” serves as a tribute to Dr. Gao’s contributions to the field of computer vision and his continued support of our research endeavors.

<img src="paper/7d36d1c9cd551d9554675c3f3528f0c6.jpg" alt="GYN" width="25%"/>
Figure: Dr.Gao's portrait.

---

## 📜 Attention
The core implementation of LeakAgent has been uploaded to a GitHub repository. The repository currently contains the main components of the framework, while the LTGFM module cannot be made publicly available at this stage.

---

## 📜 License

MIT License

---
