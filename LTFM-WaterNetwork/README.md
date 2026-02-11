# LTFM: Look-Twice Feature Matching for Water Distribution Network Anomaly Detection

A graph neural network-based anomaly detection and localization system for water distribution networks. This system simultaneously performs global anomaly detection and precise region localization, providing an advanced solution for intelligent water network monitoring.

## 🎯 Features

- **Dual Detection**: Simultaneous global anomaly detection and regional anomaly localization
- **Physics-Driven**: Feature engineering based on pressure sensitivity analysis
- **Graph Neural Network**: Leverages Graph2Vec and attention mechanisms for network topology processing
- **High Accuracy**: Achieves 100% detection and localization accuracy on NET-1 benchmark network
- **Real-Time**: Supports real-time monitoring and rapid response

## 📊 Performance

| Metric | LTFM |
|--------|------|
| Global Detection Accuracy | **100%** |
| Regional Localization Accuracy | **100%** |
| F1 Score | **1.0000** |
| AUC | **1.0000** |
| Training Convergence | **3 epochs** |

## 🚀 Quick Start

### Requirements
- Python 3.8+
- PyTorch 2.0+
- EPANET 2.2

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LTFM-WaterNetwork.git
cd LTFM-WaterNetwork

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default configuration
python main.py --mode train

# Train with custom parameters
python main.py --mode train --n-scenarios 1000
```

### Inference

```bash
# Real-time monitoring mode
python main.py --mode inference

# Batch prediction
python main.py --mode inference --test-data data/test.csv
```

## 📁 Project Structure

```
LTFM-WaterNetwork/
├── README.md              # Project documentation
├── LICENSE                # MIT License
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── config.yaml           # Configuration file
├── main.py               # Main program entry
├── src/                  # Core source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── epanet_handler.py       # EPANET interface
│   │   └── sensitivity_analyzer.py # Pressure sensitivity analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── graph2vec_encoder.py    # Graph embedding encoder
│   │   └── ltfm_model.py          # LTFM main model
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # Training logic
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py           # Inference predictor
│   └── utils/
│       ├── __init__.py
│       ├── fcm_partitioner.py     # FCM network partitioning
│       └── graph_features.py      # Graph feature extraction
└── data/
    └── Net1.inp                   # Example EPANET network file
```

## 🔧 Configuration

Main configuration options in `config.yaml`:

```yaml
# Network configuration
data:
  epanet_file: "data/Net1.inp"

# FCM partitioning
fcm:
  n_clusters: 3

# Training
training:
  batch_size: 1
  learning_rate: 0.0005
  epochs: 20
```

## 📈 Technical Architecture

### 1. Pressure Sensitivity Analysis
- Hydraulic computation based on EPANET
- Calculates pressure propagation relationships between nodes
- Builds sensitivity matrix as feature foundation

### 2. Network Partitioning
- Uses FCM fuzzy clustering algorithm
- Partitions based on pressure sensitivity
- Generates hydraulically-related regional divisions

### 3. Graph Embedding Learning
- Graph2Vec algorithm for graph representation
- Combines topological features with statistical features
- Enhanced feature vector representation

### 4. LTFM Model
- Dual-branch architecture: Global + Regional
- Multi-head self-attention mechanism
- Cross-modal attention fusion
- Composite loss function optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*LTFM - Making water network monitoring smarter and more precise!* 🌊✨
