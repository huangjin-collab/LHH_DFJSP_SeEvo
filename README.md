# SeEvo for Dynamic Flexible Job Shop Scheduling Problem (DFJSSP)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An LLM-driven evolutionary algorithm framework for automatically discovering effective heuristics for Dynamic Flexible Job Shop Scheduling Problems.

## 📑 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Example Workflow](#-example-workflow)
- [Configuration](#-configuration)
- [Dataset Information](#-dataset-information)
- [Project Structure](#-project-structure)
- [Algorithm Comparison](#-algorithm-comparison)
- [Output and Results](#-output-and-results)
- [Troubleshooting](#-troubleshooting)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 📋 Overview

**SeEvo** (Self-Evolution) is an advanced evolutionary algorithm that leverages Large Language Models (LLMs) to automatically generate and evolve scheduling heuristics for DFJSSP. The framework includes:

- **SeEvo (Main)**: Full algorithm with individual self-evolution mechanism
- **ReEvo (Baseline)**: Simplified version without individual self-evolution for comparison

### Key Features

✨ **LLM-Driven Evolution**: Automatically generates and refines heuristics using GPT/Qwen/GLM models  
✨ **Intelligent Reflection**: Short-term and long-term reflection mechanisms with performance gap analysis  
✨ **EMA Adaptation**: Exponential Moving Average tracking for uncertainty-aware scheduling  
✨ **Train/Test Modes**: Separate modes for training new heuristics and testing pre-trained rules  
✨ **Flexible Architecture**: Easy to extend with different LLM providers and problem variants  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see Installation)
- API key for your chosen LLM provider

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd LHH_DFJSP_SeEvo
```

2. **Install dependencies**

⚠️ **Important**: Different LLM providers require different packages:

**For OpenAI (GPT models)**:
```bash
pip install openai hydra-core numpy pandas
```

**For Alibaba Cloud (Qwen models)**:
```bash
pip install dashscope hydra-core numpy pandas
```

**For Zhipu AI (GLM models)**:
```bash
pip install zhipuai hydra-core numpy pandas
```

**For Moonshot AI**:
```bash
pip install openai hydra-core numpy pandas  # Uses OpenAI-compatible API
```

3. **Configure API Key**

Option 1: Environment variable (recommended)
```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Qwen
export DASHSCOPE_API_KEY="your-api-key-here"

# For GLM
export ZHIPUAI_API_KEY="your-api-key-here"
```

Option 2: Configuration file
```yaml
# cfg/config.yaml
api_key: "your-api-key-here"  # Add this line
```

---

## 📖 Usage

### Training Mode

Train new heuristics from scratch:

```bash
# Edit cfg/config.yaml
mode: train
algorithm: seevo  # or reevo
model: qwen-plus  # or gpt-4, GLM-4, etc.
max_fe: 20

# Run training
python main.py
```

Training will:
1. Start from a seed heuristic function
2. Generate initial population using LLM
3. Evolve through multiple iterations with reflection and mutation
4. Save best heuristics to `outputs/train/main/`

### Testing Mode

⚠️ **Data Extraction Required**: Before testing, you need to manually extract trained heuristic rules:

1. **Prepare the `data/` directory**:
```bash
mkdir -p data
```

2. **Extract trained rules**: Copy your best heuristic files from training runs into `data/` with the following structure:
```
data/
├── 2025-05-04_15-02-19/
│   └── problem_iter18_response7.txt
├── 2025-05-04_15-10-13/
│   └── problem_iter15_response3.txt
├── ...
└── 2025-05-04_17-28-29/
    └── problem_iter12_response11.txt
```

3. **Run testing**:
```bash
# Edit cfg/config.yaml
mode: test
data_dir: data

# Run test
python main.py
```

Testing will:
1. Load all trained rules from `data/` directory
2. Run one evolution iteration on the test dataset
3. Save results to `outputs/test/main/`

---

## 💡 Example Workflow

Here's a complete example from training to testing:

### Step 1: Train Heuristics

```bash
# 1. Configure for training
vim cfg/config.yaml
# Set: mode: train, algorithm: seevo, model: qwen-plus, max_fe: 20

# 2. Set API key
export DASHSCOPE_API_KEY="your-qwen-api-key"

# 3. Run training
python main.py

# Training output will be saved to outputs/train/main/2025-12-30_10-30-45/
```

### Step 2: Extract Best Heuristics

```bash
# Create data directory
mkdir -p data

# Copy best heuristics from training runs
# Look for files like problem_iter15_response3.txt with good performance
cp outputs/train/main/2025-12-30_10-30-45/problem_iter15_response3.txt data/run1/
cp outputs/train/main/2025-12-30_11-15-20/problem_iter18_response7.txt data/run2/
# ... repeat for ~20 best heuristics from different runs
```

### Step 3: Test Heuristics

```bash
# 1. Configure for testing
vim cfg/config.yaml
# Set: mode: test, data_dir: data

# 2. Run testing
python main.py

# Test results will be saved to outputs/test/main/2025-12-30_14-20-30/
```

### Step 4: Analyze Results

```bash
# Check training logs
cat outputs/train/main/2025-12-30_10-30-45/main.log | grep "Best obj"

# Check test results
cat outputs/test/main/2025-12-30_14-20-30/main.log | grep "Average makespan"
```

---

## ⚙️ Configuration

### Main Configuration (`cfg/config.yaml`)

```yaml
# Algorithm selection
algorithm: seevo  # Options: seevo, reevo

# Execution mode
mode: train  # Options: train, test
data_dir: data  # Directory for trained rules (test mode only)

# LLM parameters
model: qwen-plus  # Options: gpt-3.5-turbo, gpt-4, GLM-4, qwen-plus, moonshot-v1-8k
temperature: 1

# GA parameters
max_fe: 20  # Maximum function evaluations
pop_size: 20  # Population size
init_pop_size: 20  # Initial population size
mutation_rate: 0.5
timeout: 100  # Evaluation timeout (seconds)
```

### Supported LLM Models

| Provider | Model Name | Config Value |
|----------|-----------|-------------|
| OpenAI | GPT-3.5 Turbo | `gpt-3.5-turbo-0125` |
| OpenAI | GPT-4 Turbo | `gpt-4-turbo-preview` |
| Alibaba | Qwen Plus | `qwen-plus` |
| Alibaba | Qwen Turbo | `qwen-turbo` |
| Zhipu AI | GLM-3 Turbo | `GLM-3-Turbo` |
| Zhipu AI | GLM-4 | `GLM-4` |
| Moonshot | Moonshot v1 8K | `moonshot-v1-8k` |

---

## 📊 Dataset Information

The framework uses different datasets for training and testing:

### Training Dataset
```
problems/jsp_constructive/test_data/
├── jsp_cases/          # Training plan data
└── real_jsp_cases/     # Training actual data
```

### Testing Dataset
```
problems/jsp_constructive/test_data/
├── test_jsp/           # Test plan data
└── real_test_jsp/      # Test actual data
```

**Note**: Test dataset files do NOT have the `_real` suffix (e.g., `prob_00.pkl` in both directories).

---

## ⚡ Performance and Requirements

### System Requirements

- **CPU**: Multi-core processor recommended (parallel evaluation)
- **RAM**: Minimum 8GB, 16GB recommended for larger populations
- **Storage**: ~1-2GB for code, datasets, and outputs
- **Network**: Stable internet connection for LLM API calls

### Performance Considerations

**Training Time**:
- Single iteration: ~2-5 minutes (depending on LLM response time)
- Full training (20 iterations): ~1-2 hours
- Influenced by: LLM API latency, population size, dataset complexity

**API Costs**:
- GPT-3.5-turbo: ~$0.50-1.00 per full training run
- GPT-4: ~$5.00-10.00 per full training run  
- Qwen/GLM: Varies by provider pricing

**Tips for Efficiency**:
- Use smaller `pop_size` for initial experiments (e.g., 10 instead of 20)
- Reduce `max_fe` for quick prototyping
- Cache successful heuristics to avoid regeneration
- Use faster models (GPT-3.5, Qwen-turbo) for development

---

## 💡 Best Practices

### For Training

1. **Start Small**: Begin with smaller populations and fewer iterations to validate setup
2. **Monitor Progress**: Check `main.log` regularly for convergence patterns
3. **Save Checkpoints**: Keep backups of promising heuristics from intermediate iterations
4. **Diverse Seeds**: Try multiple training runs with different random seeds
5. **Temperature Tuning**: Adjust `temperature` (0.7-1.5) to balance exploration vs exploitation

### For Testing

1. **Quality over Quantity**: Select diverse, high-performing heuristics rather than random sampling
2. **Consistent Evaluation**: Always test on the same dataset for fair comparison
3. **Multiple Runs**: Run testing multiple times to account for stochasticity
4. **Baseline Comparison**: Compare against ReEvo and traditional heuristics

### For Different LLM Providers

**OpenAI (GPT)**:
- Best for: High-quality heuristics, complex reasoning
- Tip: Use GPT-3.5 for development, GPT-4 for final runs

**Qwen**:
- Best for: Cost-effective Chinese/English bilingual scenarios
- Tip: `qwen-plus` offers good balance of cost and performance

**GLM**:
- Best for: Specialized domain knowledge in Chinese contexts
- Tip: GLM-4 provides stronger reasoning capabilities

---

## 🔍 Project Structure

```
.
├── cfg/                    # Configuration files
│   ├── config.yaml        # Main configuration
│   └── problem/           # Problem-specific configs
├── data/                   # Trained heuristics (manual extraction)
├── problems/              # Problem implementations
│   └── jsp_constructive/
│       ├── eval.py        # Evaluation script
│       ├── Real_Time_FJSP.py  # DFJSSP environment
│       └── test_data/     # Dataset files
├── prompts/               # LLM prompt templates
│   ├── common/            # Common prompts
│   └── jsp_constructive/  # Problem-specific prompts
├── utils/                 # Utility functions
├── seevo.py              # SeEvo algorithm implementation
├── reevo.py              # ReEvo baseline implementation
├── main.py               # Entry point
└── README.md             # This file
```

---

## 🆚 Algorithm Comparison

### SeEvo vs ReEvo

| Feature | SeEvo | ReEvo |
|---------|-------|-------|
| Population Inter-Evolution | ✅ | ✅ |
| Individual Self-Evolution | ✅ | ❌ |
| Long-term Reflection | ✅ | ✅ |
| Mutation | ✅ | ✅ |
| Performance Gap Analysis | ✅ | ✅ |
| EMA Tracking | ✅ | ✅ |

**When to use SeEvo**: When you need more thorough exploration and refinement of individual heuristics.

**When to use ReEvo**: When you want faster iterations or as a baseline for comparison.

---

## 📈 Output and Results

### Training Output

After training, you'll find results in `outputs/train/main/<timestamp>/`:

- `problem_iter{N}_code{M}.py`: Generated heuristic code files
- `problem_iter{N}_response{M}.txt`: LLM response texts
- `problem_iter{N}_stdout{M}.txt`: Evaluation logs
- `main.log`: Complete execution log

### Testing Output

Test results are saved in `outputs/test/main/<timestamp>/`:

- Best heuristic performance on test dataset
- Evaluation metrics and logs

---

## � Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| Train with SeEvo | Set `algorithm: seevo`, `mode: train` in config, run `python main.py` |
| Train with ReEvo | Set `algorithm: reevo`, `mode: train` in config, run `python main.py` |
| Test heuristics | Set `mode: test`, prepare `data/` directory, run `python main.py` |
| Change LLM model | Edit `model:` in `cfg/config.yaml` |
| Set API key | `export OPENAI_API_KEY="key"` or add to `cfg/config.yaml` |
| Check logs | `cat outputs/train/main/<timestamp>/main.log` |
| View best code | `cat outputs/train/main/<timestamp>/problem_iter{N}_code{M}.py` |

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm` | `seevo` | Algorithm choice: `seevo` or `reevo` |
| `mode` | `train` | Execution mode: `train` or `test` |
| `model` | `qwen-plus` | LLM model name |
| `temperature` | `1` | LLM sampling temperature (0-2) |
| `max_fe` | `20` | Maximum function evaluations |
| `pop_size` | `20` | Population size |
| `mutation_rate` | `0.5` | Mutation probability (0-1) |
| `timeout` | `100` | Evaluation timeout in seconds |

---

## �🔧 Troubleshooting

### Common Issues

**1. All individuals have `inf` objective value**
- Check if dataset files exist in the correct path
- Verify test dataset filenames (no `_real` suffix for test data)
- Check evaluation logs in `problem_iter0_stdout0.txt`

**2. API Key errors**
- Ensure correct environment variable is set for your LLM provider
- Verify API key has sufficient quota
- Check `utils/utils.py` for supported providers

**3. Import errors**
- Install provider-specific packages (see Installation section)
- Different LLM providers require different libraries

**4. Unicode errors on Windows**
- Fixed in current version with UTF-8 encoding
- If still occurs, check terminal encoding settings

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{huang2025seevo,
  title={Automatic programming via large language models with population self-evolution for dynamic fuzzy job shop scheduling problem},
  author={Jin Huang, Qihao Liu, Xinyu Li, Liang Gao, Yue Teng},
  journal={IEEE TRANSACTIONS ON FUZZY SYSTEMS},
  year={2025}
}
```

---

## 🔮 Future Work

We are actively working on improving the framework with the following enhancements:

### 🎯 Generalization Enhancement (Primary Focus)

One of the key challenges in LLM-driven heuristic discovery is ensuring that evolved heuristics can generalize well across different problem instances and scales. We are developing several approaches to address this:

#### 🚧 Cross-Scale Generalization
- Developing techniques to ensure heuristics trained on small-scale problems can effectively handle larger instances
- Implementing multi-scale training strategies that expose the LLM to diverse problem sizes during evolution
- Creating evaluation metrics that specifically measure generalization capability

#### 🚧 Domain Adaptation
- Enabling transfer of learned heuristics from one scheduling domain to related domains
- Building mechanisms to identify transferable knowledge components in evolved heuristics
- Reducing the training cost when adapting to new problem variants

#### 🚧 Meta-Learning for Fast Adaptation
- Integrating meta-learning frameworks to enable quick adaptation to new problem instances
- Building a knowledge base of effective heuristic patterns that can be reused across problems
- Implementing few-shot learning capabilities for rapid fine-tuning on new scenarios

#### 🚧 Generalization-Aware Training
- Designing new fitness evaluation metrics that balance performance and generalization
- Implementing regularization techniques in the evolutionary process to prevent overfitting
- Creating diverse training benchmarks that better represent real-world scheduling complexity

### 💬 Community Input Welcome!

We encourage the community to share insights and suggestions on improving generalization. If you have ideas or encounter generalization issues in your experiments, please open an issue or discussion on GitHub!

### Stay tuned for updates! 🚀

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd LHH_DFJSP_SeEvo

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

## 🙏 Acknowledgments

- Thanks to all contributors and researchers in the field of LLM-driven optimization
- Special thanks to the open-source community for various tools and libraries used in this project

---

**Happy Scheduling! 🎉**
