# Algorithm Selection Guide

This project supports two evolutionary algorithms:

## 1. SeEvo (Self-Evolution) - Main Algorithm ⭐
**File**: `seevo.py`  
**Class**: `SeEvo`

### Description
SeEvo is the main algorithm that combines:
- **Population inter-evolution**: Crossover with reflection between different individuals
- **Individual self-evolution**: Self-improvement of each individual
- **Long-term reflection**: Accumulates insights across generations
- **Mutation**: Elitist-based mutation with external knowledge

### When to use
- **Primary research**: This is your main contribution
- **Best performance**: Expected to perform better than baseline methods
- **Full feature set**: Uses all evolutionary operators

---

## 2. ReEvo (Reflective Evolution) - Baseline ⚖️
**File**: `reevo.py`  
**Class**: `ReEvo`

### Description
ReEvo is the baseline comparison method with:
- **Population inter-evolution**: Crossover with reflection
- **Individual self-evolution**: Self-improvement mechanism
- **Long-term reflection**: Learning across generations
- **Mutation**: Standard elitist mutation

### When to use
- **Baseline comparison**: Compare SeEvo performance against this method
- **Ablation studies**: Test which components contribute to performance
- **Reproducibility**: Verify improvements over existing methods

---

## How to Switch Between Algorithms

### Method 1: Edit Configuration File (Recommended)
Edit `cfg/config.yaml`:

```yaml
# For SeEvo (main algorithm)
algorithm: seevo

# For ReEvo (baseline)
algorithm: reevo
```

### Method 2: Command Line Override
```bash
# Run with SeEvo
python main.py algorithm=seevo

# Run with ReEvo
python main.py algorithm=reevo
```

### Method 3: Multiple Runs for Comparison
```bash
# Run SeEvo
python main.py algorithm=seevo case_num=0

# Run ReEvo on same case for comparison
python main.py algorithm=reevo case_num=0
```

---

## Configuration Parameters

Both algorithms share the same configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm` | `seevo` | Algorithm choice: `seevo` or `reevo` |
| `model` | `gpt-4.1-mini-2025-04-14` | LLM model to use |
| `temperature` | `1.0` | LLM sampling temperature |
| `max_fe` | `6` | Maximum function evaluations |
| `pop_size` | `20` | Population size |
| `init_pop_size` | `20` | Initial population size |
| `mutation_rate` | `0.5` | Mutation rate (0-1) |
| `timeout` | `100` | Evaluation timeout (seconds) |
| `case_num` | `0` | Test case number |

---

## Example Usage

### Basic Run with SeEvo
```bash
python main.py
```

### Run with Different Models
```bash
# SeEvo with GPT-4
python main.py algorithm=seevo model=gpt-4-turbo-preview

# ReEvo with GLM-4
python main.py algorithm=reevo model=GLM-4
```

### Batch Comparison
```bash
# Run both algorithms on multiple cases
for case in 0 1 2 3 4; do
    echo "Running SeEvo on case $case"
    python main.py algorithm=seevo case_num=$case
    
    echo "Running ReEvo on case $case"
    python main.py algorithm=reevo case_num=$case
done
```

---

## Output Files

Both algorithms produce similar outputs in `outputs/` directory:
- `problem_iter{N}_code{M}.py` - Generated code for each iteration
- `problem_iter{N}_response{M}.txt` - LLM responses
- `problem_iter{N}_stdout{M}.txt` - Execution logs
- `problem_iter{N}_short_term_reflections.txt` - Short-term insights
- `problem_iter{N}_long_term_reflection.txt` - Long-term insights
- `best_code_overall_val_stdout.txt` - Final validation results

---

## Performance Comparison Tips

1. **Use same random seed** for fair comparison
2. **Run multiple trials** (at least 5-10) to account for LLM randomness
3. **Compare on same test cases** to ensure consistency
4. **Track both objective values and execution time**
5. **Analyze reflection quality** to understand algorithmic insights

---

## Troubleshooting

**Error: Algorithm 'xxx' is not implemented**
- Check spelling in config file
- Only `seevo` and `reevo` are supported

**Different results each run**
- This is expected due to LLM stochasticity
- Use lower `temperature` for more deterministic results
- Run multiple trials and report statistics

---

## Citation

If you use this code, please cite both algorithms appropriately:
- **SeEvo**: Your main contribution (add your paper citation)
- **ReEvo**: Baseline method (add ReEvo paper citation if applicable)
