# GPU-Based Decision Tree Classifier

## Features
✅ Implements a **Decision Tree Classifier** using PyTorch and OpenCL.
✅ Supports **parallel processing** for faster decision tree computations.
✅ Uses **MNIST dataset** for handwritten digit classification.
✅ Performance measurement tools for **execution time and resource utilization**.
✅ Optimized for **AMD Radeon GPUs** but adaptable to other OpenCL-supported hardware.

## Installation Prerequisites
The following packages should be installed:
- **Python 3.10+**
- **PyTorch (2.5.1)**
- **NumPy (2.2.2)**
- **Pandas (2.2.3)**
- **scikit-learn (1.6.1)**
- **OpenCL (2025.1)**
- **TensorFlow (2.18.0)** *(optional, used for dataset loading)*

### Run the Classifier
python main.py

### Performance Testing
The script will output execution times and accuracy metrics:
```bash
Non-parallelized execution time: 210.99 seconds
Parallelized execution time: 20.45 seconds
Accuracy: 0.6550
```

## File Structure
```
├── DecisionTree
│   ├── DecisionTreeClassifierGPU.py
│   ├── node.py
├── main.py
├── README.md
```

## Benchmarks
| Metric | CPU (AMD Ryzen 5) | GPU (AMD Radeon) |
|--------|------------------|------------------|
| Execution Time (Optimized) | ~62 sec | ~20 sec |
| Speedup Ratio (CPU/GPU) | 3.1x | - |

## Future Improvements
- Implement **CUDA support** for NVIDIA GPUs.

## Contributors
- **Klejda Rrapaj** - k.rrapaj@student.unisi.it
- **Sildi Riçku** - s.ricku@student.unisi.it

