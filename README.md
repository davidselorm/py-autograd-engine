# py-autograd-engine

A zero-dependency, lightweight reverse-mode automatic differentiation DAG engine and neural network library implemented in pure Python.

## Features
- **Dynamic Computation Graph**: Builds computation DAG on the fly during forward evaluation.
- **Topological Sorting**: Reverse-mode automatic gradient backpropagation in $O(V + E)$ linear time.
- **Arithmetic Primitives**: Full support for `+`, `-`, `*`, `/`, `**`, `neg`, `exp`, `log`.
- **Activation Functions**: Continuous `tanh`, `sigmoid`, and piecewise `ReLU` with subgradient handling.
- **Neural Network Primitives**: `Module`, `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron).
- **Optimization**: `SGD` with momentum and `mse_loss` loss function.
- **Zero External Dependencies**: Standard library Python only.

## Quickstart
```python
from autograd import Value, MLP, SGD, mse_loss

# Scalar differentiation
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = (a * b).tanh()
c.backward()
print(f"dc/da: {a.grad}")

# Neural Network Training
model = MLP(2, [4, 1])
optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
```

## Tests & Benchmarks
```bash
python -m unittest tests/test_autograd.py
python benchmarks/benchmark_engine.py
```
