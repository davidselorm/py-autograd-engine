# PyAutograd 🧠
Reverse-mode automatic differentiation engine in pure Python.

```python
from autograd import Value
x = Value(2.0)
y = x * 3 + 1
y.backward()
print(x.grad) # 3.0
```
