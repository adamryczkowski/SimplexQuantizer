# Simplex Quantizer

This project provides a Python implementation for nested quantization of weights (simplex values) using an underlying tree model.

## Underlying Tree Model

The underlying tree model used in this project is the `QuantizationTree` class. This class represents a tree structure where each node can either be a string or another `QuantizationTree`. The tree is used to represent the nested quantization of weights.

## Example Usage

Here is an example of how to use the `simplify_simplex` function, which is the only exported function in this project:

```python
from SimplexQuantizer import simplify_simplex
import numpy as np

# Define a simplex with 8 values including 0 
simplex = np.array([0.05, 0.25, 0.01, 0, 0.015, 0.005, 0.1, 0.55, 0.02]) 

# Define the print function
from fractions import Fraction
def repr_simplex(simplex:np.ndarray)->str:
    values = [str(Fraction(val).limit_denominator(10000)) for val in simplex]
    return "[" +  ", ".join(values) + "]"


# Simplify the simplex
print(f"Simplified simplex with 2-level quantization: {repr_simplex(simplify_simplex(simplex, level_count=2))}")
print(f"Simplified simplex with 3-level quantization: {repr_simplex(simplify_simplex(simplex, level_count=3))}")
print(f"Simplified simplex with 4-level quantization: {repr_simplex(simplify_simplex(simplex, level_count=4))}")
print(f"Simplified simplex with 5-level quantization: {repr_simplex(simplify_simplex(simplex, level_count=5))}")
```

The output of this code is:

```
Simplified simplex with 2-level quantization: [1/16, 1/4, 1/128, 0, 1/64, 1/128, 1/8, 1/2, 1/32]
Simplified simplex with 3-level quantization: [2/81, 2/9, 32/2187, 0, 8/729, 16/2187, 1/27, 2/3, 4/243]
Simplified simplex with 4-level quantization: [1/16, 1/4, 1/128, 0, 1/64, 1/128, 1/8, 1/2, 1/32]
Simplified simplex with 5-level quantization: [1/25, 1/5, 6/625, 0, 1/125, 4/625, 3/25, 3/5, 2/125]
```

## Installation

To install this project, run the following command in the root directory of the project:

```
poetry install
```

