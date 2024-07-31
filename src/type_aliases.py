from typing import Literal

TransformLiteral = Literal["Linear", "ClippedLinear", "SoftPlus", "NegExp", "Sigmoid"]
CombinerLiteral = Literal[
    "Mean", "Multiply", "GeometricMean", "Min", "Max", "LNorm", "SmoothMax"
]
