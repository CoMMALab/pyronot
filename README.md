# `PyRoFFI`: A Python Library for Robot Kinematics Using Spherical Approximations


[![Format Check](https://github.com/CoMMALab/pyroffi/actions/workflows/formatting.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/formatting.yml)
[![Pyright](https://github.com/CoMMALab/pyroffi/actions/workflows/pyright.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/pyright.yml)
[![Pytest](https://github.com/CoMMALab/pyroffi/actions/workflows/pytest.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/pytest.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/pyroffi)](https://pypi.org/project/pyroffi/)

By Weihang Guo, Sai Coumar

PyRoFFI is a toolkit optimized with Jax JIT tracing for accelerated kinematics research. This repository is expands on the work [pyroki](https://github.com/chungmin99/pyroki). 

Additional Features:
- Spherized Batched Robot Collision Checking
- SRDF parsing
- Runtime Neural SDFs
- Expanded Primitive Sets
- Improved Jax performance


## Installation
```
pip install git+https://github.com/brentyi/jaxls.git
pip install pyroffi
```
