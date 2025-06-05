# 📘 MemTorch API Documentation

**Version**: 1.1.6
**Source**: [MemTorch Documentation](https://memtorch.readthedocs.io/en/latest/)
**Overview**: MemTorch is a simulation framework for memristive deep learning systems, integrating seamlessly with PyTorch. It enables modeling of memristive devices, their non-idealities, and mapping of neural network components to memristive hardware.

---

## 📂 Module Structure

* [`memtorch.bh`](#memtorchbh-behavioral-modeling)
* [`memtorch.map`](#memtorchmap-mapping-and-scaling)
* [`memtorch.mn`](#memtorchmn-memristive-neural-network-modules)
* [📘 Tutorials and Examples](#-tutorials-and-examples)

---

## `memtorch.bh`: Behavioral Modeling

This module provides tools to simulate the behavior of memristive devices and crossbar arrays.

### `memtorch.bh.memristor`

Encapsulates various memristor models:

* **LinearIonDrift**: Models linear ion drift behavior.
* **VTEAM**: Voltage Threshold Adaptive Memristor model.
* **Data\_Driven**: Data-driven model based on empirical data.

**Example**:

```python
from memtorch.bh.memristor import VTEAM

memristor = VTEAM(r_on=100, r_off=16000)
```

### `memtorch.bh.nonideality`

Models non-ideal behaviors such as:

* Finite conductance states
* Device faults
* Non-linear I/V characteristics
* Endurance and retention effects

**Example**:

```python
from memtorch.bh.nonideality import NonIdeality

non_ideal = NonIdeality()
non_ideal.apply_nonidealities(model)
```

### `memtorch.bh.crossbar`

Simulates crossbar architectures.

#### `Crossbar`

Models memristor crossbars and manages modular crossbar tiles.

**Example**:

```python
import torch
from memtorch.bh.crossbar import Crossbar
from memtorch.bh.memristor import VTEAM

crossbar = Crossbar(memristor_model=VTEAM,
                    memristor_model_params={"r_on": 1e2, "r_off": 1e4},
                    shape=(100, 100),
                    tile_shape=(64, 64))
```

#### `Program`

Provides routines to program the conductance of devices within a crossbar.

#### `Tile`

Facilitates the creation of modular crossbar tiles to represent large-scale networks.

---

## `memtorch.map`: Mapping and Scaling

Handles the translation of neural network parameters and inputs to memristive hardware equivalents.

### `memtorch.map.Input`

Encodes input values as bit-line voltages.

**Example**:

```python
from memtorch.map.Input import naive_scale

scaled_input = naive_scale(module, input_tensor)
```

### `memtorch.map.Parameter`

Maps neural network weights to device conductance values.

**Example**:

```python
from memtorch.map.Parameter import naive_map

mapped_params = naive_map(layer)
```

### `memtorch.map.Module`

Determines relationships between readout currents of memristive crossbars and desired outputs.

**Example**:

```python
from memtorch.map.Module import naive_tune

naive_tune(module, input_shape=(1, 28, 28))
```

---

## `memtorch.mn`: Memristive Neural Network Modules

Offers memristive equivalents of PyTorch neural network layers.

### `memtorch.mn.Module`

Includes the `patch_model` function to convert standard PyTorch models into memristive versions.

**Example**:

```python
import copy
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.map.Input import naive_scale
from memtorch.bh.memristor import VTEAM

model = Net()
patched_model = patch_model(copy.deepcopy(model),
                            memristor_model=VTEAM,
                            memristor_model_params={},
                            module_parameters_to_patch=[torch.nn.Linear, torch.nn.Conv2d],
                            mapping_routine=naive_map,
                            scaling_routine=naive_scale)
```

### Layer Implementations

Provides memristive versions of layers:

* `Linear`
* `Conv1d`
* `Conv2d`
* `Conv3d`
* `RNN`

**Example**:

```python
from memtorch.mn import Linear
from memtorch.bh.memristor import VTEAM

memristive_linear = Linear(torch.nn.Linear(10, 5),
                           memristor_model=VTEAM,
                           memristor_model_params={})
```

---

## 📘 Tutorials and Examples

MemTorch offers a suite of interactive tutorials in Jupyter Notebook format to help users get started and explore advanced features:

### Introductory Tutorial

A starting point for new users to understand the basics of MemTorch.

**Link**: [Open in Colab](https://colab.research.google.com/github/coreylammie/MemTorch/blob/master/tutorials/Introductory_Tutorial.ipynb)

### Exemplar Simulations

Demonstrates various simulations as presented in the original MemTorch paper.

**Link**: [Open in Colab](https://colab.research.google.com/github/coreylammie/MemTorch/blob/master/tutorials/Exemplar_Simulations.ipynb)

### Case Study (Legacy)

An application of MemTorch in epileptic seizure detection.

**Link**: [Open in Colab](https://colab.research.google.com/github/coreylammie/MemTorch/blob/master/tutorials/Case_Study.ipynb)

### Novel Simulations (Legacy)

Explores simulations using the CIFAR-10 dataset.

**Link**: [Open in Colab](https://colab.research.google.com/github/coreylammie/MemTorch/blob/master/tutorials/Novel_Simulations.ipynb)

These tutorials are accessible via [Google Colab](https://memtorch.readthedocs.io/en/latest/tutorials.html), allowing users to run them without local setup.

---

## 🧠 Academic Reference

For an in-depth understanding of the framework and its applications, refer to the original paper:

* **Title**: *MemTorch: An Open-source Simulation Framework for Memristive Deep Learning Systems*
* **Authors**: Corey Lammie, Wei Xiang, Bernabé Linares-Barranco, Mostafa Rahimi Azghadi
* **Published**: April 23, 2020
* **Abstract**: Discusses the potential of memristive devices in accelerating deep learning systems and introduces MemTorch as a tool for simulating such systems, accounting for device non-idealities and peripheral circuitry.