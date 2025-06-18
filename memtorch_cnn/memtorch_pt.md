# 📘 Documentação da API MemTorch

**Versão**: 1.1.6
**Fonte**: [Documentação MemTorch](https://memtorch.readthedocs.io/en/latest/)
**Visão Geral**: MemTorch é um framework de simulação para sistemas de aprendizado profundo memristivos, integrando-se perfeitamente com PyTorch. Permite modelagem de dispositivos memristivos, suas não-idealidades, e mapeamento de componentes de redes neurais para hardware memristivo.

---

## 📂 Estrutura de Módulos

* [`memtorch.bh`](#memtorchbh-modelagem-comportamental)
* [`memtorch.map`](#memtorchmap-mapeamento-e-escalonamento)
* [`memtorch.mn`](#memtorchmn-módulos-de-rede-neural-memristiva)
* [📘 Tutoriais e Exemplos](#-tutoriais-e-exemplos)

---

## `memtorch.bh`: Modelagem Comportamental

Este módulo fornece ferramentas para simular o comportamento de dispositivos memristivos e arrays crossbar.

### `memtorch.bh.memristor`

Encapsula vários modelos de memristor:

* **LinearIonDrift**: Modela comportamento de deriva iônica linear.
* **VTEAM**: Modelo de Memristor Adaptativo de Limiar de Tensão.
* **Data\_Driven**: Modelo baseado em dados empíricos.

**Exemplo**:

```python
from memtorch.bh.memristor import VTEAM

memristor = VTEAM(r_on=100, r_off=16000)
```

### `memtorch.bh.nonideality`

Modela comportamentos não-ideais como:

* Estados de condutância finitos
* Falhas de dispositivo
* Características I/V não-lineares
* Efeitos de resistência e retenção

**Exemplo**:

```python
from memtorch.bh.nonideality import NonIdeality

non_ideal = NonIdeality()
non_ideal.apply_nonidealities(model)
```

### `memtorch.bh.crossbar`

Simula arquiteturas crossbar.

#### `Crossbar`

Modela crossbars memristivos e gerencia tiles crossbar modulares.

**Exemplo**:

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

Fornece rotinas para programar a condutância de dispositivos dentro de um crossbar.

#### `Tile`

Facilita a criação de tiles crossbar modulares para representar redes de grande escala.

---

## `memtorch.map`: Mapeamento e Escalonamento

Lida com a tradução de parâmetros de redes neurais e entradas para equivalentes de hardware memristivo.

### `memtorch.map.Input`

Codifica valores de entrada como tensões de bit-line.

**Exemplo**:

```python
from memtorch.map.Input import naive_scale

scaled_input = naive_scale(module, input_tensor)
```

### `memtorch.map.Parameter`

Mapeia pesos de redes neurais para valores de condutância de dispositivos.

**Exemplo**:

```python
from memtorch.map.Parameter import naive_map

mapped_params = naive_map(layer)
```

### `memtorch.map.Module`

Determina relações entre correntes de leitura de crossbars memristivos e saídas desejadas.

**Exemplo**:

```python
from memtorch.map.Module import naive_tune

naive_tune(module, input_shape=(1, 28, 28))
```

---

## `memtorch.mn`: Módulos de Rede Neural Memristiva

Oferece equivalentes memristivos de camadas de redes neurais PyTorch.

### `memtorch.mn.Module`

Inclui a função `patch_model` para converter modelos PyTorch padrão em versões memristivas.

**Exemplo**:

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

### Implementações de Camadas

Fornece versões memristivas de camadas:

* `Linear`
* `Conv1d`
* `Conv2d`
* `Conv3d`
* `RNN`

**Exemplo**:

```python
from memtorch.mn import Linear
from memtorch.bh.memristor import VTEAM

memristive_linear = Linear(torch.nn.Linear(10, 5),
                           memristor_model=VTEAM,
                           memristor_model_params={})
```

---

## 📘 Tutoriais e Exemplos

MemTorch oferece uma suíte de tutoriais interativos em formato Jupyter Notebook para ajudar usuários a começar e explorar recursos avançados:

### Tutorial Introdutório

Um ponto de partida para novos usuários entenderem os básicos do MemTorch.

**Link**: [Abrir no Colab](https://colab.research.google.com/github/coreylammie/MemTorch/blob/master/tutorials/Introductory_Tutorial.ipynb)

### Simulações Exemplares

Demonstra várias simulações apresentadas no artigo original do MemTorch.

**Link**: [Abrir no Colab](https://colab.research.google.com/github/coreylammie/MemTorch/blob/master/tutorials/Exemplar_Simulations.ipynb)

### Estudo de Caso (Legado)

Uma aplicação do MemTorch na detecção de convulsões epilépticas.

**Link**: [Abrir no Colab](https://colab.research.google.com/github/coreylammie/MemTorch/blob/master/tutorials/Case_Study.ipynb)

### Simulações Novas (Legado)

Explora simulações usando o dataset CIFAR-10.

**Link**: [Abrir no Colab](https://colab.research.google.com/github/coreylammie/MemTorch/blob/master/tutorials/Novel_Simulations.ipynb)

Estes tutoriais são acessíveis via [Google Colab](https://memtorch.readthedocs.io/en/latest/tutorials.html), permitindo aos usuários executá-los sem configuração local.

---

## 🧠 Referência Acadêmica

Para um entendimento aprofundado do framework e suas aplicações, consulte o artigo original:

* **Título**: *MemTorch: Um Framework de Simulação de Código Aberto para Sistemas de Aprendizado Profundo Memristivos*
* **Autores**: Corey Lammie, Wei Xiang, Bernabé Linares-Barranco, Mostafa Rahimi Azghadi
* **Publicado**: 23 de Abril de 2020
* **Resumo**: Discute o potencial de dispositivos memristivos na aceleração de sistemas de aprendizado profundo e introduz MemTorch como uma ferramenta para simular tais sistemas, considerando não-idealidades de dispositivos e circuitos periféricos.
