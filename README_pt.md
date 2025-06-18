# Melhorias do MobileNetV2 para Classificação de Doenças Foliares

Este projeto implementa e avalia várias melhorias na arquitetura MobileNetV2 para tarefas de classificação de doenças foliares. As melhorias são implementadas em estágios, permitindo testes e avaliações incrementais.

## 🔄 Atualizações Recentes

**[2025-05-28]** Adicionada implementação CNN baseada em MemTorch:
- Implementada uma CNN completa usando MemTorch para modelagem de dispositivos memristores
- Adicionada abordagem de treinamento híbrido com fases ex-situ e in-situ
- Integrado treinamento consciente de hardware com restrições de memristores
- Incluídas ferramentas de análise de eficiência energética e latência
- Criada documentação abrangente e testes

**[2025-04-27]** Reduzido tamanho do modelo usando width_mult=0.75:
- Diminuído tamanho do modelo em ~38.5% (de 8.7MB para 5.3MB)
- Adicionado parâmetro width_mult a todas as variantes do modelo
- Atualizadas configurações do modelo para usar tamanho reduzido por padrão
- Adicionados testes para verificar redução de tamanho

**[2025-04-27]** Corrigida implementação dos módulos Triplet Attention e CNSN:
- Corrigida implementação do Triplet Attention com rotações de tensor adequadas e Z-pooling
- Corrigido módulo CNSN com troca adequada de estatísticas CrossNorm e recalibração SelfNorm
- Adicionada suíte de testes abrangente para ambos os módulos
- Melhorada integração com arquitetura MobileNetV2

## Estrutura do Projeto

O projeto foi reorganizado com uma estrutura unificada:

```
project_root/
├── mobilenetv2_improvements/  # Implementação de melhorias MobileNetV2
│   ├── base_mobilenetv2/      # Definição do modelo MobileNetV2 base
│   │   └── models/            # Implementação da arquitetura do modelo
│   ├── stage1_mish/           # MobileNetV2 com ativação Mish
│   │   └── models/            # Implementação da arquitetura do modelo
│   ├── stage2_triplet/        # MobileNetV2 com Mish e Triplet Attention
│   │   └── models/            # Implementação da arquitetura do modelo
│   ├── stage3_cnsn/           # MobileNetV2 com Mish, Triplet Attention e CNSN
│   │   └── models/            # Implementação da arquitetura do modelo
│   ├── configs/               # Arquivos de configuração unificados
│   └── utils/                 # Funções utilitárias unificadas
├── memristor_cnn/             # Implementação CNN personalizada baseada em memristor
│   ├── models/                # Arquitetura do modelo memristor
│   └── utils/                 # Funções utilitárias memristor
├── memtorch_cnn/              # Implementação CNN baseada em MemTorch
│   ├── models/                # Arquitetura do modelo MemTorch
│   ├── utils/                 # Funções utilitárias MemTorch
│   └── tests/                 # Testes para implementação MemTorch
├── datasets/                  # Armazenamento de datasets (não rastreado pelo git)
├── checkpoints/               # Checkpoints do modelo (organizados por tipo de modelo)
├── experiments/               # Resultados e logs de experimentos
└── logs/                      # Logs de treinamento e avaliação
```

## Executando Comparação de Modelos

Para comparar todas as variantes do modelo com e sem redução de parâmetros:

```bash
# Comparação básica
python train_comparison.py --data_dir datasets/leaf_disease

# Com aumento de dados aprimorado
python train_comparison.py --data_dir datasets/leaf_disease --enhanced_augmentation

# Com aceleração GPU
python train_comparison.py --data_dir datasets/leaf_disease --device cuda
```

Isso treinará e avaliará os seguintes modelos:
1. MobileNetV2 base sem redução de parâmetros (width_mult=1.0)
2. MobileNetV2 base com redução de parâmetros (width_mult=0.75)
3. MobileNetV2 com Mish e redução de parâmetros (width_mult=0.75)
4. MobileNetV2 com Mish, Triplet Attention e redução de parâmetros (width_mult=0.75)
5. MobileNetV2 com Mish, Triplet Attention, CNSN e redução de parâmetros (width_mult=0.75)

Os resultados serão salvos em `experiments/comparison/model_comparison_results.csv` com métricas detalhadas.

## Implementações de Modelos

### 1. MobileNetV2 Base

O modelo fundamental com:
- Estrutura residual invertida
- Convoluções separáveis em profundidade
- Gargalos lineares
- Função de ativação ReLU6
- Multiplicador de largura opcional (padrão: 0.75) para reduzir tamanho do modelo

### 2. MobileNetV2 com Ativação Mish (Estágio 1)

Substitui ReLU6 pela função de ativação Mish para melhorar características não-lineares:
- Função de ativação mais suave com fórmula: `f(x) = x * tanh(softplus(x))`
- Melhor propagação de gradiente sem problemas de gradiente desaparecendo
- Sem truncamento de limite superior (diferente do limite de 6 do ReLU6)
- Permite saídas negativas, preservando informações negativas importantes

### 3. MobileNetV2 com Mish e Triplet Attention (Estágio 2)

Adiciona uma estrutura de atenção de três ramos para capturar interações entre dimensões:
- Três ramos paralelos focando em diferentes interações dimensionais:
  - Ramo 1: Interação Canal-Altura
  - Ramo 2: Interação Canal-Largura
  - Ramo 3: Atenção espacial padrão
- Operação Z-Pool combina max-pooling e average-pooling
- Sem redução de canal, preservando fluxo completo de informações
- Design eficiente em parâmetros com sobrecarga computacional mínima

### 4. MobileNetV2 com Mish, Triplet Attention e CNSN (Estágio 3)

Integra módulos CrossNorm e SelfNorm:
- CrossNorm: Amplia distribuição de treinamento trocando estatísticas por canal
  - Ativo apenas durante treinamento
  - Cria representações de características diversas
  - Múltiplos modos: 1-instância, 2-instâncias e crop
- SelfNorm: Conecta lacuna de distribuição treino-teste usando mecanismo de atenção
  - Ativo durante treinamento e teste
  - Usa funções de atenção para recalibrar estatísticas
  - Processa cada canal independentemente

### 5. Implementação CNN Baseada em Memristor

Implementação personalizada de CNN baseada em memristor:
- Usa arrays crossbar de memristor simulados para computação
- Implementa pares de condutância diferencial para representação de pesos
- Inclui variação de dispositivo e simulação de deriva de estado
- Fornece análise básica de eficiência energética e latência

### 6. Implementação CNN Baseada em MemTorch

Implementação avançada usando framework MemTorch:
- Modelagem precisa de dispositivo memristor com modelo LinearIonDrift
- Treinamento consciente de hardware com restrições de memristor
- Abordagem de treinamento híbrido:
  - Treinamento ex-situ em hardware convencional
  - Transferência de pesos para arrays memristor
  - Ajuste fino in-situ com atualizações baseadas em limiar
- Análise abrangente de eficiência energética e latência
- Suporte integrado para variações de dispositivo e não-idealidades

## Instruções de Configuração

### Configuração do Ambiente

1. Clone o repositório:
```bash
git clone <repository-url>
cd mobilenetv2_improvements
```

2. Crie e ative um ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Configuração de Aceleração GPU

Para treinamento mais rápido com GPUs NVIDIA:

1. Instale CUDA Toolkit e cuDNN:
   - Baixe e instale [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.8 ou 12.1 recomendado)
   - Baixe e instale [cuDNN](https://developer.nvidia.com/cudnn) (requer conta gratuita NVIDIA)

2. Instale PyTorch habilitado para GPU:
```bash
# Ative seu ambiente virtual primeiro
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Para CUDA 11.8
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. Verifique detecção de GPU:
```bash
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Preparação do Dataset

1. Baixe um dataset de doenças foliares (ex: Plant Village, PlantDoc, Rice Leaf Disease Dataset)
2. Organize-o de acordo com a seguinte estrutura:
```
datasets/
└── leaf_disease/
    ├── class1/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    ├── class2/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    └── ...
```

O dataset será automaticamente dividido em conjuntos de treino, validação e teste quando usado pela primeira vez.

## Executando os Modelos

### Treinamento

Todas as variantes do modelo são agora treinadas usando o script unificado `train.py`:

#### Comandos Básicos de Treinamento

```bash
# MobileNetV2 Base
python train.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 com Mish
python train.py --data_dir datasets/leaf_disease --model_type mish --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 com Mish e Triplet Attention
python train.py --data_dir datasets/leaf_disease --model_type triplet --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 com Mish, Triplet Attention e CNSN
python train.py --data_dir datasets/leaf_disease --model_type cnsn --epochs 60 --batch_size 32 --lr 0.001
```

#### Treinamento com Aumento de Dados Aprimorado

```bash
# Adicione flag --enhanced_augmentation para usar aumento de dados aprimorado
python train.py --data_dir datasets/leaf_disease --model_type base --enhanced_augmentation --epochs 60 --batch_size 32 --lr 0.001
```

#### Comandos de Treinamento GPU

```bash
# Adicione --device cuda para usar aceleração GPU
python train.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

### Avaliação

Todas as variantes do modelo são avaliadas usando o script unificado `evaluate.py`:

```bash
# MobileNetV2 Base
python evaluate.py --data_dir datasets/leaf_disease --model_type base --checkpoint checkpoints/base/model_best.pth

# MobileNetV2 com Mish
python evaluate.py --data_dir datasets/leaf_disease --model_type mish --checkpoint checkpoints/mish/model_best.pth
```

## Estratégia de Divisão do Dataset

Este projeto usa uma estratégia de divisão em três partes:
1. **Conjunto de teste**: 10% de todo o dataset
2. **Conjunto de treinamento**: 72% de todo o dataset (80% dos 90% restantes)
3. **Conjunto de validação**: 18% de todo o dataset (20% dos 90% restantes)

A divisão é realizada automaticamente ao carregar o dataset pela primeira vez, criando diretórios separados para cada divisão mantendo a estrutura de classes.

## Testes

Execute testes para verificar a implementação:

```bash
pytest
```

## Diretrizes de Desenvolvimento

1. Siga diretrizes de estilo PEP 8
2. Escreva docstrings para todas as funções e classes
3. Adicione testes unitários para nova funcionalidade
4. Use branches git para novas funcionalidades
5. Execute testes antes de fazer commits

## Licença

[Licença MIT](LICENSE)
