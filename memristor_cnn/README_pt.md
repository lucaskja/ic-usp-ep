# MemristorCNN: CNN Baseada em Memristor para Detecção de Doenças Foliares

Este projeto implementa uma arquitetura de Rede Neural Convolucional baseada em Memristor (mCNN) otimizada para detecção de doenças foliares usando TTN-MobileNetV2 (atenção Triplet, normalização CNSN e ativação Mish).

## Visão Geral da Arquitetura

A arquitetura MemristorCNN combina várias inovações-chave:

1. **Modelo Base**: TTN-MobileNetV2 com:
   - Função de ativação Mish substituindo ReLU6
   - Triplet Attention para atenção espacial-canal aprimorada
   - CNSN (CrossNorm e SelfNorm) para normalização de características melhorada

2. **Integração Memristor**:
   - Arrays memristor de 2048 células (128×16 1T1R)
   - Estados de condutância de ponto fixo de 15 níveis
   - Pares de condutância diferencial para representação de pesos
   - Tensão de leitura de 0.2V e pulsos de programação de 50ns

3. **Abordagem de Treinamento Híbrido**:
   - Fase 1: Treinamento ex-situ (retropropagação convencional)
   - Transferência de pesos com programação em circuito fechado
   - Fase 2: Treinamento in-situ (atualizações baseadas em limiar apenas para camada FC)

## Características Principais

- **Eficiência Energética**: ~110x mais eficiente que implementações baseadas em GPU
- **Redução de Latência**: 3x mais rápido com conversores paralelos
- **Meta de Precisão**: >96% em tarefas de classificação de doenças foliares
- **Tolerância a Erros**: Compensação de treinamento híbrido para variações de dispositivo

## Estrutura de Diretórios

```
memristor_cnn/
├── models/                  # Implementação da arquitetura do modelo
│   ├── memristor_crossbar.py  # Implementação do array crossbar memristor
│   ├── memristor_cnn.py       # Arquitetura principal do modelo
│   ├── memristor_mapping.py   # Mapeamento de camadas para arrays memristor
│   └── memristor_pe.py        # Implementação do elemento de processamento
├── utils/                   # Funções utilitárias
│   ├── data_utils.py          # Utilitários de carregamento de dados
│   ├── evaluation_utils.py    # Utilitários de avaliação do modelo
│   ├── memristor_utils.py     # Utilitários específicos de memristor
│   └── training_utils.py      # Utilitários de treinamento
├── tests/                   # Casos de teste
│   └── test_memristor_cnn.py  # Testes para componentes do modelo
├── train.py                 # Script de treinamento
├── evaluate.py              # Script de avaliação
└── visualize.py             # Script de visualização
```

## Instalação

### Pré-requisitos

- Python 3.8+
- PyTorch 2.1.0+
- CUDA (recomendado para treinamento)

### Configuração

1. Clone o repositório:
```bash
git clone <repository-url>
cd memristor_cnn
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scikit-learn tqdm pandas seaborn torchviz graphviz
```

## Uso

### Treinamento

Para treinar o modelo usando a abordagem de treinamento híbrido:

```bash
# Treinamento básico
python train.py --data_dir datasets/leaf_disease --batch_size 100 --ex_situ_epochs 50 --in_situ_epochs 10

# Com aumento de dados aprimorado
python train.py --data_dir datasets/leaf_disease --enhanced_augmentation --batch_size 100

# Com aceleração GPU
python train.py --data_dir datasets/leaf_disease --device cuda --batch_size 100

# Pular treinamento ex-situ (se você tem um modelo pré-treinado)
python train.py --data_dir datasets/leaf_disease --skip_ex_situ --checkpoint checkpoints/memristor_cnn/model_best.pth
```

### Avaliação

Para avaliar um modelo treinado:

```bash
# Avaliação básica
python evaluate.py --data_dir datasets/leaf_disease --checkpoint checkpoints/memristor_cnn/model_best.pth

# Com aceleração GPU
python evaluate.py --data_dir datasets/leaf_disease --checkpoint checkpoints/memristor_cnn/model_best.pth --device cuda
```

### Visualização

Para visualizar arquitetura do modelo, histórico de treinamento e métricas de performance:

```bash
# Visualizar todos os aspectos
python visualize.py --results_dir results/memristor_cnn --visualize_all

# Visualizar aspectos específicos
python visualize.py --results_dir results/memristor_cnn --visualize_model --visualize_metrics
```

## Treinamento GPU no Windows

Para treinar o modelo em uma máquina Windows com suporte GPU:

### Configuração para Windows com GPU

1. **Crie e ative um ambiente virtual**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Instale PyTorch com suporte CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Nota: Para CUDA 12.1, use:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Instale outras dependências**:
   ```bash
   pip install numpy matplotlib scikit-learn tqdm pandas seaborn torchviz graphviz
   ```

4. **Verifique detecção de GPU**:
   ```bash
   python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
   ```

### Treinamento no Windows GPU

1. **Defina PYTHONPATH para a raiz do projeto**:
   ```bash
   set PYTHONPATH=C:\caminho\para\seu\projeto
   ```

2. **Execute o script de treinamento com GPU**:
   ```bash
   python train.py --data_dir datasets/leaf_disease --device cuda --batch_size 64 --ex_situ_epochs 50 --in_situ_epochs 10
   ```

3. **Para experimentação mais rápida**, você pode começar com:
   ```bash
   python train.py --data_dir datasets/leaf_disease --device cuda --batch_size 64 --ex_situ_epochs 10 --in_situ_epochs 5
   ```

### Requisitos de GPU

- GPU NVIDIA com suporte CUDA
- Pelo menos 4GB VRAM (8GB+ recomendado para tamanhos de lote maiores)
- Drivers NVIDIA atualizados
- Versão CUDA compatível (11.8 ou 12.1 recomendado)

## Mapeamento Memristor

O modelo mapeia camadas de rede neural para arrays memristor da seguinte forma:

1. **Camadas Convolucionais**:
   - Primeira camada convolucional (C1): kernels 3×3, 8 canais → PE1
   - Terceira camada convolucional (C3): kernels 3×3×8, 12 canais → PE1, PE3

2. **Camadas Totalmente Conectadas**:
   - Tamanho de entrada: 192, Tamanho de saída: 10 → PE5, PE7

## Métricas de Performance

Métricas de performance esperadas:

- **Precisão**: >96% na classificação de doenças foliares
- **Eficiência Energética**: 110x comparado à implementação GPU
- **Redução de Latência**: 3x com conversores paralelos
- **Pegada de Memória**: Significativamente reduzida comparado a modelos convencionais

## Testes

Execute a suíte de testes para verificar a implementação:

```bash
python -m unittest discover -s tests
```

## Referências

1. MobileNetV2: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
2. Triplet Attention: "Rotate to Attend: Convolutional Triplet Attention Module"
3. CNSN: "CrossNorm and SelfNorm for Generalization under Distribution Shifts"
4. Mish: "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
5. Memristor CNN: "Fully hardware-implemented memristor convolutional neural network"

## Licença

[Licença MIT](LICENSE)

## Solução de Problemas

### Erro de Incompatibilidade de Dispositivo CUDA

Se você encontrar o seguinte erro:
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

Este erro ocorre quando tensores em dispositivos diferentes (CPU e GPU) são usados juntos. A correção foi implementada na versão mais recente, que garante que todos os componentes do modelo sejam movidos para o mesmo dispositivo.

Se você ainda está enfrentando este problema:

1. Certifique-se de que está usando a versão mais recente do código
2. Verifique se sua GPU tem memória suficiente para o modelo
3. Tente reduzir o tamanho do lote com `--batch_size 32` ou ainda menor
4. Verifique se sua instalação CUDA está funcionando corretamente com:
   ```python
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
   ```
