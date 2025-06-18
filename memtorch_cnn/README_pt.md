# CNN Baseada em MemTorch para Classificação de Doenças Foliares

Este projeto implementa uma Rede Neural Convolucional (CNN) baseada em MemTorch para classificação de doenças foliares. A implementação aproveita arrays crossbar de memristores para computação eficiente, fornecendo melhorias significativas em eficiência energética e latência comparado a implementações CNN convencionais.

## Características

- **Integração MemTorch**: Usa MemTorch para modelagem precisa de dispositivos memristores
- **Abordagem de Treinamento Híbrido**: Treinamento ex-situ seguido por ajuste fino in-situ
- **Treinamento Consciente de Hardware**: Considera restrições de memristores durante treinamento
- **Análise de Eficiência Energética**: Compara consumo de energia com CNNs convencionais
- **Análise de Latência**: Compara latência de inferência com CNNs convencionais
- **Modelagem de Não-Idealidades**: Simula variações de dispositivo e deriva de estado

## Instalação

### Pré-requisitos

- Python 3.7+
- PyTorch 1.7+
- CUDA (opcional, para aceleração GPU)

### Configuração

1. Clone o repositório:
```bash
git clone <repository-url>
cd ic-usp-ep
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r memtorch_cnn/requirements.txt
```

## Uso

### Treinamento

Para treinar o modelo CNN baseado em MemTorch:

```bash
python memtorch_cnn/train.py --data_dir datasets/leaf_disease --device cuda --batch_size 32 --ex_situ_epochs 50 --in_situ_epochs 10
```

#### Opções de Treinamento

- `--data_dir`: Caminho para o diretório do dataset
- `--enhanced_augmentation`: Usar aumento de dados aprimorado
- `--width_mult`: Multiplicador de largura para a rede (padrão: 0.75)
- `--tile_shape`: Forma dos tiles crossbar memristor (padrão: 128 128)
- `--adc_resolution`: Resolução ADC em bits (padrão: 8)
- `--dac_resolution`: Resolução DAC em bits (padrão: 8)
- `--max_input_voltage`: Tensão máxima de entrada (padrão: 0.3)
- `--batch_size`: Tamanho do lote para treinamento (padrão: 32)
- `--ex_situ_epochs`: Número de épocas de treinamento ex-situ (padrão: 50)
- `--in_situ_epochs`: Número de épocas de treinamento in-situ (padrão: 10)
- `--lr`: Taxa de aprendizado inicial (padrão: 0.001)
- `--weight_decay`: Decaimento de peso (padrão: 1e-4)
- `--threshold`: Limiar para atualizações de peso in-situ (padrão: 0.1)
- `--device`: Dispositivo a usar (cuda ou cpu)
- `--checkpoint_dir`: Diretório para salvar checkpoints
- `--results_dir`: Diretório para salvar resultados
- `--skip_ex_situ`: Pular fase de treinamento ex-situ
- `--skip_in_situ`: Pular fase de treinamento in-situ
- `--resume`: Caminho para checkpoint para retomar
- `--debug`: Habilitar modo debug (tamanho de dataset reduzido)

### Avaliação

Para avaliar um modelo treinado:

```bash
python memtorch_cnn/evaluate.py --data_dir datasets/leaf_disease --checkpoint checkpoints/memtorch_cnn/model_best_in_situ.pth --device cuda
```

#### Opções de Avaliação

- `--data_dir`: Caminho para o diretório do dataset
- `--width_mult`: Multiplicador de largura para a rede (padrão: 0.75)
- `--checkpoint`: Caminho para o checkpoint a avaliar
- `--device`: Dispositivo a usar (cuda ou cpu)
- `--batch_size`: Tamanho do lote para avaliação (padrão: 32)
- `--results_dir`: Diretório para salvar resultados

## Arquitetura do Modelo

A CNN baseada em MemTorch usa uma arquitetura similar ao MobileNetV2 com os seguintes componentes:

1. **Primeira Camada Conv**: Camada convolucional padrão (mantida como digital)
2. **Blocos Residuais Invertidos**: Convertidos para camadas memristivas
3. **Última Camada Conv**: Convertida para camada memristiva
4. **Pooling Médio Global**: Camada de pooling padrão (mantida como digital)
5. **Classificador**: Camada totalmente conectada convertida para camada memristiva

### Configuração do Memristor

- **Modelo de Dispositivo**: LinearIonDrift
- **Tamanho do Array Crossbar**: 128×128
- **Faixa de Resistência**: 100Ω (ON) a 16kΩ (OFF)
- **Resolução ADC/DAC**: 8 bits
- **Quantização de Peso**: 4 bits (16 níveis)

## Abordagem de Treinamento Híbrido

O processo de treinamento consiste em duas fases:

1. **Treinamento Ex-situ**:
   - Treinamento convencional em GPU/CPU
   - Todas as camadas são treináveis
   - Retropropagação padrão

2. **Transferência de Pesos**:
   - Converter modelo para memristivo
   - Aplicar quantização de pesos
   - Aplicar não-idealidades

3. **Treinamento In-situ**:
   - Congelar camadas convolucionais
   - Atualizar apenas pesos da camada FC
   - Atualizações baseadas em limiar
   - Treinamento consciente de hardware

## Análise de Performance

A CNN baseada em MemTorch fornece melhorias significativas sobre CNNs convencionais:

- **Eficiência Energética**: Melhoria de 10-100×
- **Latência**: Melhoria de 2-5×
- **Tamanho do Modelo**: Similar à CNN convencional

## Testes

Para executar os testes:

```bash
python -m unittest discover memtorch_cnn/tests
```

## Comparação com Implementação Memristor Personalizada

Esta implementação baseada em MemTorch oferece várias vantagens sobre a implementação memristor personalizada:

1. **Modelos de Dispositivo Mais Precisos**: MemTorch inclui modelos memristor realistas que capturam física real do dispositivo
2. **Não-idealidades Integradas**: Simula variações dispositivo-a-dispositivo, deriva de estado e outras características não-ideais
3. **Integração Perfeita com PyTorch**: Estende a classe Module do PyTorch para conversão fácil de modelo
4. **Treinamento Consciente de Hardware**: Considera restrições de memristor durante treinamento
5. **Análise de Energia e Performance**: Ferramentas integradas para analisar consumo de energia e performance

## Licença

[Licença MIT](LICENSE)

## Agradecimentos

- MemTorch: https://github.com/coreylammie/MemTorch
- PyTorch: https://pytorch.org/
- MobileNetV2: https://arxiv.org/abs/1801.04381
