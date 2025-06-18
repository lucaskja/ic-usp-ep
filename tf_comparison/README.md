# Comparação de Frameworks MobileNetV2

Este diretório contém código e resultados para comparar implementações MobileNetV2 entre PyTorch e TensorFlow.

## Configuração do Ambiente

- Python 3.12
- TensorFlow 2.19.0
- PyTorch 2.7.0

## Arquivos

- `compare_models.py`: Script para comparar diretamente implementações MobileNetV2 PyTorch e TensorFlow
- `detailed_comparison.md`: Análise abrangente dos resultados da comparação

## Principais Descobertas

1. **Contagem de Parâmetros**:
   - PyTorch MobileNetV2: 3.504.872 parâmetros
   - TensorFlow MobileNetV2: 3.538.984 parâmetros
   - Diferença: 34.112 parâmetros (0,96%)

2. **Tamanho do Modelo**:
   - Ambas implementações: 13,50 MB

3. **Distribuição de Camadas**:
   - PyTorch: 52 Conv2d, 52 BatchNorm2d, 1 Linear
   - TensorFlow: 35 Conv2D, 17 DepthwiseConv2D, 52 BatchNormalization, 1 Dense

4. **Performance de Inferência**:
   - PyTorch: 49,62 ms por inferência
   - TensorFlow: 59,58 ms por inferência
   - PyTorch é aproximadamente 20% mais rápido neste hardware

## Executando a Comparação

```bash
# Ative o ambiente virtual
source tf_venv/bin/activate

# Execute o script de comparação
python compare_models.py
```

## Conclusão

As implementações PyTorch e TensorFlow do MobileNetV2 são funcionalmente equivalentes com pequenas diferenças na contagem de parâmetros (0,96%) e detalhes de implementação. A principal diferença arquitetural está em como a normalização em lote é implementada e como as convoluções depthwise são tratadas.

Para fins de pesquisa, ambas as implementações podem ser consideradas equivalentes, com PyTorch mostrando uma ligeira vantagem de performance na velocidade de inferência neste hardware específico.
