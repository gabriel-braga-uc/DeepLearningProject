# Projeto 1: Reconhecimento de Objetos com YOLO

**MS960/MT862: Tópicos em Aprendizado de Máquina (Deep Learning)**
Prof. João Florindo, IMECC/UNICAMP, 2026/1

## Estrutura dos arquivos

```
.
├── YOLO_Projeto1.ipynb   # Notebook principal (editável)
├── utils.py              # Funções manuais de IoU e NMS
├── Relatorio.tex         # Código fonte do relatório (editável)
├── Relatorio.pdf         # Relatório compilado
├── resultados/           # Imagens com bounding boxes
│   ├── padrao/           # Detecção em cada imagem com s_min=0.5, IoU=0.45
│   ├── limiares/         # Variação de IoU e score em horses.jpg e city_scene.jpg
│   └── grade_iou_horses.jpg  # Grade comparativa de IoU
└── README.md             # Este arquivo
```

## Como rodar o notebook

O notebook depende de três pastas adicionais que vêm no enunciado (`cfg/`, `data/`, `images/`, `weights/`). A pasta `weights/` contém o arquivo `yolov3.weights` com os pesos pré-treinados.

Estrutura esperada do diretório de trabalho:

```
.
├── YOLO_Projeto1.ipynb
├── utils.py
├── cfg/yolov3.cfg
├── data/coco.names
├── images/
│   ├── cat.jpg
│   ├── dog.jpg
│   └── ...
└── weights/yolov3.weights
```

### Dependências

```bash
pip install torch torchvision pillow matplotlib numpy
```

### Execução

Basta abrir o notebook `YOLO_Projeto1.ipynb` (Jupyter, VS Code ou Google Colab) e executar as células em sequência. Em CPU a inferência leva alguns segundos por imagem; em GPU (Colab com `cuda`) é praticamente instantânea.

## Resumo dos resultados principais

Detecção em todas as imagens com limiares padrão ($s_{\min} = 0.5$, $\tau_{\mathrm{IoU}} = 0.45$):

| Imagem | Objetos | Classes |
|---|---|---|
| dog.jpg | 3 | bicycle, dog, truck |
| horses.jpg | 4 | horse (×4) |
| city_scene.jpg | 15 | person, car, traffic light, truck |
| food.jpg | 18 | bowl, spoon, diningtable, fork |
| eagle.jpg | 1 | bird |
| giraffe.jpg | 2 | giraffe, zebra |
| cat.jpg | 1 | cat |
| dog2.jpg | 7 | dog, person, bicycle, umbrella |
| motorbike.jpg | 2 | motorbike, person |
| person.jpg | 3 | person, dog, horse |
| surf.jpg | 2 | person, surfboard |
| wine.jpg | 4 | wine glass, bottle, chair |

## Observações

- O notebook segue a estrutura do notebook de referência distribuído pelo professor, com a separação `from utils import manual_nms, manual_iou`.
- A função `manual_nms` recebe `(boxes, scores, classes, iou_threshold)` e processa o NMS de forma independente por classe, conforme a errata.
- Tudo é editável: marcadores claros para os nomes do grupo no notebook (capa) e no relatório (`Integrante 1, 2, 3, 4`).

## Compilação do relatório

```bash
pdflatex Relatorio.tex
pdflatex Relatorio.tex   # segunda passada para fixar referências cruzadas
```
