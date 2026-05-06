"""
utils.py: implementacoes manuais de IoU e Supressao Nao Maximal (NMS)
para o Projeto 1 de MS960/MT862 (Deep Learning).

A IoU mede a sobreposicao entre duas caixas: razao entre a area da
intersecao e a area da uniao. O NMS usa a IoU para descartar caixas
redundantes que se referem a um mesmo objeto. Conforme a errata do
professor, o NMS recebe tambem o vetor de classes e e aplicado de
forma independente para cada classe.

Formato das caixas adotado em todo o projeto: [y_min, x_min, y_max, x_max].
"""
import torch


def manual_iou(box1, boxes2):
    """
    Calcula a IoU entre uma caixa de referencia e um conjunto de caixas.

    Parametros
    ----------
    box1   : tensor de forma (4,)   no formato [y1, x1, y2, x2]
    boxes2 : tensor de forma (N, 4) no mesmo formato

    Retorno
    -------
    tensor de forma (N,) com os valores de IoU em [0, 1].
    """
    # Coordenadas da caixa de referencia
    y1_1, x1_1, y2_1, x2_1 = box1[0], box1[1], box1[2], box1[3]

    # Coordenadas das demais caixas
    y1_2 = boxes2[:, 0]
    x1_2 = boxes2[:, 1]
    y2_2 = boxes2[:, 2]
    x2_2 = boxes2[:, 3]

    # Cantos da regiao de intersecao
    inter_y1 = torch.maximum(y1_1, y1_2)
    inter_x1 = torch.maximum(x1_1, x1_2)
    inter_y2 = torch.minimum(y2_1, y2_2)
    inter_x2 = torch.minimum(x2_1, x2_2)

    # Largura e altura da intersecao (zero se nao se sobrepoem)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_area = inter_h * inter_w

    # Areas individuais
    area1 = torch.clamp(y2_1 - y1_1, min=0) * torch.clamp(x2_1 - x1_1, min=0)
    area2 = torch.clamp(y2_2 - y1_2, min=0) * torch.clamp(x2_2 - x1_2, min=0)

    # Area da uniao (com pequena protecao numerica)
    union_area = area1 + area2 - inter_area
    return inter_area / torch.clamp(union_area, min=1e-9)


def manual_nms(boxes, scores, classes, iou_threshold=0.45):
    """
    Supressao Nao Maximal aplicada de forma independente por classe.

    A logica eh: para cada classe presente, ordene as caixas por score
    em ordem decrescente; pegue a caixa de maior score, marque as demais
    com IoU acima do limiar como redundantes e descarte. Repita o
    processo com as caixas restantes ate esgotar.

    Parametros
    ----------
    boxes        : tensor (N, 4) no formato [y1, x1, y2, x2]
    scores       : tensor (N,)  com a confianca final de cada caixa
    classes      : tensor (N,)  com o indice de classe de cada caixa
    iou_threshold: float em [0, 1]; caixas com IoU acima deste valor
                   serao descartadas

    Retorno
    -------
    tensor de indices (long) das caixas mantidas, ordenadas por score
    decrescente.
    """
    keep_global = []
    classes_unicas = torch.unique(classes)

    for cls in classes_unicas:
        # Seleciona apenas as caixas desta classe
        mask_cls = (classes == cls)
        idx_cls = torch.nonzero(mask_cls, as_tuple=False).flatten()

        boxes_cls = boxes[mask_cls]
        scores_cls = scores[mask_cls]

        # Ordena por score em ordem decrescente
        ordem = torch.argsort(scores_cls, descending=True)

        keep_local = []
        while ordem.numel() > 0:
            # Caixa de maior score sobrevive
            i = ordem[0].item()
            keep_local.append(idx_cls[i].item())

            if ordem.numel() == 1:
                break

            # Calcula IoU contra as demais e mantem apenas as de baixa sobreposicao
            ious = manual_iou(boxes_cls[i], boxes_cls[ordem[1:]])
            mantidas = torch.nonzero(ious <= iou_threshold, as_tuple=False).flatten()
            ordem = ordem[1:][mantidas]

        keep_global.extend(keep_local)

    if len(keep_global) == 0:
        return torch.empty(0, dtype=torch.long)

    # Reordena globalmente pelos scores (apresentacao mais util)
    keep_global = torch.tensor(keep_global, dtype=torch.long)
    nova_ordem = torch.argsort(scores[keep_global], descending=True)
    return keep_global[nova_ordem]
