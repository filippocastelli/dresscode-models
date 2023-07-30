import torch

def iou_metric(y_pred_batch: torch.Tensor,
                y_true_batch: torch.Tensor) -> float:
    B = y_pred_batch.shape[0]
    iou = 0
    for i in range(B):
        y_pred = y_pred_batch[i]
        y_true = y_true_batch[i]
        # y_pred is not one-hot, so need to threshold it
        y_pred = y_pred > 0.5
        
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

    
        intersection = torch.sum(y_pred[y_true == 1])
        union = torch.sum(y_pred) + torch.sum(y_true)

    
        iou += (intersection + 1e-7) / (union - intersection + 1e-7) / B
    return iou