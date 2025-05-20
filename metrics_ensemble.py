import torch
from torch import nn


class Dice(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Initialize the DiceLoss class.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Calculate the Dice loss.

        Args:
            preds (torch.Tensor): Predicted tensor of shape (batch_size, num_classes, signal_length).
            targets (torch.Tensor): Ground truth tensor of shape (batch_size, signal_length) with class labels.

        Returns:
            torch.Tensor: Dice loss.
        """
        num_classes = 3
        # Apply softmax to ensure predictions are in the probability space if logits are provided

        # Initialize dice score list
        dice_scores = []

        # Calculate Dice coefficient for each class
        for class_idx in range(num_classes):
            # Create binary masks for predictions and targets
            pred_class = (preds == class_idx).float()  # shape: (batch_size, signal_length)
            target_class = (targets == class_idx).float()  # shape: (batch_size, signal_length)

            # Calculate intersection and union
            intersection = torch.sum(pred_class * target_class, dim=1)  # sum over signal_length
            union = torch.sum(pred_class, dim=1) + torch.sum(target_class, dim=1)

            # Dice score for current class
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)

        # Stack and average scores
        dice_scores = torch.stack(dice_scores, dim=1)  # shape: (batch_size, num_classes)
        dice = dice_scores.mean()  # average over classes and batches

        return dice
class Dice_Acc(nn.Module):
    def __init__(self, smooth=1e-6):
        super(Dice_Acc, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        # Apply softmax to ensure predictions are in the probability space if logits are provided
        #加速标签为1
        class_idx = 1
        # Create binary masks for predictions and targets
        pred_class = (preds == class_idx).float()  # shape: (batch_size, signal_length)
        target_class = (targets == class_idx).float()  # shape: (batch_size, signal_length)

        # Calculate intersection and union
        intersection = torch.sum(pred_class * target_class, dim=1)  # sum over signal_length
        union = torch.sum(pred_class, dim=1) + torch.sum(target_class, dim=1)

        # Dice score for current class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)


        return dice_score


class Dice_Dec(nn.Module):
    def __init__(self, smooth=1e-6):
        super(Dice_Dec, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Apply softmax to ensure predictions are in the probability space if logits are provided
        # 减速标签为2
        class_idx = 2
        # Create binary masks for predictions and targets
        pred_class = (preds == class_idx).float()  # shape: (batch_size, signal_length)
        target_class = (targets == class_idx).float()  # shape: (batch_size, signal_length)

        # Calculate intersection and union
        intersection = torch.sum(pred_class * target_class, dim=1)  # sum over signal_length
        union = torch.sum(pred_class, dim=1) + torch.sum(target_class, dim=1)

        # Dice score for current class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return dice_score


class IOU(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IOU, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        # preds: (B, C, L) 未经argmax的输出
        # labels: (B, L) 类别索引
        num_classes = 3
        labels = labels.long()

        iou_per_class = []
        for c in range(num_classes):
            pred_c = (preds == c)
            label_c = (labels == c)

            intersection = (pred_c & label_c).sum().float()
            union = (pred_c | label_c).sum().float()

            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_per_class.append(iou)

        miou = torch.mean(torch.stack(iou_per_class))
        return miou

class IOU_Acc(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IOU_Acc, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        # preds: (B, C, L) 未经argmax的输出
        # labels: (B, L) 类别索引
        labels = labels.long()
        class_id = 1
        pred_c = (preds == class_id)
        label_c = (labels == class_id)

        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou

class IOU_Dec(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IOU_Dec, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        # preds: (B, C, L) 未经argmax的输出
        # labels: (B, L) 类别索引
        labels = labels.long()
        class_id = 2
        pred_c = (preds == class_id)
        label_c = (labels == class_id)

        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou



def mstatscompare_P(ground_labels,pred_labels):
    ground_cls = ground_labels.detach().cpu().squeeze().numpy().copy()
    pred_cls = pred_labels.detach().cpu().squeeze().numpy().copy()  # (32,4800)  # Predicted labels

    Index_Dice = Dice()
    Index_Dice_Acc = Dice_Acc()
    Index_Dice_Dec = Dice_Dec()

    Index_Iou = IOU()
    Index_Iou_Acc = IOU_Acc()
    Index_Iou_Dec = IOU_Dec()
    stats = {}
    # Dice指标
    stats['Dice'] = Index_Dice(pred_labels, ground_labels).cpu().numpy()
    stats['Dice_Acc'] = Index_Dice_Acc(pred_labels, ground_labels).cpu().numpy()
    stats['Dice_Dec'] = Index_Dice_Dec(pred_labels, ground_labels).cpu().numpy()

    # Iou指标
    stats['Iou'] = Index_Iou(pred_labels, ground_labels).cpu().numpy()
    stats['Iou_Acc'] = Index_Iou_Acc(pred_labels, ground_labels).cpu().numpy()
    stats['Iou_Dec'] = Index_Iou_Dec(pred_labels, ground_labels).cpu().numpy()
    #Accuracy
    correct = (pred_cls == ground_cls).sum()
    stats['Accuracy'] = (correct / len(pred_cls))


    return stats
