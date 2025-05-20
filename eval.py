# %%
from Voting import hard_voting
from torch.utils.data import  Dataset
from torch.utils.data import DataLoader
from utils import *
from glob import glob
from WaveFHR_VNet import UNet
from metrics_ensemble import *
import numpy as np
class CTGDataset_test(Dataset):
    def __init__(self, fhr_paths, label_paths,baseline_paths):
        """
        Args:
            data_root (str): Root directory of the data.
            target_length (int): Target length for each sample in data points, set to 20 minutes at 4 Hz (20 * 240).
        """

        # Initialize lists to hold data
        self.fhrs_list = []
        self.labels_list = []
        self.baseline_list = []

        # Load each JSON file and process the data
        for fhr_path, label_path,baseline_path in zip(fhr_paths, label_paths,baseline_paths):#将数据和标签一一对应
            fhrs = np.load(fhr_path) / 255. #np.load函数来读取二进制信号数据 (4800,)归一化
            label = np.load(label_path)
            baseline = np.load(baseline_path)
            self.fhrs_list.append(fhrs)
            self.labels_list.append(label)
            self.baseline_list.append(baseline)

    def __len__(self):
        # Number of data samples
        return len(self.fhrs_list)

    def __getitem__(self, idx):
        """
        Retrieve data for a specific index from the preloaded lists.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            dict: Contains 'fhrs' and 'labels' as torch tensors, sampled and padded if necessary.
        """
        fhrs = self.fhrs_list[idx]
        labels = self.labels_list[idx]
        labels = np.where(labels == -1, 2, labels) #将CTU数据集的标签-1变成2
        baselines = self.baseline_list[idx]
        # Convert to PyTorch tensors

        fhrs_tensor = torch.tensor(fhrs, dtype=torch.float32).reshape(1, 4800)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        baseline_tensor = torch.tensor(baselines, dtype=torch.float32)

        return {
            'fhrs': fhrs_tensor,
            'labels': labels_tensor,
            'baselines':baseline_tensor
        }
# %%

# data_root = r'E:\暨南\小论文\Data\npy_pro\JNU_DB-331'
# data_root = r'E:\暨南\小论文\Data\npy_pro\GMU_DB-84'
# data_root = r'E:\暨南\小论文\Data\npy_pro\SMU_DB-234'
data_root = r'E:\暨南\小论文\Data\npy_pro\LCU_DB-66'

test_fhr_dir = os.path.join(data_root, 'test_sigs')  # 训练数据目录
test_label_dir = os.path.join(data_root, 'test_labels')  # 训练数据标签
test_baseline_dir = os.path.join(data_root, 'test_regression')  # 训练数据标签


test_fhr_paths = glob(os.path.join(test_fhr_dir, "*.npy"))  # 数据都是npy后缀
test_label_paths = [os.path.join(test_label_dir, fname.split('\\')[-1]) for fname in test_fhr_paths]
test_baseline_paths = [os.path.join(test_baseline_dir, fname.split('\\')[-1]) for fname in test_fhr_paths]

# %%
test_dataset = CTGDataset_test(test_fhr_paths, test_label_paths,test_baseline_paths)  # 返回字典'fhrs'和'labels' （1，4800）

# Initialize DataLoaders for train and test sets
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_bior = UNet(1, 3, wavelet='bior3.5').to(device)
model_bior.load_state_dict(torch.load('trained_model/WaveFHR-VNet/model_LCU_bior3.5.pt'))

model_db4 = UNet(1, 3, wavelet='db4').to(device)
model_db4.load_state_dict(torch.load('trained_model/WaveFHR-VNet/model_LCU_db4.pt'))

model_db6 = UNet(1, 3, wavelet='db6').to(device)
model_db6.load_state_dict(torch.load('trained_model/WaveFHR-VNet/model_LCU_db6.pt'))

model_sym4 = UNet(1, 3, wavelet='sym4').to(device)
model_sym4.load_state_dict(torch.load('trained_model/WaveFHR-VNet/model_LCU_sym4.pt'))

model_sym5 = UNet(1, 3, wavelet='sym5').to(device)
model_sym5.load_state_dict(torch.load('trained_model/WaveFHR-VNet/model_LCU_sym5.pt'))

# %%
model_bior.eval()
model_db4.eval()
model_db6.eval()
model_sym4.eval()
model_sym5.eval()

total_correct = 0
total_samples = 0
Dice = 0
Iou = 0
# Iterate through the DataLoader
MeasureList = {}  # 存储评估指标的特点
MeasureList['Dice'] = []
MeasureList['Dice_Acc'] = []
MeasureList['Dice_Dec'] = []

MeasureList['Iou'] = []
MeasureList['Iou_Acc'] = []
MeasureList['Iou_Dec'] = []

MeasureList['Accuracy'] = []


for batch in test_dataloader:
    fhrs = batch['fhrs'].to(device)  # 输入（batchsize,1,4800）
    labels = batch['labels'].to(device)
    baselines = batch['baselines'].to(device)
    with torch.no_grad():
        preds_bior = model_bior(fhrs)  # （batchsize,3,4800）
        preds_db4 = model_db4(fhrs)
        preds_db6 = model_db6(fhrs)
        preds_sym4 = model_sym4(fhrs)
        preds_sym5 = model_sym5(fhrs)
    preds = hard_voting([preds_bior,preds_db4,preds_db6,preds_sym4,preds_sym5])
    stats = mstatscompare_P(ground_labels=labels,pred_labels=preds)
    Dice = stats['Dice'] * 100
    Dice_Acc = stats['Dice_Acc'] * 100
    Dice_Dec = stats['Dice_Dec'] * 100

    Iou = stats['Iou'] * 100
    Iou_Acc = stats['Iou_Acc'] * 100
    Iou_Dec = stats['Iou_Dec'] * 100

    Accuracy = stats['Accuracy'] * 100
    MeasureList['Accuracy'] = np.append(MeasureList['Accuracy'], Accuracy)

    MeasureList['Dice'] = np.append(MeasureList['Dice'], Dice)
    MeasureList['Dice_Acc'] = np.append(MeasureList['Dice_Acc'], Dice_Acc)
    MeasureList['Dice_Dec'] = np.append(MeasureList['Dice_Dec'], Dice_Dec)

    MeasureList['Iou'] = np.append(MeasureList['Iou'], Iou)
    MeasureList['Iou_Acc'] = np.append(MeasureList['Iou_Acc'], Iou_Acc)
    MeasureList['Iou_Dec'] = np.append(MeasureList['Iou_Dec'], Iou_Dec)



print('......SUMMARY......')
print('Median Dice_Acc = %0.4f\t' % (np.nanmedian(MeasureList['Dice_Acc'])),
      'Mean Dice_Acc = %0.4f\t' % (np.nanmean(MeasureList['Dice_Acc'])))
print('Median Dice_Dec = %0.4f\t' % (np.nanmedian(MeasureList['Dice_Dec'])),
      'Mean Dice_Dec = %0.4f\t' % (np.nanmean(MeasureList['Dice_Dec']))
      )
print('Median Dice = %0.4f\t' % (np.nanmedian(MeasureList['Dice'])),
      'Mean Dice = %0.4f\t' % (np.nanmean(MeasureList['Dice']))
      )


print('Median Iou_Acc = %0.4f\t' % (np.nanmedian(MeasureList['Iou_Acc'])),
      'Mean Iou_Acc = %0.4f\t' % (np.nanmean(MeasureList['Iou_Acc']))
      )
print('Median Iou_Dec = %0.4f\t' % (np.nanmedian(MeasureList['Iou_Dec'])),
      'Mean Iou_Dec = %0.4f\t' % (np.nanmean(MeasureList['Iou_Dec']))
      )
print('Median Iou = %0.4f\t' % (np.nanmedian(MeasureList['Iou'])),
      'Mean Iou = %0.4f\t' % (np.nanmean(MeasureList['Iou']))
      )
print('Median Accuracy = %0.4f\t' % (np.nanmedian(MeasureList['Accuracy'])),
      'Mean Accuracy = %0.4f\t' % (np.nanmean(MeasureList['Accuracy']))
      )

print('......  END  ......')

