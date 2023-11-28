import torch
import utils
import Utils.transforms as T



from torch.utils.data import Dataset

import CardFieldsDetection
import Mask_RCNN
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 6+1
# use our dataset and defined transformations
dataset = CardFieldsDetection('Dataset/train/CNIC_UK', 'Dataset_Annotations',
                                is_Train=True, transform=T.ToTensor())
dataset_test = CardFieldsDetection('Dataset/val/CNIC_UK', 'Dataset_Annotations',
                                is_Train=False, transform=T.ToTensor())

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=8, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = Mask_RCNN.get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs = 10
