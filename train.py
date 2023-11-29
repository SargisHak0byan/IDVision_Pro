import torch
import Utils.utils
import Utils.transforms as T
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from torch.utils.data import Dataset
import CardFieldsDetection
import Mask_RCNN
from Utils.engine import train_one_epoch, evaluate
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 6+1
# use our dataset and defined transformations
dataset = CardFieldsDetection.CardFieldsDetection('Dataset/train/CNIC_UK', 'Dataset_Annotations',
                                is_Train=True, transform=T.ToTensor())
dataset_test = CardFieldsDetection.CardFieldsDetection('Dataset/val/CNIC_UK', 'Dataset_Annotations',
                                is_Train=False, transform=T.ToTensor())

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=Utils.utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=8, shuffle=False, num_workers=4,
    collate_fn=Utils.utils.collate_fn)

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

for epoch in range(num_epochs):
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    coco_logger = evaluate(model, data_loader_test, device=device)

    # training metrics
    # train_loss = metric_logger.meters['loss'].avg
    # loss_classifier = metric_logger.meters['loss_classifier'].avg
    # loss_box_reg = metric_logger.meters['loss_box_reg'].avg
    # loss_mask = metric_logger.meters['loss_mask'].avg

    # evaluation metrics
    bbox_vals = coco_logger.coco_eval['bbox']
    bbox_precision = bbox_vals.stats[:3]
    segm_vals = coco_logger.coco_eval['segm']
    segm_mAP = segm_vals.stats[:3]

    # tensorboard bookeeping
    # writer.add_scalar("training loss", train_loss, epoch)
    # writer.add_scalar("training classifier loss", loss_classifier, epoch)
    # writer.add_scalar("training bbox regression loss", loss_box_reg, epoch)
    # writer.add_scalar("training segmentation mask loss", loss_mask, epoch)

    writer.add_scalar("bbox mAP IoU @ 0.5:0.95", bbox_precision[0], epoch)
    writer.add_scalar("bbox mAP IoU @ 0.5", bbox_precision[1], epoch)
    writer.add_scalar("bbox mAP IoU @ 0.75", bbox_precision[2], epoch)

    writer.add_scalar("segm mAP IoU @ 0.5:0.95", segm_mAP[0], epoch)
    writer.add_scalar("segm mAP IoU @ 0.5", segm_mAP[1], epoch)
    writer.add_scalar("segm mAP IoU @ 0.75", segm_mAP[2], epoch)

writer.flush()