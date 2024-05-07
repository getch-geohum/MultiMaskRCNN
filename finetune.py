import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from engine import train_one_epoch, evaluate, evaluate_one_epoch
from skimage.io import imread
from shapely.geometry import Polygon
from skimage import measure
import numpy as np
import os
import random
import utils
import argparse
from PIL import Image, ImageDraw

def make_normal(img):
    arr = np.clip(((img - np.amin(img)) + 0.00001) / ((np.amax(img) - np.amin(img)) + 0.00001),0,1)
    return arr

def mask2coco(MASK):
    assert len(MASK.shape)==2, 'shape of the mask should be two dimensional. Please check it'

    contours = measure.find_contours(MASK, 0.5)

    boxes = []
    masks = []
    areas = []

    if len(contours)>=1:
        for cont in contours:
            for i in range(len(cont)):
                row, col = cont[i]
                cont[i] = (col, row)
            if len(cont)<4: # invalid geometries
                continue
            poly = Polygon(cont)
            poly = poly.simplify(1.0, preserve_topology=False)
            if poly.is_empty:
                continue
            if poly.geom_type == 'MultiPolygon':
                for spoly in list(poly.geoms): # fix for shapely 2.11
                    if spoly.is_empty:
                        continue
                    if not spoly.is_valid:
                        continue
                    min_x, min_y, max_x, max_y = spoly.bounds
                    area = spoly.area
                    boxes.append([min_x, min_y, max_x, max_y])
                    areas.append(area)

                    seg = np.array(spoly.exterior.coords)
                    row, col = seg[:,0], seg[:,1]
                    cords = [(row[i], col[i]) for i in range(len(row))]

                    img = Image.new('L', (MASK.shape[0], MASK.shape[0]), 0)
                    ImageDraw.Draw(img).polygon(cords, outline=1, fill=1)
                    mask = np.array(img)
                    masks.append(mask)
            else:
                if not poly.is_valid:
                    continue
                min_x, min_y, max_x, max_y = poly.bounds
                area = poly.area
                boxes.append([min_x, min_y, max_x, max_y])
                areas.append(area)

                seg = np.array(poly.exterior.coords)
                row, col = seg[:,0], seg[:,1]
                cords = [(row[i], col[i]) for i in range(len(row))]

                img = Image.new('L', (MASK.shape[0], MASK.shape[0]), 0)
                ImageDraw.Draw(img).polygon(cords, outline=1, fill=1)
                mask = np.array(img)
                masks.append(mask)
    if len(masks)>=1:
        masks = np.array(masks)
        boxes = np.array(boxes)
        areas = np.array(areas)
        cats = np.ones(masks.shape[0])
        return {'mask':masks, 'box':boxes, 'area':areas, 'cat':cats}
    else:
        masks = np.zeros((0,MASK.shape[0], MASK.shape[1]),int)
        boxes = np.zeros((0,4), float)
        areas = np.zeros((0), float)
        cats = np.zeros((0), float)
        return {'mask':masks, 'box':boxes, 'area':areas, 'cat':cats}

class CampDataset(torch.utils.data.Dataset):
    def __init__(self, images,labels):
        self.imgs = images
        self.masks = labels
        
        print(f'{len(self.imgs)} images and {len(self.masks)} masks obtained')

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = torch.from_numpy(make_normal(imread(img_path)).astype(float))[:256, :256,:3].permute(2,0,1)
        mask = imread(mask_path).astype(float) # )[:256, :256]
        A = mask2coco(mask)

        image_id = idx
        iscrowd = torch.zeros((A['box'].shape[0]), dtype=torch.int64)

        target = {}
        target["boxes"] = torch.from_numpy(A['box'])
        target["masks"] = torch.from_numpy(A['mask']).to(dtype=torch.uint8)
        target["labels"] = torch.from_numpy(A['cat']).to(dtype=torch.int64)
        target["image_id"] = image_id
        target["area"] = torch.from_numpy(A['area'])
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def systematic_split(imgs,split_ratio= 0.7, val_test_split=False, val_ratio=None):
    #imgs = sorted(glob(root_dir + '/images' + '/*.tif'))
    N = len(imgs)
    
    tr = int(N*split_ratio)
    tr_ts = int(N*(1-split_ratio))
    t_ind = list(range(0, N, int(N/tr)))

    tr_im = [imgs[i] for i in t_ind]
    tr_lb = [am.replace('images', 'labels') for am in tr_im]


    val_ts_im = list(set(imgs).symmetric_difference(set(tr_im)))
    val_ts_lb = [am.replace('images', 'labels') for am in val_ts_im] # labels

    print("**")
    print(len(val_ts_im), len(tr_im), tr_ts, tr, N)
    
    if val_test_split:
        if val_ratio is None:
            val = [val_ts_im[i] for i in range(0, len(val_ts_im), 2)]
            ts = [val_ts_im[i] for i in range(1, len(val_ts_im), 2)]
        else:
            v_n = int(val_ratio * len(val_ts_im))
            val = random.sample(val_ts_im, v_n)
            ts = list(set(val_ts_im).symmetric_difference(set(val)))
        
        val_lb = [am.replace('images', 'labels') for am in val] # labels
        ts_lb = [am.replace('images', 'labels') for am in ts] # labels
        
        return (tr_im, tr_lb), (val, val_lb), (ts, ts_lb)
    else:
        return (tr_im, tr_lb), (val_ts_im, val_ts_lb)

def random_split(imgs,split_ratio= 0.7, val_test_split=False, val_ratio=None):
    #imgs = sorted(glob(root_dir + '/images' + '/*.tif'))
    random.seed(0)
    N = len(imgs)

    tr = round(N*split_ratio)
    
    tr_im = random.sample(imgs, tr) 
    tr_lb = [am.replace('images', 'labels') for am in tr_im] # labels

    val_ts_im = list(set(imgs).symmetric_difference(set(tr_im)))
    val_ts_lb = [am.replace('images', 'labels') for am in val_ts_im] # labels


    if val_test_split:
        if val_ratio is None:
            val = [val_ts_im[i] for i in range(0, len(val_ts_im), 2)]
            ts = [val_ts_im[i] for i in range(1, len(val_ts_im), 2)]
        else:
            v_n = int(val_ratio*len(val_ts_im))
            val = random.sample(val_ts_im, v_n)
            ts = list(set(val_ts_im).symmetric_difference(set(val)))

        val_lb = [am.replace('images', 'labels') for am in val] # labels
        ts_lb = [am.replace('images', 'labels') for am in ts] # labels

        return (tr_im, tr_lb), (val, val_lb), (ts, ts_lb)
    else:
        return (tr_im, tr_lb), (val_ts_im, val_ts_lb), None


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone = resnet_fpn_backbone("resnet101", pretrained=True, trainable_layers=3)
    model = MaskRCNN(backbone=backbone, num_classes=2, min_size=args.row, max_size=args.row) # make sure the images in dataloader have equal dimesnsions
    # model = torch.load(f'{args.save_dir}/{args.name}.pt')
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))  # resume training
        print('Model weight loaded from {}'.format(args.weight))
    else:
        raise Warning('Model weight is not provided and training will start from scratch')
    model = model.to(device)
    model.train()
    print('Model loaded properly!...')
    print(f'Model placed on device: {device}')

    # Sampling per folder
    # Samping per entire dataset

    im_files_train = []
    lb_files_train = []

    im_files_valid = []
    lb_files_valid = []

    im_files_test = []
    lb_files_test = []

    print('Started sampling...')
    for fold in ["Minawao_june_2016"]: #os.listdir(args.root):
        # if fold == args.ignore:
        #     print(f'{fold} --> ignored for testing')
        #     pass
        # else:
        imfs = os.listdir(f'{args.root}/{fold}/images')
        lbls = os.listdir(f'{args.root}/{fold}/labels')


        file_names = list(set(imfs).intersection(lbls))

        images = [f'{args.root}/{fold}/images/{im}' for im in file_names]
        labels = [f'{args.root}/{fold}/labels/{im}' for im in file_names]

        assert len(images) == len(
            labels), f'The length of images {len(images)} and the length of labels {len(labels)} is not the same!'
        # labels = glob(f'{args.root}/{fold}/labels/*.tif')

        if args.validate:
            if args.sample_type == 'systematic':
                train_f, valid_f, test_f = systematic_split(images,
                                                            split_ratio=args.train_ratio,
                                                            val_test_split=args.reserve_test_sample,
                                                            val_ratio=args.val_ratio)

                print(len(train_f[0]), len(valid_f[0]), len(test_f[0]))
            else:
                train_f, valid_f, test_f = random_split(images,
                                                        split_ratio=args.train_ratio,
                                                        val_test_split=args.reserve_test_sample,
                                                        val_ratio=args.val_ratio)

            im_files_train.extend(train_f[0])
            lb_files_train.extend(train_f[1])

            im_files_valid.extend(valid_f[0])
            lb_files_valid.extend(valid_f[1])

            if test_f is not None:
                im_files_test.extend(test_f[0])
                lb_files_test.extend(test_f[1])
        else:
            im_files_train.extend(images)
            lb_files_train.extend(labels)

    assert len(im_files_train) == len(lb_files_train), 'images and labels are not the same for train set!'
    if args.validate:
        assert len(im_files_valid) == len(lb_files_valid), 'images and labels are not the same for valid set!'
        if len(im_files_test) >= 1:
            assert len(im_files_test) == len(lb_files_test), 'images and labels are not the same for the test set!'

    print(f'Total number of training samples--> {len(im_files_train)}')
    print(f'Total number of validation samples--> {len(im_files_valid)}')
    if len(im_files_test) >= 1:
        print(f'Total number of test samples--> {len(im_files_test)}')
    else:
        print('test samples are not reserved for later test')

    train_dataset = CampDataset(images=im_files_train, labels=lb_files_train)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn)

    if args.validate:
        valid_dataset = CampDataset(images=im_files_valid, labels=lb_files_valid)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.valid_batch,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # early stoping and validation
    # notes either using loss or accuray to track learning process

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    PATH_m = f'{args.save_dir}/model_{args.name}.pt'
    PATH_w = f'{args.save_dir}/weight_{args.name}.pt'
    PATH_bw = f'{args.save_dir}/model_{args.name}_best_weight.pt'

    best_val_metric = 0
    monitor = 0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        lr_scheduler.step()
        if args.validate:
            if args.val_type == 'accuracy':
                _, val_metric = evaluate(model=model, data_loader=valid_dataloader,
                                         device=device)  # based on validation accuracy
            else:
                _, val_metric = evaluate_one_epoch(model=model, data_loader=valid_dataloader, device=device,
                                                   epoch=epoch, print_freq=10)  # based on validation loss
            # early stoping approach to early terminate training if the model skill is not improving
            if epoch == 1:
                best_val_metric = val_metric
            else:
                if val_metric <= best_val_metric:
                    best_val_metric = val_metric
                    monitor = 0
                    torch.save(model.state_dict(), PATH_bw)
                else:
                    monitor += 1
                    if monitor >= args.petience:
                        torch.save(model, PATH_m)
                        torch.save(model.state_dict(), PATH_w)
                        break
        # evaluate(model, data_loader_test, device=device)
        # evaluate_one_epoch(model, train_dataloader, device, epoch, print_freq=10) # evaluate
    torch.save(model, PATH_m)
    torch.save(model.state_dict(), PATH_w)
    print("That's it!")

def parse_args():
    parser = argparse.ArgumentParser('MaskRCNN')
    parser.add_argument("--save_dir", type=str, default="/MaskRCNN_output/model")
    parser.add_argument('--root', type=str, default="/DATA") # data root
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_batch', type=int, default=12)
    parser.add_argument('--valid_batch', type=int, default=12)
    parser.add_argument('--name', type=str, default='model_first')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--val_type', type=str, choices = ['loss', 'accuracy'], default='accuracy')
    parser.add_argument('--sample_type', type=str, choices=['systematic', 'random'], default='systematic')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--reserve_test_sample', action='store_true')
    parser.add_argument('--petience', type=int, default=10)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--row', type=int, default=256, help='number of pixels in row')
    parser.add_argument('--col', type=int, default=256, help='number of pixels in column')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    from configs import finetune_configs
    for key, value in finetune_configs.items():
        setattr(args, key, value)
    main(args=args)

