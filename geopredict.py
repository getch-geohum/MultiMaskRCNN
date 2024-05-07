import numpy as np
from shapely.geometry import Polygon
from skimage import measure
import rasterio as rio
from rasterio.windows import Window
import torch
import argparse
from glob import glob
import os
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from torchvision.ops import nms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN


def pad_array(x, out_r=256, out_c=256, mode='reflect'):
    shp = x.shape
    r, c = shp[1], shp[2]
    r_d = out_r-r
    c_d = out_c-c

    assert r_d+r == out_r, 'row delta and row are not the same'
    assert c_d+c == out_c, 'col delta and column are not the same'

    x_out = np.pad(x, ((0, 0), (0, r_d), (c_d)), mode=mode)
    return x_out

def getObjectSingle(masks, trsfm, scor, mask_treshold):
    polygons = []
    score_sel = []
    cc = 0
    for i in range(masks.shape[0]):
        mask = masks[i]
        mask[mask<mask_treshold] = 0
        mask[mask>=mask_treshold] = 1
        contours = measure.find_contours(mask, 0.5)
        try:
            poly = Polygon(contours[0])
            cords = np.array(poly.exterior.coords)
            row = cords[:,0]
            col = cords[:,1]
            lons, lats = rio.transform.xy(trsfm, row, col) # trsfm is from raster
            new_poly = Polygon([[lons[j], lats[j]] for j in range(len(lons))])
            polygons.append(new_poly)
            score_sel.append(scor[i])
            cc+=1
        except:
            pass
    if cc>=1:
        s = gpd.GeoSeries(polygons)
        return s, score_sel
    else:
        return None, None

def make_normal(img):
    arr = np.clip(((img - np.amin(img)) + 0.00001) / ((np.amax(img) - np.amin(img)) + 0.00001),0,1)
    return arr

def IOU(P, R):
    intersection = np.sum(P*R)
    union = np.sum(P+R)-intersection
    iou_val = intersection/union
    return iou_val

def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

def remove_overlaping_polygons(masks, scores, mask_iou_treshold):
    other_masks = []
    other_scores = []

    for i, mask in enumerate(masks):
        print('Cleaning polygons-->:', i)
        if i == 0:
            other_masks.append(mask)
            other_scores.append(scores[i])
        else:
            for j, mask_ in enumerate(other_masks):
                iou = IOU(mask_, mask)
                if iou>=mask_iou_treshold:
                    if other_scores[j]>=scores[i]:
                        pass
                    else:
                        other_masks[j] = mask
                        other_scores[j] = scores[i]
                else:
                    other_masks.append(mask)
                    other_scores.append(scores[i])
    return other_masks, other_scores


def predictSingle(model,rst,score_treshold,mask_iou_treshold, bbox_iou_treshold, return_prob):
    out = model(rst)
    masks = out[0]['masks'].squeeze() # .detach().cpu().numpy()
    scors = out[0]['scores']
    boxes = out[0]['boxes']
    
    ind = out[0]['scores']>=score_treshold
    if len(masks.shape)<=2:
        return None, None
    else:
        masks = masks[ind] # change this
        scors = scors[ind] # change this
        boxes = boxes[ind]
        idxs = nms(boxes=boxes, scores=scors, iou_threshold=bbox_iou_treshold)
        masks = masks[idxs] # change this
        scors = scors[idxs] # change this

        #masks_num = masks.detach().cpu().numpy()
        #scors_num = scors.detach().cpu().numpy()
        #mask, score = remove_overlaping_polygons(masks=masks_num, scores=scors_num, mask_iou_treshold=mask_iou_treshold) 
    
        if return_prob:
            #return mask, score
            return masks.detach().cpu().numpy(), scors.detach().cpu().numpy() 
        else:
            #return mask, None
            return masks.detach().cpu().numpy(), None
        
def geoPredict_chips(model, files, save_root, name, device, mask_treshold, score_treshold,mask_iou_treshold, bbox_iou_treshold, with_prob):
    file_name = f'{save_root}/{name}.shp'
    epsg = None
    alls = []
    probs = []
    for i, file in tqdm(enumerate(files)):
        with rio.open(file) as oss:
            prf = oss.profile
            rst = oss.read()[:3,:,:]
            if rst.sum()>=1:
                if i == 0:
                    epsg = int(prf['crs'].to_dict()['init'].split(':')[1])
                rst = make_normal(rst)
                rst = [torch.from_numpy(rst).float().to(device)]
                masks, prob = predictSingle(model,
                                            rst,
                                            score_treshold=score_treshold,
                                            mask_iou_treshold=mask_iou_treshold,
                                            bbox_iou_treshold=bbox_iou_treshold,
                                            return_prob=with_prob)
                if masks is None:
                    pass
                else:
                    series, scors = getObjectSingle(masks, trsfm=prf['transform'], scor=prob, mask_treshold=mask_treshold)
                    if series is not None:
                        alls.append(series)
                        probs.extend(scors)
    alls = pd.concat(alls)
    assert len(alls) == len(probs), f'Length of dataframe {len(alls)} and object probabilities {len(probs)}'

    vals = {'ID':[f'{i}' for i in range(len(alls.geometry))], 'geometry':alls.geometry, 'scores':probs}
    dfs = gpd.GeoDataFrame(vals).set_crs(epsg=epsg)
    dfs.to_file(file_name)


def geoPredict_scene(model, file, save_root, name, device, mask_treshold, score_treshold,mask_iou_treshold, bbox_iou_treshold, with_prob,row=None, col=None):
    file_name = f'{save_root}/{name}.shp'
    epsg = None
    alls = []
    probs = []
    
    with rio.open(file) as src:
        prf = src.profile
        width = prf['width']  # column
        height = prf['height'] # row
        for i in range(0, height, row):
            for j in range(0, width, col):
                win = Window(col_off=j, row_off=i, width=col, height=row)
                transform = rio.windows.transform(win, src.transform)
                rst = src.read(window=win)[:3,:,:]                                  
                if rst.sum()>=1:
                    if i == 0:
                        epsg = int(prf['crs'].to_dict()['init'].split(':')[1])
                rst = make_normal(rst)
                shape_ = rst.shape

                if (rst.shape[1] != row) or (rst.shape[2] != col): # to treat boundary pixels
                    rst = pad_array(x=rst, out_r=row, out_c=col, mode='reflect')
                    
                rst = [torch.from_numpy(rst).float().to(device)]
                masks, prob = predictSingle(model,
                                            rst,
                                            score_treshold=score_treshold,
                                            mask_iou_treshold=mask_iou_treshold,
                                            bbox_iou_treshold=bbox_iou_treshold,
                                            return_prob=with_prob)
                if masks is None:
                    pass
                else:
                    masks = masks[:,shape_[1], shape_[2]]  # make sure dimensionality of predicted shape
                    series, scors = getObjectSingle(masks, trsfm=transform, scor=prob, mask_treshold=mask_treshold)
                    if series is not None:
                        alls.append(series)
                        probs.extend(scors)
    alls = pd.concat(alls)
    assert len(alls) == len(probs), f'Length of dataframe {len(alls)} and object probabilities {len(probs)}'

    vals = {'ID':[f'{i}' for i in range(len(alls.geometry))], 'geometry':alls.geometry, 'scores':probs}
    dfs = gpd.GeoDataFrame(vals).set_crs(epsg=epsg)
    dfs.to_file(file_name)

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = torch.device('cuda') # if torch.cuda.is_available() else torch.device('cpu')
    print("Using Cuda device: ", torch.cuda.is_available())

    if args.weight is not None:
        backbone = resnet_fpn_backbone("resnet101", weights=None, trainable_layers=3)
        model = MaskRCNN(backbone=backbone, num_classes=2, min_size=args.row, max_size=args.row)
        weight = torch.load(args.weight)
        model.load_state_dict(weight, strict=False)
        model.to(device)
        model.eval()
        print('Loaded weights from {}'.format(args.weight))
    else:
        model = torch.load(args.model).eval()
        model.to(device)

    if args.image_type == 'chips':
        files = glob(f'{args.data_dir}/images/*.tif')
        print(f'Got a total of {len(files)} image chips!')

        if len(files) <= 0:
            raise ValueError(f'There is no file in folder {args.data_dir} or the file is not GeoTIF file with .tif extension')

        geoPredict_chips(model=model,
                         files=files,
                         save_root=args.save_dir,
                         name=args.name,
                         mask_treshold=args.mask_treshold,
                         score_treshold=args.score_treshold,
                         mask_iou_treshold=args.mask_iou_treshold,
                         bbox_iou_treshold=args.bbox_iou_treshold,
                         device=device,
                         with_prob=True)

    elif args.image_type == 'full_scene':
        file = glob(f'{args.data_dir}/*.tif')
        if len(file) > 1:
            raise Warning('The folder contains more than one full-scene image, Only the first image will be predicted')
        if len(file) <= 0:
            raise ValueError('There is no file in folder {} or the file is not GeoTIF file with .tif extension')
        geoPredict_scene(model=model,
                         file=file[0],
                         save_root=args.save_dir,
                         name=args.name,
                         mask_treshold=args.mask_treshold,
                         score_treshold=args.score_treshold,
                         mask_iou_treshold=args.mask_iou_treshold,
                         bbox_iou_treshold=args.bbox_iou_treshold,
                         device=device,
                         with_prob=True,
                         row=args.row,
                         col=args.col)
    else:
        raise ValueError('The provided {} is not valid chip type'.format(args.image_type))

    
def parse_args():
    parser = argparse.ArgumentParser('MaskRCNN prediction based on multidataset image training and provides prediction as ESRI shapefile')
    parser.add_argument("--save_dir", type=str, default="/MaskRCNN_output/preds", help='Directory to save outputs')
    parser.add_argument('--data_dir', type=str, default="/DATA/SPOT06", help='directory where data is available')
    parser.add_argument('--weight', type=str,help='saved model weight')
    parser.add_argument('--model', type=str, default='/MaskRCNN_output/model/model_model_first.pt', help='saved_model')
    parser.add_argument('--name', type=str, default='spot_pred_s06_full_ep50_mask05_maskiou08_bbox_nms08_bbx03')
    parser.add_argument('--mask_treshold', type=float, default=0.5, help='cut of value to convert probability to hard binary class')
    parser.add_argument('--score_treshold', type=float, default=0.6, help='Cut of point for objectness confidence of predicted objetcs')
    parser.add_argument('--mask_iou_treshold', type=float, default=0.8, help='Cut of point to discard duplicate predictions based on mask overlap')
    parser.add_argument('--bbox_iou_treshold', type=float, default=0.3, help='Cut of point to discard duplicate predictions basd on object bounding box overlap')
    parser.add_argument('--image_type', type=str, choices=['chips', 'full_scene'], default='chips')
    parser.add_argument('--row', type=int, default=256, help='number of pixels in row')
    parser.add_argument('--col', type=int, default=256, help='number of pixels in column')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    from configs import predict_configs
    for key, value in predict_configs.items():
        setattr(args, key, value)
    print('Parameters copied')

    main(args=args)
    
    
    
