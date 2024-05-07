import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window

def point2mask(raster, polygon, points, out_root, width=256, height=256):
    """"
    raster: the raster image to generate training samples
    polygon: selected regular tesselations where samples are taken, could be fishnet
    points: points indicating building centroids. They could be generated either by direct digitization
    or from feature to point
    out_root: directory to save image and mask chips. It automatically adds "images" and "labels" folders
    """
    root_image = '{}/images'.format(out_root)
    root_mask = '{}/labels'.format(out_root)

    if not os.path.exists(root_mask):
        os.makedirs(root_mask, exist_ok=True)

    if not os.path.exists(root_image):
        os.makedirs(root_image, exist_ok=True)

    polys = gpd.read_file(polygon)
    pts = gpd.read_file(points)

    assert polys.crs == pts.crs, 'Polygon and Point features do not have the same coordinate reference system {}, {}'.format(polys.crs, pts.crs)

    ref = rasterio.open(raster)

    for i, geom in enumerate(polys.geometry):
        lon_min, lat_min, lon_max, lat_max = geom.bounds
        row, col = rasterio.transform.rowcol(ref.profile['transform'], lon_min, lat_max)

        win = Window(col_off=col, row_off=row, width=width, height=height)
        transform = rasterio.windows.transform(win, ref.profile['transform'])
        image = ref.read(window=win)  # [:3, :, :]

        sel_pts = [pts.geometry[j] for j in range(len(pts.geometry)) if pts.geometry[j].within(geom)]
        pxs = [cc.x for cc in sel_pts]
        pys = [cc.y for cc in sel_pts]

        # https://rasterio.readthedocs.io/en/stable/api/rasterio.transform.html

        row_, col_ = rasterio.transform.rowcol(transform, pxs, pys)
        mask = np.zeros((height, width), np.uint8)
        mask[row_, col_] = 1
        print('sum of points', np.sum(mask), mask.shape)

        profile_array = ref.profile
        profile_mask = ref.profile

        profile_array.update(
            transform=transform,
            width=width,
            height=height)

        profile_mask.update(
            transform=transform,
            count=1,
            width=width,
            height=height,
            dtype=np.uint8)

        with rasterio.open('{}/image_{}.tif'.format(root_image, i), 'w', **profile_array) as dst:
            dst.write(image)

        with rasterio.open('{}/mask_{}.tif'.format(root_mask, i), 'w', **profile_mask) as dst_:
            dst_.write(mask.astype(np.uint8), 1)

if __name__ == '__main__':
    raster = './Building_data_preparation/L15-1147E-1100N.tif'
    polygon = './Building_data_preparation/selected_new.shp'
    points = './Building_data_preparation/fish_point.shp'
    out_root = './Building_data_preparation'
    point2mask(raster, polygon, points, out_root, width=256, height=256)

