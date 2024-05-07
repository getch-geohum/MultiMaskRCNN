import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box # to define polygon object to extract
from rasterio import features # Functions for working with features in a raster dataset

def polygon2mask(raster, boundary, samples, out_root, column=None, width=256, height=256):
    """"
    raster: the raster image to generate training samples
    bounds: selected regular tesselations where samples are taken, could be fishnet
    samples: polygons indicating building footprints. They could be generated either by direct digitization
    or from existing database
    out_root: directory to save image and mask chips. It automatically adds "images" and "labels" folders
    column: The column that contains values to write into the new raster
    """
    root_image = '{}/images'.format(out_root)
    root_mask = '{}/labels'.format(out_root)

    if not os.path.exists(root_mask):
        os.makedirs(root_mask, exist_ok=True)

    if not os.path.exists(root_image):
        os.makedirs(root_image, exist_ok=True)

    bounds_ = gpd.read_file(boundary)
    polys = gpd.read_file(samples)

    assert bounds_.crs == polys.crs, 'Polygon and Point features do not have the same \
    coordinate reference system {}, {}'.format(bounds_.crs, polys.crs)

    ref = rasterio.open(raster)

    for i, geom in enumerate(bounds_.geometry):
        lon_min, lat_min, lon_max, lat_max = geom.bounds
        row, col = rasterio.transform.rowcol(ref.profile['transform'], lon_min, lat_max)

        win = Window(col_off=col, row_off=row, width=256, height=256)
        transform = rasterio.windows.transform(win, ref.profile['transform'])
        image = ref.read(window=win)  # [:3, :, :]

        aoi = box(lon_min, lat_min, lon_max, lat_max)  # geometry to clip samples within fishnet boundary
        sel_polys = gpd.clip(polys, aoi)
        if column is None:
            shapes = ((geom, value) for geom, value in zip(sel_polys.geometry, [1]*len(sel_polys)))
            if i == 0:
                print('Specfic column is not selected to burn. Value of 1 will be written to all valid polygon pixel locations')
        else:
            shapes = ((geom, value) for geom, value in zip(sel_polys.geometry, sel_polys[column]))
            if i == 0:
                print('Values from column {} will be written to raster'.format(column))

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
            # print(dir(dst_))
            out_array = np.zeros((height, width), np.uint8)  # dst_.read(1)
            mask = features.rasterize(shapes=shapes, fill=0, out=out_array, transform=dst_.transform)
            dst_.write_band(1, mask)

if __name__ == '__main__':
    raster = './Building_data_preparation/L15-1147E-1100N.tif'
    boundary = '/Building_data_preparation/selected_new.shp'
    samples = '/Building_data_preparation/samples_poly.shp'
    save_dir ='/Building_data_preparation/CHIPS'

    polygon2mask(raster=raster,
                 boundary=boundary,
                 samples=samples,
                 out_root=save_dir,
                 column='code',
                 width=256,
                 height=256)



