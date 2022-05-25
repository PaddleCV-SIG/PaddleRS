# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path as osp
from typing import List, Tuple, Union

import numpy as np

from paddlers.transforms.functions import to_uint8 as raster2uint8

try:
    from osgeo import gdal
except:
    import gdal


class Raster:
    def __init__(self,
                 path: str,
                 band_list: Union[List[int], Tuple[int], None]=None,
                 to_uint8: bool=False) -> None:
        """ Class of read raster.

        Args:
            path (str): The path of raster.
            band_list (Union[List[int], Tuple[int], None], optional): 
                band list (start with 1) or None (all of bands). Defaults to None.
            to_uint8 (bool, optional): 
                Convert uint8 or return raw data. Defaults to False.
        """
        super(Raster, self).__init__()
        if osp.exists(path):
            self.path = path
            self.ext_type = path.split(".")[-1]
            if self.ext_type.lower() in ["npy", "npz"]:
                self._src_data = None
            else:
                try:
                    # raster format support in GDAL: 
                    # https://www.osgeo.cn/gdal/drivers/raster/index.html
                    self._src_data = gdal.Open(path)
                except:
                    raise TypeError("Unsupported data format: `{}`".format(
                        self.ext_type))
            self.to_uint8 = to_uint8
            self.setBands(band_list)
            self._getInfo()
        else:
            raise ValueError("The path {0} not exists.".format(path))

    def setBands(self, band_list: Union[List[int], Tuple[int], None]) -> None:
        """ Set band of data.

        Args:
            band_list (Union[List[int], Tuple[int], None]): 
                band list (start with 1) or None (all of bands).
        """
        self.bands = self._src_data.RasterCount
        if band_list is not None:
            if len(band_list) > self.bands:
                raise ValueError(
                    "The lenght of band_list must be less than {0}.".format(
                        str(self.bands)))
            if max(band_list) > self.bands or min(band_list) < 1:
                raise ValueError("The range of band_list must within [1, {0}].".
                                 format(str(self.bands)))
        self.band_list = band_list

    def getArray(
            self,
            start_loc: Union[List[int], Tuple[int], None]=None,
            block_size: Union[List[int], Tuple[int]]=[512, 512]) -> np.ndarray:
        """ Get ndarray data 

        Args:
            start_loc (Union[List[int], Tuple[int], None], optional): 
                Coordinates of the upper left corner of the block, if None means return full image.
            block_size (Union[List[int], Tuple[int]], optional): 
                Block size. Defaults to [512, 512].

        Returns:
            np.ndarray: data's ndarray.
        """
        if self._src_data is not None:
            if start_loc is None:
                return self._getArray()
            else:
                return self._getBlock(start_loc, block_size)
        else:
            print("Numpy doesn't support blocking temporarily.")
            return self._getNumpy()

    def _getInfo(self) -> None:
        if self._src_data is not None:
            self.width = self._src_data.RasterXSize
            self.height = self._src_data.RasterYSize
            self.geot = self._src_data.GetGeoTransform()
            self.proj = self._src_data.GetProjection()
            d_name = self._getBlock([0, 0], [1, 1]).dtype.name
        else:
            d_img = self._getNumpy()
            d_shape = d_img.shape
            d_name = d_img.dtype.name
            if len(d_shape) == 3:
                self.height, self.width, self.bands = d_shape
            else:
                self.height, self.width = d_shape
                self.bands = 1
            self.geot = None
            self.proj = None
        if "int8" in d_name:
            self.datatype = gdal.GDT_Byte
        elif "int16" in d_name:
            self.datatype = gdal.GDT_UInt16
        else:
            self.datatype = gdal.GDT_Float32

    def _getNumpy(self):
        ima = np.load(self.path)
        if self.band_list is not None:
            band_array = []
            for b in self.band_list:
                band_i = ima[:, :, b - 1]
                band_array.append(band_i)
            ima = np.stack(band_array, axis=0)
        return ima

    def _getArray(
            self,
            window: Union[None, List[int], Tuple[int]]=None) -> np.ndarray:
        if window is not None:
            xoff, yoff, xsize, ysize = window
        if self.band_list is None:
            if window is None:
                ima = self._src_data.ReadAsArray()
            else:
                ima = self._src_data.ReadAsArray(xoff, yoff, xsize, ysize)
        else:
            band_array = []
            for b in self.band_list:
                if window is None:
                    band_i = self._src_data.GetRasterBand(b).ReadAsArray()
                else:
                    band_i = self._src_data.GetRasterBand(b).ReadAsArray(
                        xoff, yoff, xsize, ysize)
                band_array.append(band_i)
            ima = np.stack(band_array, axis=0)
        if self.bands == 1:
            if len(ima.shape) == 3:
                ima = ima.squeeze(0)
            # the type is complex means this is a SAR data
            if isinstance(type(ima[0, 0]), complex):
                ima = abs(ima)
        else:
            ima = ima.transpose((1, 2, 0))
        if self.to_uint8 is True:
            ima = raster2uint8(ima)
        return ima

    def _getBlock(
            self,
            start_loc: Union[List[int], Tuple[int]],
            block_size: Union[List[int], Tuple[int]]=[512, 512]) -> np.ndarray:
        if len(start_loc) != 2 or len(block_size) != 2:
            raise ValueError("The length start_loc/block_size must be 2.")
        xoff, yoff = start_loc
        xsize, ysize = block_size
        if (xoff < 0 or xoff > self.width) or (yoff < 0 or yoff > self.height):
            raise ValueError("start_loc must be within [0-{0}, 0-{1}].".format(
                str(self.width), str(self.height)))
        if xoff + xsize > self.width:
            xsize = self.width - xoff
        if yoff + ysize > self.height:
            ysize = self.height - yoff
        ima = self._getArray([int(xoff), int(yoff), int(xsize), int(ysize)])
        h, w = ima.shape[:2] if len(ima.shape) == 3 else ima.shape
        if self.bands != 1:
            tmp = np.zeros(
                (block_size[0], block_size[1], self.bands), dtype=ima.dtype)
            tmp[:h, :w, :] = ima
        else:
            tmp = np.zeros((block_size[0], block_size[1]), dtype=ima.dtype)
            tmp[:h, :w] = ima
        return tmp


def save_mask_geotiff(mask: np.ndarray, save_path: str, proj: str, geotf: Tuple) -> None:
    height, width = mask.shape
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(save_path, width, height, 1, gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(geotf)
    dst_ds.SetProjection(proj)
    band = dst_ds.GetRasterBand(1)
    band.WriteArray(mask)
    dst_ds.FlushCache()
    dst_ds = None
