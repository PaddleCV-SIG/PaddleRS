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

import numpy as np
import paddlers.transforms as T
from paddlers.transforms.operators import Transform
from paddlers.transforms.functions import to_intensity
from typing import List, Dict, Union

try:
    from osgeo import gdal
except:
    import gdal


class RasterLoader(object):
    TS = [
        "Normalize", "RandomBlur", "Defogging", "DimReducing", "BandSelecting"
    ]

    def __init__(self,
                 path: str,
                 block_size: Union[List[int], int]=512,
                 overlap: Union[List[int], int]=32,
                 transforms: Union[Transform, None]=None) -> None:
        """ Dataloadr about geo-raster.

        Args:
            path (str): Path of big-geo-image.
            block_size (Union[List[int], int], optional): Size of image's block. Defaults to 512.
            overlap (Union[List[int], int], optional): Overlap between two blocks. Defaults to 32.
            transforms (Transform or None, optional): PaddleRS's transform. Defaults to None.

        Raises:
            ValueError: Can't read iamge from this path. 
        """
        self._src_data = gdal.Open(path)
        if self._src_data is None:
            raise ValueError("Can't read iamge from file {0}.".format(path))
        self.path = path
        if isinstance(block_size, int):
            self.block_size = [block_size, block_size]
        else:
            self.block_size = list(block_size)
        if isinstance(overlap, int):
            self.overlap = [overlap, overlap]
        else:
            self.overlap = list(overlap)
        if len(self.block_size) != 2 or len(self.overlap) != 2:
            raise IndexError(
                "Lenght of `block_size`/`overlap` must be 2, not {0}/{1}.".
                format(len(self.block_size), len(self.overlap)))
        self.transforms = []
        for op in transforms.transforms:
            if op.__class__.__name__ in self.TS:
                self.transforms.append(op)
        if len(self.transforms) >= 1:
            self.transforms = T.Compose(self.transforms)
        else:
            self.transforms = None
        self.transforms.apply_im_only = True
        self.__getInfos()
        self.__getStart()

    def __getitem__(self, index) -> np.ndarray:
        start_loc = self._start_list[index]
        return self.__getBlock(start_loc)

    def __len__(self) -> int:
        return len(self._start_list)

    @property
    def config(self) -> Dict:
        return {
            "file_path": self.path,
            "width": self.width,
            "height": self.height,
            "proj": self.proj,
            "geotf": self.geotf,
            "block_size": self.block_size,
            "overlap": self.overlap,
            "num_block": self.__len__,
        }

    def __getInfos(self) -> None:
        self.bands = self._src_data.RasterCount
        self.width = self._src_data.RasterXSize
        self.height = self._src_data.RasterYSize
        self.geotf = self._src_data.GetGeoTransform()
        self.proj = self._src_data.GetProjection()

    def __getStart(self) -> None:
        self._start_list = []
        step_r = self.block_size[1] - self.overlap[1]
        step_c = self.block_size[0] - self.overlap[0]
        for r in range(0, self.height, step_r):
            for c in range(0, self.width, step_c):
                self._start_list.append([c, r])

    def __preProcessing(self, im: np.ndarray) -> np.ndarray:
        if im.ndim == 2:
            im = to_intensity(im)  # is read SAR
            im = im[:, :, np.newaxis]
        elif im.ndim == 3:
            im = im.transpose((1, 2, 0))
        if self.transforms is not None:
            data = {"image": im}
            im = self.transforms(data)["image"]
        return im

    def __getBlock(self, start_loc: List[int]) -> np.ndarray:
        xoff, yoff = start_loc
        xsize, ysize = self.block_size
        if xoff + xsize > self.width:
            xsize = self.width - xoff
        if yoff + ysize > self.height:
            ysize = self.height - yoff
        im = self._src_data.ReadAsArray(
            int(xoff), int(yoff), int(xsize), int(ysize))
        im = self.__preProcessing(im)
        h, w = im.shape[:2]
        out = np.zeros(
            (self.block_size[1], self.block_size[0], self.bands),
            dtype=im.dtype)
        out[:h, :w, :] = im
        return {"block": out, "start": start_loc}
