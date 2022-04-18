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
from typing import List, Dict
from .base import BaseWriter

try:
    from osgeo import gdal
except:
    import gdal


class SegWriter(BaseWriter):
    def __init__(self, config: Dict) -> None:
        super(SegWriter, self).__init__(config, "tif")
        driver = gdal.GetDriverByName("GTiff")
        self.dst_ds = driver.Create(self.save_path, self.width, self.height, 1,
                                    gdal.GDT_UInt16)
        self.dst_ds.SetGeoTransform(self.geotf)
        self.dst_ds.SetProjection(self.proj)
        self.band = self.dst_ds.GetRasterBand(1)
        self.band.WriteArray(255 * np.ones(
            (self.height, self.width), dtype="uint8"))

    def write(self, block: np.ndarray, start: List[int]) -> None:
        bw, bh = self.block_size
        xoff, yoff = start
        xsize = xoff + bw
        ysize = yoff + bh
        xsize = int(self.width - xoff) if xsize > self.width else int(bw)
        ysize = int(self.height - yoff) if ysize > self.height else int(bh)
        rd_block = self.band.ReadAsArray(int(xoff), int(yoff), xsize, ysize)
        h, w = rd_block.shape
        mask = (rd_block == block[:h, :w]) | (rd_block == 255)
        temp = block[:h, :w].copy()
        temp[mask == False] = 0
        self.band.WriteArray(temp, int(xoff), int(yoff))
        self.dst_ds.FlushCache()

    def close(self) -> None:
        self.dst_ds = None
