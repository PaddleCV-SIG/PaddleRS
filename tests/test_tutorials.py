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
import re
import tempfile
import shutil
from glob import iglob

from testing_utils import run_script, CpuCommonTest


class TestTutorial(CpuCommonTest):
    SUBDIR = "./"
    TIMEOUT = 300
    REGEX = ".*"

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(dir='./')
        # Recursively copy the content of `cls.SUBDIR` to td.
        # This is necessary for running scripts in td.
        cls._TSUBDIR = osp.join(cls._td.name, osp.basename(cls.SUBDIR))
        shutil.copytree(cls.SUBDIR, cls._TSUBDIR)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    @staticmethod
    def add_tests(cls):
        """
        Automatically patch testing functions to cls.
        """

        def _test_tutorial(script_name):
            def _test_tutorial_impl(self):
                # Set working directory to `cls._TSUBDIR` such that the 
                # files generated by the script will be automatically cleaned.
                run_script(f"python {script_name}", wd=cls._TSUBDIR)

            return _test_tutorial_impl

        for script_path in filter(
                re.compile(cls.REGEX).match,
                iglob(osp.join(cls.SUBDIR, '*.py'))):
            script_name = osp.basename(script_path)
            if osp.normpath(osp.join(cls.SUBDIR, script_name)) != osp.normpath(
                    script_path):
                raise ValueError(
                    f"{script_name} should be directly contained in {cls.SUBDIR}"
                )
            setattr(cls, 'test_' + script_name, _test_tutorial(script_name))

        return cls


@TestTutorial.add_tests
class TestCDTutorial(TestTutorial):
    SUBDIR = "../tutorials/train/change_detection"


@TestTutorial.add_tests
class TestClasTutorial(TestTutorial):
    SUBDIR = "../tutorials/train/classification"


@TestTutorial.add_tests
class TestDetTutorial(TestTutorial):
    SUBDIR = "../tutorials/train/object_detection"


@TestTutorial.add_tests
class TestSegTutorial(TestTutorial):
    SUBDIR = "../tutorials/train/semantic_segmentation"
    REGEX = r".*(?<!run_with_clean_log\.py)$"
