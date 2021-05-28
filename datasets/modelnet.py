# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Mapping, Generator, Tuple

import dataclasses
import enum
import pathlib
import trimesh
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
The goal of the Princeton ModelNet project is to provide researchers in computer vision, computer graphics, robotics and cognitive science, with a comprehensive clean collection of 3D CAD models for objects. To build the core of the dataset, we compiled a list of the most common object categories in the world, using the statistics obtained from the SUN database. Once we established a vocabulary for objects, we collected 3D CAD models belonging to each object category using online search engines by querying for each object category term. Then, we hired human workers on Amazon Mechanical Turk to manually decide whether each CAD model belongs to the specified cateogries, using our in-house designed tool with quality control. To obtain a very clean dataset, we choose 10 popular object categories, and manually deleted the models that did not belong to these categories. Furthermore, we manually aligned the orientation of the CAD models for this 10-class subset as well. We provide both the 10-class subset and the full dataset for download.
"""

_CITATION = """\
@misc{wu20153d,
      title={3D ShapeNets: A Deep Representation for Volumetric Shapes}, 
      author={Zhirong Wu and Shuran Song and Aditya Khosla and Fisher Yu and Linguang Zhang and Xiaoou Tang and Jianxiong Xiao},
      year={2015},
      eprint={1406.5670},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""


class ModelNetCategoriesNum(enum.IntEnum):
    """ModelNet has 3 version, with 10 categories,
    with 40 and full. 10 and 40 has easy access
    directly from homepage.
    """
    SMALL = 10
    MEDIUM = 40


@dataclasses.dataclass
class ModelNetConfig(tfds.core.BuilderConfig):
    """Builder config for ModelNet"""
    sample_points_num: int = 2048
    categories_num: ModelNetCategoriesNum = ModelNetCategoriesNum.SMALL
    url: str = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    classes_name: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.categories_num == ModelNetCategoriesNum.SMALL:
            self.url = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
            self.classes_name = [
                "bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"
            ]

        elif self.categories_num == ModelNetCategoriesNum.MEDIUM:
            self.url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
            self.classes_name = [
                "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone", "cup",
                "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
                "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink", "sofa",
                "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
            ]


class ModelNet(tfds.core.GeneratorBasedBuilder):
    """Family of ModelNet datasets"""
    BUILDER_CONFIGS = [
        # pytype: disable=wrong-keyword-args
        ModelNetConfig(
            categories_num=ModelNetCategoriesNum.SMALL,
            name="10",
            description="ModelNet with 10 categories",
        ),
        ModelNetConfig(
            categories_num=ModelNetCategoriesNum.MEDIUM,
            name="40",
            description="ModelNet with 40 categories",
        ),
        # pytype: enable=wrong-keyword-args
    ]

    VERSION = tfds.core.Version('1.0.0')

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "point_cloud": tfds.features.Tensor(shape=(self.builder_config.sample_points_num, 3), dtype=tf.float32),
                "label": tfds.features.ClassLabel(names=self.builder_config.classes_name)
            }),
            homepage='https://modelnet.cs.princeton.edu',
            description=_DESCRIPTION,
            citation=_CITATION)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager) -> Mapping[str, object]:
        extracted_path = dl_manager.download_and_extract(tfds.download.Resource(url=self.builder_config.url))
        dataset_path = extracted_path / f"ModelNet{self.builder_config.name}"
        return {
            'train': self._generate_examples(extracted_path=dataset_path, split='train'),
            'test': self._generate_examples(extracted_path=dataset_path, split='test'),
        }

    def _generate_examples(self, extracted_path: pathlib.Path,
                           split: str) -> Generator[Tuple[str, Mapping[str, object]], None, None]:
        for class_name in self.builder_config.classes_name:
            split_objects_of_class = (extracted_path / class_name / split)
            if split_objects_of_class.exists():
                for off in split_objects_of_class.iterdir():
                    key = off.stem
                    example = {
                        "point_cloud":
                            np.array(trimesh.load(str(off)).sample(self.builder_config.sample_points_num), np.float32),
                        "label":
                            class_name
                    }
                    yield key, example
            else:
                warnings.warn(f"{self.builder_config.name} doesn't contain {split} data for {class_name}")
