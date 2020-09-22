"""largescale_lung_ct dataset."""

import tensorflow_datasets.public_api as tfds
import pydicom
import tensorflow as tf
import os
import xml.etree.ElementTree as ET

# TODO(largescale_lung_ct): BibTeX citation
_CITATION = """
"""

# TODO(largescale_lung_ct):
_DESCRIPTION = """
"""


class LargescaleLungCt(tfds.core.GeneratorBasedBuilder):
  """TODO(largescale_lung_ct): Short description of my dataset."""

  # TODO(largescale_lung_ct): Set up version.
  VERSION = tfds.core.Version('0.1.0')
  BUILDER_CONFIGS = [
          tfds.core.BuilderConfig(
              version=VERSION,
              name="3d",
              description="3D CT with bounding boxes assigned to corresponding slices."
          ),
          tfds.core.BuilderConfig(
              version=VERSION,
              name="2d",
              description="2D CT slice with bounding boxes."
          )
      ]
  def _info(self):
    if self.builder_config.name is '3d':
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION['3d'],
            features=tfds.features.FeaturesDict({
                'cube': tfds.features.Sequence(tfds.features.Tensor(shape=(512,512),dtype=tf.uint16)),
                'bbox': tfds.features.Sequence(tfds.features.Tensor(shape=(5,1),dtype=tf.uint16),None)
            }),
            supervised_keys=('3d/image', '3d/bbox'),
            homepage=_HOMEPAGE,
            citation=_CITATION
        )
    if self.builder_config.name is '2d':
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION['2d'],
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Tensor(shape=(512,512,None),dtype=tf.uint16),
                'bbox': tfds.features.Tensor(shape=(4,1),dtype=tf.uint16)
            }),
            supervised_keys=('2d/image', '2d/bbox'),
            homepage=_HOMEPAGE,
            citation=_CITATION
        )
  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(largescale_lung_ct): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    #manual_dir = 'home/Jupyter/large_lung_ct'
    data_dir = dl_manager.manual_dir 
    ## need to hardcode location of data to my cloud bucket
    ## currently hardcoded to directory with 2 patients
    # data_dir = dl_manager.manual_dir # manual_dir = gs://bme590/erik/patients 
    # during testing data_dir = /tensorflow_datasets/testing/fake_data/image_detection/
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "images": os.path.join(data_dir,'Patients'), ## test and actual data set need to replace 'Patients' with Lung-PET-CT-Dx
                "bbox": os.path.join(data_dir,'Annotations') ## test and actual data set need to replace 'Annotations' with Lung-PET-CT-Dx-Annotations-XML-Files-rev07142020
            },
        ),
    ]##################
    
  def _generate_examples(self,images,bbox):
    """Yields examples."""
    # TODO(largescale_lung_ct): Yields (key, example) tuples from the dataset
    PATH = '/home/jupyter/large_lung_ct/Annotations' ## list of all annotation files
    annotationFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.xml']
    DIRECTORY_OF_ALL_DICOMS = '/home/jupyter/large_lung_ct/Patients' # len = 260,826 
    patients = tf.io.gfile.listdir(DIRECTORY_OF_ALL_DICOMS) ## list of patients as their dicom folder path
    root = 'large_lung_ct/Patients'
    for patient in tf.io.gfile.listdir(root):
        if patient[0] is '.': continue
        for study in tf.io.gfile.listdir('{}/{}'.format(root, patient)):
            if study[0] is '.': continue
            for series in tf.io.gfile.listdir('{}/{}/{}'.format(root, patient, study)):
                if series[0] is '.': continue
                cube = []
                for image in tf.io.gfile.listdir('{}/{}/{}/{}'.format(root, patient, study, series)):
                    if image[0] is '.': continue 
                    imagedcm = pydicom.dcmread('{}/{}/{}/{}/{}'.format(root, patient, study, series,image)).pixel_array
                    cube.append(imagedcm)
                    uid = pydicom.dcmread('{}/{}/{}/{}/{}'.format(root, patient, study, series, image)).SOPInstanceUID
                    if tf.io.gfile.exists('{}/{}/{}.xml'.format('large_lung_ct/Annotations',patient,uid)):
                        boxes = read_content('{}/{}/{}.xml'.format('large_lung_ct/Annotations',patient,uid))
                        annotation_for_scan_typeCube.append((i, boxes))
                        if self.builder_config.name is '2d':
                            yield '{}/{}/{}/{}/{}'.format(root,patient,study,series,image),{
                                'image': imagedcm,
                                'bbox': boxes
                            }
                    if self.builder_config.name is '3d':
                        yield '{}/{}/{}/{}'.format(root,patient,study,series), {
                            'cube': cube,
                            'bbox': patient_scan_annotations
                        }
    
  def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    list_with_all_boxes = []
    for boxes in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    return list_with_all_boxes