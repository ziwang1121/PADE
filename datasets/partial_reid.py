# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import re
import urllib
import zipfile

import os.path as osp

from utils.iotools import mkdir_if_missing
from .bases import BaseImageDataset


class Partial_REID(BaseImageDataset):

    dataset_dir_train = 'market1501'
    dataset_dir_test = 'Partial_REID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Partial_REID, self).__init__()
        self.dataset_dir_train = osp.join(root, self.dataset_dir_train)
        self.dataset_dir_test = osp.join(root, self.dataset_dir_test)

        self.train_dir = osp.join(self.dataset_dir_train, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir_test, 'partial_body_images')
        self.gallery_dir = osp.join(self.dataset_dir_test, 'whole_body_images')
        self.pid_begin = pid_begin
        self._check_before_run()

        train = self._process_dir_train(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, camera_id=1, relabel=False)
        gallery = self._process_dir_test(self.gallery_dir, camera_id=2, relabel=False)

        if verbose:
            print("=> Partial REID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir_train):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir_train))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_dir_test(self, dir_path, camera_id=1, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pid_container = set()
        for img_path in img_paths:
            jpg_name = img_path.split('/')[-1]
            pid = int(jpg_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            jpg_name = img_path.split('/')[-1]
            pid = int(jpg_name.split('_')[0])
            camid = camera_id
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, 1))
        return data
