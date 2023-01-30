# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""""""
import pickle
from pathlib import Path
import lmdb


class LMDBDataset:

    def __init__(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )
        self.path = path
        self.env = self._connect_db(path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def __getitem__(self, idx):
        pickled_data = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(pickled_data)
        return data
    
    def __len__(self):
        return len(self._keys)

    def _connect_db(self, path):
        env = lmdb.open(
            path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256
        )
        return env
    