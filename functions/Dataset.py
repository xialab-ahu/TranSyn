"""
@description: encapsulate torch or lmdb data
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""


import io
import torch
import lmdb
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
from sklearn.model_selection import train_test_split


# for torch data
class MyDataset(Dataset): 
	def __init__(
		self,
		data_path: str,
		label=None,
		feature_keys: list = None,
	):
		"""
		Desc:
			For torch data
		Args:
			data_path, the torch data path
			label, provide external label
			feature_keys, feature names
		"""
		self.data = torch.load(data_path)
		self.label = label
		self.keys = list(self.data.keys())
		self.feature_keys = feature_keys

	def __len__(self):
		return len(self.keys)

	@staticmethod
	def _to_float_tensor(x):
		"""compatible torch.Tensor / numpy.ndarray / scalar"""
		if isinstance(x, torch.Tensor):
			return x.float()
		if isinstance(x, np.ndarray):
			return torch.from_numpy(x).float()
		return torch.tensor(x, dtype=torch.float32)

	def __getitem__(self, idx):
		key = self.keys[idx]
		item = self.data[key]

		# label
		label = self._to_float_tensor(
			self.label if self.label is not None else item['label']
		)

		# features
		features = {}
		if self.feature_keys is None:
			for k, v in item.items():
				if k != 'label' and isinstance(v, (torch.Tensor, np.ndarray)):
					features[k] = self._to_float_tensor(v)
		else:
			for k in self.feature_keys:
				if k not in item:
					raise KeyError(f"Feature key '{k}' not found in data item.")
				features[k] = self._to_float_tensor(item[k])
		return {
			'index': idx,
			'vid': key,
			**features,
			'label': label
		}


class LMDBDataset(Dataset):
	def __init__(
		self,
		lmdb_path: str,
		label=None,
		feature_keys: list = None,
		preload: bool = False,
		readonly: bool = True,
		lock: bool = False,
	):
		"""
		Desc:
			For LMDB data
		Args:
			lmdb_path: LMDB dir
			label: provide external label
			feature_keys: feature names
			preload: if preload to cache
			readonly: if read only
			lock: if write lock
		"""
		self.lmdb_path = lmdb_path
		self.label = label
		self.feature_keys = feature_keys
		self.preload = preload
		self.readonly = readonly
		self.lock = lock

		env = lmdb.open(
			self.lmdb_path,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False,
			max_readers=1
		)
		with env.begin() as txn:
			self.keys = pickle.loads(txn.get(b"__keys__"))
		env.close()

		# LMDB env delayed until it is opened within the worker
		self.env = None
		self.cache = None
		self.txn = None # Used for storing persistent read-only transactions

	def _init_env(self):
		"""Delay opening LMDB within the DataLoader worker and initiate a persistent read-only transaction务"""
		if self.env is None:
			# 1 Open LMDB environment
			self.env = lmdb.open(
				self.lmdb_path,
				readonly=self.readonly,
				lock=self.lock,
				readahead=not self.preload,
				meminit=False,
				max_readers=2048
			)

			# 2 Enable persistent read-only transactions (non preloaded model)
			if not self.preload:
				self.txn = self.env.begin(write=False)

			# 3 Preloading processing
			if self.preload:
				self.cache = []
				with self.env.begin() as txn:
					for k in self.keys:
						v = txn.get(k)
						if v is not None:
							npz_data = np.load(io.BytesIO(v))
							self.cache.append({kk: npz_data[kk] for kk in npz_data})

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, idx):
		if self.env is None:
			self._init_env()
		key = self.keys[idx]
		if self.cache is None:
			# Avoiding the overhead of starting/commit() every time __getitem__ is called
			if self.txn is None:
				with self.env.begin(write=False) as txn:
					v = txn.get(key)
			else:
				v = self.txn.get(key)
			if v is None:
				raise KeyError(f"Key {key} not found in LMDB.")
			# Data deserialization
			npz_data = np.load(io.BytesIO(v))
			item = {kk: npz_data[kk] for kk in npz_data}
		else:
			item = self.cache[idx]
		# label
		label = torch.tensor(
			self.label if self.label is not None else item.get("label", 0.0),
			dtype=torch.float32
		)
		# features 
		features = {}
		# Convert all required keys to torch. Sensor
		tensor_item = {k: torch.from_numpy(v).float() for k, v in item.items() if k != "label"}
		if self.feature_keys is None:
			features = tensor_item
		else:
			for k in self.feature_keys:
				if k not in tensor_item:
					raise KeyError(f"Feature key '{k}' not found in item.")
				features[k] = tensor_item[k]
		return {
			"vid": key.decode() if isinstance(key, bytes) else key,
			**features,
			"label": label
		}


def concat_and_loader(train_full, split_rate=0.8, batch_size=32, seed=42):
	"""
	Desc:
		Split the dataset
	Args:
		train_full, the full training data
		split_rate, take x% data for training
		batch_size, batch size
		seed, random seed
	"""
	dataset_size = len(train_full)
	indices = list(range(dataset_size))
	train_indices, val_indices = train_test_split(
		indices, 
		train_size=split_rate, 
		random_state=seed, 
		shuffle=True
	)

	train_subset = Subset(train_full, train_indices)
	val_subset = Subset(train_full, val_indices)

	train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
	val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
	return train_loader, val_loader