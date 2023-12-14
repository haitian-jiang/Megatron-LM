# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
import logging
import numpy as np
import torch
from typing import Dict, List

from megatron.core.datasets.blended_megatron_dataset_config import (
    convert_split_vector_to_split_matrix,
    parse_and_normalize_split,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.datasets.utils import Split, log_single_rank
# >>>
import os
# from typing import List

from megatron.core.models.retro.data.config import RetroPreprocessingConfig
from megatron.core.models.retro.data.utils import get_gpt_data_dir

# from .multi_split_gpt_dataset import MultiSplitGPTDatasetConfig
# <<<

logger = logging.getLogger(__name__)


@dataclass
class MultiSplitGPTDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core blended and megatron Retro datasets

    Attributes:
        return_document_ids (bool): Whether to return the document ids when querying the dataset.
        Turn this option on during preprocessing.

        split_preprocessing (str): The Retro preprocessing split string. It follows the same
        pattern convention as 'split'. Not to be used with 'blend_per_split'.
    """

    return_document_ids: bool = None

    split_preprocessing: str = None

    def __post_init__(self):
        super().__post_init__()
        assert self.split is not None, "the Retro data pipeline does not support 'blend_per_split'"
        assert self.return_document_ids is not None, "this attribute must be user defined"
        assert self.split_preprocessing is not None, "this attribute must be user defined"
        split_vector = parse_and_normalize_split(self.split)
        split_preprocessing_vector = parse_and_normalize_split(self.split_preprocessing)
        if not np.allclose(split_vector, split_preprocessing_vector):
            self.split_matrix = convert_split_vector_to_split_matrix(
                split_vector, split_preprocessing_vector
            )
            log_single_rank(
                logger,
                logging.WARNING,
                f"split =/= split_preprocessing. Let split_matrix = {self.split_matrix}",
            )


class MultiSplitGPTDataset(GPTDataset):
    """Retro's customized GPT dataset.

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        MegatronDataset

        indexed_indices (np.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (MultiSplitGPTDatasetConfig): The Retro-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        indexed_indices: np.ndarray,
        num_samples: int,
        index_split: Split,
        config: MultiSplitGPTDatasetConfig,
    ) -> None:
        super().__init__(indexed_dataset, indexed_indices, num_samples, index_split, config)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Abstract method implementation

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, np.ndarray]: The text ids and (optionally) the document ids wrapped in a
            dictionary
        """
        # >>>
        from lutil import pax
        pax({"config": self.config})
        # <<<
        text, document_ids = self._query_document_sample_shuffle_indices(idx)
        if self.config.return_document_ids:
            return {"text": text, "document_ids": document_ids}
        else:
            return {"text": text}

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Inherited method implementation

        The preprocessing split used for preprocessing will constrain the samples available for 
        pretraining.

        Returns:
            List[str]: The key config attributes
        """
        return super(MultiSplitGPTDataset, MultiSplitGPTDataset)._key_config_attributes() + [
            "split_preprocessing"
        ]


# >>>
# def core_retro_dataset_config_from_args(args, retro_args):
#     return MultiSplitGPTDatasetConfig(
#         is_built_on_rank=is_dataset_built_on_rank,
#         random_seed=retro_args.retro_gpt_seed,
#         sequence_length=retro_args.retro_gpt_seq_length,
#         blend=args.data_path if args.data_path is not None else retro_args.retro_gpt_data_path,
#         split=args.split,
#         path_to_cache=args.data_cache_path,
#         return_document_ids=retro_args.retro_return_doc_ids,
#         split_preprocessing=retro_args.retro_gpt_split,
#     )
# +++
# def core_gpt_dataset_config_from_retro_preprocessing_config(
#     config: RetroPreprocessingConfig,
#     is_dataset_built_on_rank: bool,
# ) -> GPTDatasetConfig:
#     data_dir = get_gpt_data_dir(config.retro_project_dir)
#     blend = list(config.retro_gpt_data_path)
#     for i in range(len(blend) - 1, -1, -2):
#         blend[i] = os.path.join(data_dir, blend[i])
#     return GPTDatasetConfig(
#         is_built_on_rank=is_dataset_built_on_rank,
#         random_seed=config.retro_gpt_seed,
#         sequence_length=config.retro_gpt_seq_length,
#         blend=blend,
#         split=config.retro_gpt_split,
#         path_to_cache=config.retro_gpt_data_cache_path,
#         return_document_ids=True,
#     )
# +++
# def core_multi_split_gpt_dataset_config_from_retro_preprocessing_config(
#     config: RetroPreprocessingConfig,
#     split: str,
#     return_document_ids: bool,
#     is_dataset_built_on_rank: bool,
#     custom_data_path: List[str] = None,
# ) -> MultiSplitGPTDatasetConfig:
#     data_dir = get_gpt_data_dir(config.retro_project_dir)
#     if custom_data_path is not None:
#         blend=custom_data_path
#     else:
#         blend = list(config.retro_gpt_data_path)
#         for i in range(len(blend) - 1, -1, -2):
#             blend[i] = os.path.join(data_dir, blend[i])
#     config = MultiSplitGPTDatasetConfig(
#         is_built_on_rank=is_dataset_built_on_rank,
#         random_seed=config.retro_gpt_seed,
#         sequence_length=config.retro_gpt_seq_length,
#         blend=blend,
#         split=split,
#         split_preprocessing=config.retro_gpt_split,
#         path_to_cache=config.retro_gpt_data_cache_path,
#         return_document_ids=return_document_ids,
#     )
#     # >>>
#     if not return_document_ids or custom_data_path is not None:
#         from lutil import pax
#         pax("config")
#     # <<<
#     return config
# <<<


def multi_split_gpt_train_valid_test_datasets_provider(data_config, train_valid_test_num_samples):
    """Build train, valid, and test datasets."""

    print_rank_0('> building multi-split train, validation, and test datasets '
                 'for GPT ...')
    
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        # >>>
        # GPTDataset,
        MultiSplitGPTDataset,
        # <<<
        train_valid_test_num_samples,
        data_config,
    ).build()
    print_rank_0("> finished creating multi-split GPT datasets ...")

    return train_ds, valid_ds, test_ds