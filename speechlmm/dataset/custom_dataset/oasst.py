import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import (
    QUALITY_FILTER,
    SPAM_FILTER,
    CustomDataset,
)
from speechlmm.dataset.custom_dataset.preparers import (
    MultiTurnTextInstructPreparer,
)


class OasstDataset(CustomDataset):
    name = "OASST"
    codename = "oasst"

    # Add to optional filters list
    optional_filters = [QUALITY_FILTER, SPAM_FILTER]
    text_keys = ["text"]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        self.preparers = {
            "MultiTurnTextInstruct": MultiTurnTextInstructPreparer()
        }
        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
        )

    def _get_dataset_path(
        self,
        dataset_dir: str,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ) -> str:
        filename = f"{partition_name}.parquet"
        dataset_path = Path(dataset_dir, filename)
        return str(dataset_path)

    def _post_filter_hook(
        self,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ):
        """Create a new dataset where each sample is a complete conversation."""
        # Group messages by conversation tree
        conversations = defaultdict(list)
        for example in self.cur_dataset:
            tree_id = example.get("message_tree_id")
            if tree_id:
                conversations[tree_id].append(example)

        # check if max_num_words or min_num_words is specified
        max_num_words = partition_spec.get("max_num_words", None)
        min_num_words = partition_spec.get("min_num_words", None)

        # Create new examples with complete conversations
        new_examples = []
        for tree_id, messages in conversations.items():
            # Sort messages to ensure correct order (parent messages first)
            sorted_messages = []
            message_dict = {msg["message_id"]: msg for msg in messages}

            # Find leaf message (one without children)
            leaf_message = None
            for msg in messages:
                if not any(
                    m.get("parent_id") == msg["message_id"] for m in messages
                ):
                    leaf_message = msg
                    break

            num_words = 0
            if leaf_message:
                # Build conversation from leaf to root
                current_message = leaf_message
                labels = current_message.get("labels", {})
                quality = (
                    labels.get("quality", {}).get("value") if labels else None
                )
                while current_message:
                    sorted_messages.insert(
                        0,
                        {
                            "role": (
                                "human"
                                if current_message["role"] == "prompter"
                                else "gpt"
                            ),
                            "content": current_message["text"],
                            "message_id": current_message["message_id"],
                            "lang": current_message.get("lang"),
                            "quality": quality,
                            "review_result": current_message.get(
                                "review_result"
                            ),
                        },
                    )
                    num_words += len(current_message["text"].split())
                    parent_id = current_message.get("parent_id")
                    current_message = message_dict.get(parent_id)

                if max_num_words and num_words > max_num_words:
                    continue
                if min_num_words and num_words < min_num_words:
                    continue
                # Create new example with complete conversation
                new_example = {
                    "message_tree_id": tree_id,
                    "messages": sorted_messages,
                    "lang": sorted_messages[0][
                        "lang"
                    ],  # Use root message language
                    "num_messages": len(sorted_messages),
                }
                new_examples.append(new_example)

        # Create new dataset from conversation examples
        self.cur_dataset = Dataset.from_list(new_examples)
        logging.info(
            f"Created {len(new_examples)} complete conversations from {len(messages)} messages"
        )

    def _filter_example(
        self, example, language, partition_name, partition_spec
    ) -> bool:
        if QUALITY_FILTER in self.optional_filters:
            return self._filter_by_quality(example, partition_spec)

        if SPAM_FILTER in self.optional_filters:
            return self._filter_by_spam(example, partition_spec)
        return True

    def _filter_by_quality(
        self, example: Dict[str, Any], partition_spec: dict
    ) -> bool:
        """Filter examples based on quality score."""
        min_quality = partition_spec.get("min_quality", 0.5)
        if example["labels"] is not None and "quality" in example["labels"]:
            quality_score = example["labels"]["quality"]["value"]
            return quality_score >= min_quality

        return True

    def _filter_by_spam(
        self, example: Dict[str, Any], partition_spec: dict
    ) -> bool:
        """Filter spam or deleted examples."""
        return not example.get("deleted", False) and example.get(
            "review_result", True
        )
