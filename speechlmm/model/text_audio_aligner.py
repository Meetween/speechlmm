import io
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio.transforms as T
import whisper
from transformers import AutoModelForSpeechSeq2Seq, PreTrainedTokenizer

from speechlmm.constants import (
    DEFAULT_AUDIO_EPAD_TOKEN,
    DEFAULT_AUDIO_OUTPUT_END_TOKEN,
    DEFAULT_AUDIO_OUTPUT_START_TOKEN,
    DEFAULT_AUDIO_PAD_TOKEN,
    IGNORE_INDEX,
)


class AlignWithUnkownTokens:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        # audio tokens
        self.pad_audio_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_PAD_TOKEN
        ]
        self.bos_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_START_TOKEN
        ]
        self.eos_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_END_TOKEN
        ]
        self.bos_audio_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_START_TOKEN
        ]
        self.eos_audio_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_END_TOKEN
        ]

    def align(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        audio_features_attention_mask: torch.BoolTensor,
        talking_head_use_text_tokens: bool,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
        # get the bos_audio and eos_audio positions
        batch_bos_audio_idx, bos_audios = torch.where(
            input_ids == self.bos_audio_token_id
        )
        bos_audios += 1
        batch_eos_audio_idx, eos_audios = torch.where(
            input_ids == self.eos_audio_token_id
        )
        # check if the BOS and EOS audio tokens are correctly aligned
        assert torch.equal(
            batch_bos_audio_idx, batch_eos_audio_idx
        ), "BOS and EOS audio tokens are not in the same batch"
        batch_audio_idx = batch_bos_audio_idx
        # check if the audio features attention mask is the same size as the batch_audio_idx
        assert audio_features_attention_mask.size(0) == batch_audio_idx.size(
            0
        ), "Audio features attention mask and batch_audio_idx do not match"

        # insert tokens according to the audio features
        new_input_ids, new_labels, new_attention_mask = [], [], []
        audio_idx = 0
        for batch_idx in range(input_ids.size(0)):
            if batch_idx not in batch_audio_idx:
                new_input_ids.append(input_ids[batch_idx])
                new_labels.append(labels[batch_idx])
                new_attention_mask.append(attention_mask[batch_idx])
                continue
            start = bos_audios[audio_idx]
            end = eos_audios[audio_idx]
            new_input_ids.append(
                torch.cat(
                    [
                        input_ids[batch_idx, :start],
                        torch.full(
                            (audio_features_attention_mask[audio_idx].sum(),),
                            self.pad_audio_token_id,
                            dtype=input_ids.dtype,
                            device=input_ids.device,
                        ),
                        input_ids[batch_idx, end:],
                    ]
                )
            )
            if talking_head_use_text_tokens:
                new_labels.append(
                    torch.cat(
                        [
                            labels[batch_idx, :start],
                            torch.full(
                                (
                                    audio_features_attention_mask[
                                        audio_idx
                                    ].sum(),
                                ),
                                self.pad_audio_token_id,
                                dtype=labels.dtype,
                                device=labels.device,
                            ),
                            labels[batch_idx, end:],
                        ]
                    )
                )
            else:
                new_labels.append(
                    torch.cat(
                        [
                            labels[batch_idx, :start],
                            torch.full(
                                (1,),
                                self.pad_audio_token_id,
                                dtype=labels.dtype,
                                device=labels.device,
                            ),
                            torch.full(
                                (
                                    audio_features_attention_mask[
                                        audio_idx
                                    ].sum()
                                    - 1,
                                ),
                                IGNORE_INDEX,
                                dtype=labels.dtype,
                                device=labels.device,
                            ),
                            labels[batch_idx, end:],
                        ]
                    )
                )
            new_attention_mask.append(
                torch.cat(
                    [
                        attention_mask[batch_idx, :start],
                        torch.ones(
                            audio_features_attention_mask[audio_idx].sum(),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                        attention_mask[batch_idx, end:],
                    ]
                )
            )
            audio_idx += 1

        input_ids = torch.nn.utils.rnn.pad_sequence(
            new_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            new_labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            new_attention_mask, batch_first=True, padding_value=False
        )
        return input_ids, labels, attention_mask


class AlignCodesAndPhonemes:
    def __init__(
        self,
        pad_token_id: int,
        eos_token_id: int,
        bos_audio_token_id: int,
        eos_audio_token_id: int,
        sil_audio_token_id: int,
        duration_prediction: bool = False,
    ):
        super(AlignCodesAndPhonemes, self).__init__()
        self.duration_prediction = duration_prediction
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_audio_token_id = bos_audio_token_id
        self.eos_audio_token_id = eos_audio_token_id
        self.sil_audio_token_id = sil_audio_token_id
        self.end_sequence = torch.tensor(
            [self.eos_audio_token_id, self.eos_token_id]
        ).to(torch.long)

    def get_phonemes_and_prepare_inputs_ids(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor
    ):
        bos_audios = torch.where(input_ids == self.bos_audio_token_id)[1] + 1
        eos_audios = torch.where(input_ids == self.eos_audio_token_id)[1]
        phoneme_length = eos_audios - bos_audios
        max_len_phonemes = phoneme_length.max()
        pads_to_add = max_len_phonemes - phoneme_length
        phonemes, padded_input_ids, padded_labels = [], [], []
        for cur_inputs, bos, eos, pad_to_add in zip(
            input_ids, bos_audios, eos_audios, pads_to_add
        ):
            phonemes += [cur_inputs[bos:eos]]
            padded_input_ids += [
                torch.cat(
                    [
                        cur_inputs[:eos],
                        self.end_sequence,
                        self.pad_token_id.repeat(pad_to_add),
                    ],
                    dim=-1,
                )
            ]
            padded_labels += [
                torch.cat(
                    [
                        labels[:eos],
                        self.end_sequence,
                        -100 * torch.ones(pad_to_add, dtype=torch.long),
                    ],
                    dim=-1,
                )
            ]

        phonemes = torch.nn.utils.rnn.pad_sequence(
            phonemes, batch_first=True, padding_value=self.pad_token_id
        )
        input_ids = torch.stack(padded_input_ids)
        labels = torch.stack(padded_labels)
        phonemes_attention_mask = phonemes != self.pad_token_id
        return phonemes, bos_audios + 1, eos_audios, phonemes_attention_mask

    def get_text_ids(self, input_ids: torch.LongTensor):
        bos_audios = torch.where(input_ids == self.bos_audio_token_id)[1] + 1
        eos_audios = torch.where(input_ids == self.eos_audio_token_id)[1]
        phonemes = []
        for cur_inputs, bos, eos in zip(input_ids, bos_audios, eos_audios):
            phonemes += [cur_inputs[bos:eos]]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            phonemes, batch_first=True, padding_value=self.pad_token_id
        )
        phonemes_attention_mask = input_ids != self.pad_token_id
        return input_ids, bos_audios + 1, eos_audios, phonemes_attention_mask

    def align(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        audio_encoder_output: Dict[str, torch.Tensor],
    ):
        codes_padding_mask = ~audio_encoder_output["attention_mask"]
        code_lengths = codes_padding_mask.size(
            1
        ) - codes_padding_mask.long().sum(dim=1)
        new_inputs, new_labels, new_attention_mask = [], [], []
        # get bos_audio and eos_audio positions
        bos_audios = torch.where(input_ids == self.bos_audio_token_id)[1] + 1
        eos_audios = torch.where(input_ids == self.eos_audio_token_id)[1]
        # calculate the phonemes length
        phonemes_length = eos_audios - bos_audios
        # calculate the difference between phonemes and codes
        diff = phonemes_length - code_lengths
        max_len_codes = codes_padding_mask.size(1)
        diff_to_end_of_padded_codes = (
            max_len_codes - phonemes_length + diff - 2
        )  # -2 cause each sequence ends with <eos_audio_token_id> <eos_token_id>
        diff_to_end_of_padded_codes[diff_to_end_of_padded_codes < 0] = 0
        to_pad_inputs, to_pad_labels, to_pad_attention_mask = [], [], []
        end_sequence = (
            torch.tensor([self.eos_audio_token_id, self.eos_token_id])
            .to(torch.long)
            .to(input_ids.device)
        )
        sil_token = (
            torch.tensor([self.sil_audio_token_id])
            .to(torch.long)
            .to(input_ids.device)
        )
        pad_token = (
            torch.tensor([self.pad_token_id])
            .to(torch.long)
            .to(input_ids.device)
        )
        ignore_token = torch.tensor([-100]).to(torch.long).to(input_ids.device)
        end_sequence_attention_mask = (
            torch.tensor([True, True]).to(torch.bool).to(attention_mask.device)
        )
        True_attention_mask = (
            torch.tensor([True]).to(torch.bool).to(attention_mask.device)
        )
        False_attention_mask = (
            torch.tensor([False]).to(torch.bool).to(attention_mask.device)
        )

        for (
            cur_inputs,
            cur_labels,
            cur_attention_mask,
            cur_diff,
            eos,
            cur_diff_to_code,
        ) in zip(
            input_ids,
            labels,
            attention_mask,
            diff,
            eos_audios,
            diff_to_end_of_padded_codes,
        ):
            to_pad_inputs += [
                torch.cat(
                    [
                        cur_inputs[: eos - max(0, cur_diff)],
                        sil_token.repeat(max(0, -cur_diff)),
                        end_sequence,
                        pad_token.repeat(cur_diff_to_code),
                    ],
                    dim=-1,
                )
            ]
            to_pad_labels += [
                torch.cat(
                    [
                        cur_labels[: eos - max(0, cur_diff)],
                        sil_token.repeat(max(0, -cur_diff)),
                        end_sequence,
                        ignore_token.repeat(cur_diff_to_code),
                    ],
                    dim=-1,
                )
            ]
            to_pad_attention_mask += [
                torch.cat(
                    [
                        cur_attention_mask[: eos - max(0, cur_diff)],
                        True_attention_mask.repeat(max(0, -cur_diff)),
                        end_sequence_attention_mask,
                        False_attention_mask.repeat(cur_diff_to_code),
                    ],
                    dim=-1,
                )
            ]
        # pad to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            to_pad_inputs, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            to_pad_labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            to_pad_attention_mask, batch_first=True, padding_value=False
        )

        # prepare input for linear projector in case of adapter + phonemes
        phoneme_start_pos = bos_audios + 1  # means + 2
        phoneme_end_pos = bos_audios + codes_padding_mask.size(1)

        return (
            input_ids,
            labels,
            attention_mask,
            phoneme_start_pos,
            phoneme_end_pos,
        )


class AlignCodesAndText:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        align_with_whisper: bool = False,
        restore_punctuation_and_spaces: bool = False,
    ):
        import ssl

        ssl._create_default_https_context = (
            ssl._create_unverified_context
        )  # NOTE(st3p99) workaround to avoid SSL error on Helios when downloading the model
        self.align_with_whisper = align_with_whisper
        logging.info(
            f"Using {'Whisper' if self.align_with_whisper else 'CTC'} for alignment"
        )
        if self.align_with_whisper:
            self.whisper_model = whisper.load_model("turbo")
            self.get_word_timestamps = self.get_word_timestamps_w_whisper
        else:
            self.ctc_aligner = CTCForcedAlignment(
                restore_punctuation_and_spaces=restore_punctuation_and_spaces
            )
            self.get_word_timestamps = (
                self.ctc_aligner.get_word_timestamps_w_ctc
            )
        self.tokenizer = tokenizer
        self.pad_audio_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_PAD_TOKEN
        ]
        self.epad_audio_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_EPAD_TOKEN
        ]
        self.bos_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_START_TOKEN
        ]
        self.eos_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_END_TOKEN
        ]
        self.bos_audio_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_START_TOKEN
        ]
        self.eos_audio_token_id = self.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_END_TOKEN
        ]

    # FIXME(st3p99): use a forced aligned instead of whisper
    def align(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
        frame_rate: float,
        language: str,
        sentence: str = None,
        alignment_type: bool = False,
        word_timestamps: List[Tuple[float, str]] = None,
    ) -> List[int]:
        # if word_timestamps is provided, use it, otherwise use the aligner to get it from the audio and sentence
        if sentence is None and word_timestamps is None:
            raise ValueError(
                "Either sentence or word_timestamps must be provided"
            )
        if word_timestamps is None:
            word_timestamps = self.get_word_timestamps(
                audio_array, sampling_rate, sentence, language
            )

        assert alignment_type in [
            "text_pad_epad",
            "pad_epad",
        ], f"Invalid alignment type {alignment_type}, Options are ['text_pad_epad', 'pad_epad']"

        # "text_pad_epad"

        # compute text_tokens from word_timestamps
        text_tokens_per_word = []
        for _, word in word_timestamps:
            word_tokens = self.tokenizer(word)["input_ids"][
                1:
            ]  # remove heading space from word and remove bos token from tokenizer output
            text_tokens_per_word.append(word_tokens)

        frame_duration = 1 / frame_rate  # Duration of each frame in seconds

        # get the total duration of the audio from the audio encoder output
        total_duration = audio_array.shape[0] / sampling_rate
        total_frames = int(np.ceil(total_duration / frame_duration))

        # Initialize the aligned tokens list with 'PAD'
        aligned_tokens = [self.pad_audio_token_id] * total_frames

        # Fill aligned tokens
        offset = 0
        # assert len(word_timestamps) == len(text_tokens_per_word), "Number of words and tokens do not match"
        for word_index, (timestamp, word) in enumerate(word_timestamps):
            start_frame = int(np.floor(timestamp / frame_duration))  # ti
            end_frame = start_frame + len(
                text_tokens_per_word[word_index]
            )  # ti+ni

            # FIXME: This is a temporary fix to avoid out of bounds error
            if end_frame > total_frames:
                end_frame = total_frames

            start_frame += offset
            end_frame += offset

            # Mark the end of padding
            if (
                start_frame == 1
            ):  # Note that if ti = 1, we instead insert EPAD at index 1, and shift the text tokens. W
                aligned_tokens[1] = self.epad_audio_token_id
                offset = 1
                start_frame += offset
                end_frame += offset
                aligned_tokens.append(
                    self.pad_audio_token_id
                )  # adding one frame more
            elif (
                word_index > 0
                and aligned_tokens[start_frame - 1] == self.pad_audio_token_id
            ):
                aligned_tokens[start_frame - 1] = self.epad_audio_token_id

            for i in range(start_frame, end_frame):
                if alignment_type == "text_pad_epad":
                    aligned_tokens[i] = text_tokens_per_word[word_index][
                        i - start_frame
                    ]
                else:
                    continue

        if aligned_tokens[-1] == self.pad_audio_token_id:
            aligned_tokens[-1] = self.epad_audio_token_id

        return aligned_tokens

    def get_word_timestamps_w_whisper(
        self,
        audio_array,
        sampling_rate,
        sentence: str,
        language: str,
    ) -> List[Tuple[float, str]]:
        # Transcribe audio with word-level timestamps
        # check sampling rate is compatible with whisper model
        if sampling_rate != 16000:
            # Resample audio to 16kHz
            audio_array = T.Resample(sampling_rate, 16000)(audio_array)
        result = self.whisper_model.transcribe(
            audio_array, word_timestamps=True
        )

        # Extract word timestamps and the words themselves
        word_timestamps = []
        for segment in result["segments"]:
            for word_info in segment["words"]:
                word = word_info["word"]
                start_time = word_info["start"]
                word_timestamps.append((start_time, word))

    def _get_text_tokens_per_word(self, sentence: str) -> List[List[int]]:
        words = sentence.split()

        # Create a list to store token IDs for each word
        text_tokens_per_word = []

        # Encode each word and store the token IDs
        for word in words:
            # Tokenize the word (add special tokens for single-word tokens)
            tokenized_word = self.tokenizer(word)["input_ids"]

            # Remove the start token (if it exists)
            if tokenized_word[0] == self.bos_token_id:
                tokenized_word = tokenized_word[1:]

            text_tokens_per_word.append(tokenized_word)

        return text_tokens_per_word

    def handle_text_audio_alignment(
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        aligned_transcription_ids,
        aligned_transcription_attention_mask,
        audio_encoder_output,
        pad_token_id: int,
        pad_audio_token_id: int,
        epad_audio_token_id: int,
        bos_audio_token_id: int,
        eos_audio_token_id: int,
    ):
        audio_attn_mask = audio_encoder_output.padding_mask

        new_input_ids = []
        new_labels = []
        new_attention_mask = []
        new_aligned_transcription_ids = []
        new_aligned_transcription_attention_mask = []
        audio_idx = 0
        assert (
            audio_attn_mask.shape[0] == aligned_transcription_ids.shape[0]
        ), "Number of audio features and aligned transcriptions do not match"
        for i, (
            cur_input_ids,
            cur_labels,
            cur_attention_mask,
        ) in enumerate(
            zip(
                input_ids,
                labels,
                attention_mask,
            )
        ):
            if (
                torch.where(cur_input_ids == bos_audio_token_id)[0].shape[0]
                == 0
                or torch.where(cur_input_ids == eos_audio_token_id)[0].shape[0]
                == 0
            ):
                pad_len = audio_attn_mask.shape[1]
                cur_new_input_ids = torch.cat(
                    [
                        cur_input_ids,
                        torch.full(
                            (pad_len,),
                            pad_token_id,
                            dtype=input_ids.dtype,
                            device=input_ids.device,
                        ),
                    ]
                )

                cur_new_labels = torch.cat(
                    [
                        cur_labels,
                        torch.full(
                            (pad_len,),
                            IGNORE_INDEX,
                            dtype=labels.dtype,
                            device=labels.device,
                        ),
                    ]
                )

                cur_new_attention_mask = torch.cat(
                    [
                        cur_attention_mask,
                        torch.full(
                            (pad_len,),
                            False,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ]
                )
                new_input_ids.append(cur_new_input_ids)
                new_labels.append(cur_new_labels)
                new_attention_mask.append(cur_new_attention_mask)
                continue
            cur_audio_attn_mask = audio_attn_mask[audio_idx]
            target_len = (~cur_audio_attn_mask).sum().item()
            cur_aligned_ids = aligned_transcription_ids[audio_idx]
            cur_aligned_attn_mask = aligned_transcription_attention_mask[
                audio_idx
            ]
            cur_aligned_no_pad = cur_aligned_ids[cur_aligned_attn_mask]
            if cur_aligned_no_pad.shape[0] < target_len:
                pad_tensor = torch.full(
                    (target_len - cur_aligned_no_pad.shape[0],),
                    pad_audio_token_id,
                    dtype=cur_aligned_no_pad.dtype,
                    device=cur_aligned_no_pad.device,
                )
                cur_aligned_no_pad = torch.cat(
                    [cur_aligned_no_pad, pad_tensor]
                )
            elif (
                cur_aligned_no_pad.shape[0] > target_len
            ):  # remove padding to match audio features shape
                # to_remove = cur_aligned_no_pad[
                #     target_len - 1 :
                # ]  # target_len-1 is the last token of cur_aligned_no_pad that will be replaced by EPAD_AUDIO_TOKEN
                # for cur_to_remove in to_remove:
                #     if (
                #         cur_to_remove != pad_audio_token_id
                #         and cur_to_remove != epad_audio_token_id
                #     ):
                #         # FIXME: this should not happen
                #         print(
                #             f"Found no PAD_AUDIO_TOKEN or EPAD_AUDIO_TOKEN in cur_input_ids_no_pad while removing padding to match audio features shape. cur_input_ids_no_pad: {cur_aligned_no_pad}, to_remove: {to_remove}"
                #         )
                # cur_aligned_no_pad = cur_aligned_no_pad[:target_len]

                # # put EPAD_AUDIO_TOKEN at the end of cur_aligned_no_pad
                # cur_aligned_no_pad[-1] = epad_audio_token_id

                cur_aligned_no_pad = AlignCodesAndText._trim_aligned_tokens(
                    cur_aligned_no_pad,
                    target_len,
                    pad_audio_token_id,
                    epad_audio_token_id,
                )

            assert (
                cur_aligned_no_pad.shape[0] == target_len
            ), f"cur_aligned_no_pad.shape[0]: {cur_aligned_no_pad.shape[0]}, target_len: {target_len}"

            start = torch.where(cur_input_ids == bos_audio_token_id)[0].item()
            end = torch.where(cur_input_ids == eos_audio_token_id)[0].item()

            audio_pad_len = cur_audio_attn_mask.shape[0] - target_len

            cur_new_input_ids = torch.cat(
                [
                    cur_input_ids[: start + 1],
                    cur_aligned_no_pad,
                    cur_input_ids[end:],
                    torch.full(
                        (audio_pad_len,),
                        pad_token_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                ]
            )

            cur_new_labels = torch.cat(
                [
                    cur_labels[: start + 1],
                    cur_aligned_no_pad,
                    cur_labels[end:],
                    torch.full(
                        (audio_pad_len,),
                        IGNORE_INDEX,
                        dtype=labels.dtype,
                        device=labels.device,
                    ),
                ]
            )

            cur_new_attention_mask = torch.cat(
                [
                    cur_attention_mask[: start + 1],
                    torch.full(
                        (target_len,),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    cur_attention_mask[end:],
                    torch.full(
                        (audio_pad_len,),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ]
            )
            cur_aligned_ids = torch.cat(
                [
                    cur_aligned_no_pad,
                    torch.full(
                        (audio_pad_len,),
                        pad_token_id,
                        dtype=cur_aligned_ids.dtype,
                        device=cur_aligned_ids.device,
                    ),
                ]
            )
            cur_aligned_attn_mask = cur_audio_attn_mask

            new_aligned_transcription_ids.append(cur_aligned_ids)
            new_aligned_transcription_attention_mask.append(
                cur_aligned_attn_mask
            )

            new_input_ids.append(cur_new_input_ids)
            new_labels.append(cur_new_labels)
            new_attention_mask.append(cur_new_attention_mask)
            audio_idx += 1

        new_aligned_transcription_ids = torch.stack(
            new_aligned_transcription_ids
        )
        new_aligned_transcription_attention_mask = torch.stack(
            new_aligned_transcription_attention_mask
        )

        new_input_ids = torch.stack(new_input_ids)
        new_labels = torch.stack(new_labels)
        new_attention_mask = torch.stack(new_attention_mask)
        return (
            new_input_ids,
            new_labels,
            new_attention_mask,
            new_aligned_transcription_ids,
            new_aligned_transcription_attention_mask,
        )

    @staticmethod
    def _trim_aligned_tokens(
        cur_aligned_no_pad, target_len, pad_audio_token_id, epad_audio_token_id
    ):
        """
        Trim aligned tokens to match target length while ensuring sequence ends with EPAD token.
        Preserves content tokens and all EPAD tokens where possible.
        """
        if len(cur_aligned_no_pad) <= target_len:
            return cur_aligned_no_pad

        # Find all PAD token indices at once
        pad_indices = torch.where(cur_aligned_no_pad == pad_audio_token_id)[0]

        if len(pad_indices) >= len(cur_aligned_no_pad) - target_len:
            # We have enough PAD tokens to remove - remove them from end to start
            num_to_remove = len(cur_aligned_no_pad) - target_len
            indices_to_keep = torch.ones_like(
                cur_aligned_no_pad, dtype=torch.bool
            )
            indices_to_keep[pad_indices[-num_to_remove:]] = False
            return cur_aligned_no_pad[indices_to_keep]

        # If we can't remove enough PAD tokens, truncate while preserving final EPAD
        logging.warning(
            f"Could not reach target length {target_len} by removing only PAD tokens. "
            f"Current length: {len(cur_aligned_no_pad)}. "
            f"Truncating sequence to match target length. This may affect alignment quality."
        )
        result = cur_aligned_no_pad[: target_len - 1]
        return torch.cat(
            [result, torch.tensor([epad_audio_token_id], device=result.device)]
        )


# https://github.com/pytorch/audio

# BSD 2-Clause License

# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Implimentation of CTC forced alignment
# https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html

import re

import num2words
import torch
import torchaudio
import torchaudio.functional as F
from num2words import num2words


class CTCForcedAlignment:
    def __init__(self, restore_punctuation_and_spaces: bool = False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        bundle = torchaudio.pipelines.MMS_FA
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model(with_star=False).to(self.device)
        self.LABELS = bundle.get_labels(star=None)
        self.DICTIONARY = bundle.get_dict(star=None)
        # self.lec = inflect.engine()
        self.restore_punctuation_and_spaces = restore_punctuation_and_spaces

    def get_word_timestamps_w_ctc(
        self, audio_array, sampling_rate, sentence: str, language: str
    ) -> List[Tuple[float, str]]:
        if type(audio_array) == np.ndarray:
            audio_array = torch.tensor(audio_array)
        if len(audio_array.shape) == 1:
            audio_array = audio_array.unsqueeze(0)

        # Check sampling rate compatibility and resample if necessary
        if sampling_rate != self.sample_rate:
            audio_array = T.Resample(sampling_rate, self.sample_rate)(
                audio_array
            )

        # from tensor to bytes
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_array.to(device="cpu"),
            self.sample_rate,
            format="wav",
        )
        waveform_bytes = buffer.getvalue()
        # Align and obtain the start timestamp for each word
        alignment_results = self.align(waveform_bytes, sentence, language)
        word_timestamps = [
            (result["x0"] / self.sample_rate, result["word"])
            for result in alignment_results
        ]
        return word_timestamps

    @staticmethod
    def process_text(text: str, text_language: str = "en") -> List[str]:
        text = re.sub(
            r"\d+(\.\d+)?",
            lambda x: num2words(x.group(), lang=text_language),
            text.lower(),
        )
        text = re.sub(r"[^a-z\s]", "", text)
        return text.split()

    @staticmethod
    def process_text_w_punctuation_and_spaces(
        text: str, text_language: str = "en"
    ) -> List[str]:
        # Convert numbers to words while preserving case
        text = re.sub(
            r"\d+(\.\d+)?",
            lambda x: num2words(x.group(), lang=text_language),
            text,
        )

        tokens = re.findall(
            r'(\s*["\'\(\[]?\s*\b\w+(?:[-\']\w+)*\b[.,!?;:]*\s*["\'\)\]]?\s*)',
            text,
        )
        tokens = [token for token in tokens if token.strip()]

        return tokens

    def _unflatten(self, list_, lengths):
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret

    def get_word(self, waveform, spans, num_frames, transcript):
        ratio = waveform.size(1) / num_frames
        x0 = int(ratio * spans[0].start)
        x1 = int(ratio * spans[-1].end)
        return {"x0": x0, "x1": x1, "word": transcript}

    def _extract_world_level(
        self, aligned_tokens, alignment_scores, transcript
    ):
        token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
        word_spans = self._unflatten(
            token_spans, [len(word) for word in transcript]
        )
        return word_spans

    def _align(self, emission, tokens):
        targets = torch.tensor(
            [tokens], dtype=torch.int32, device=torch.device("cpu")
        )
        alignments, scores = F.forced_align(emission.cpu(), targets, blank=0)
        alignments, scores = alignments[0], scores[0]
        scores = scores.exp()
        return alignments, scores

    def align(self, audio, transcript, language="en"):
        waveform, sr = torchaudio.load(audio)
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=self.sample_rate
        )
        original_transcript = transcript
        transcript = CTCForcedAlignment.process_text(transcript, language)

        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))

        tokenized_transcript = [
            self.DICTIONARY[c] for word in transcript for c in word
        ]
        alignments, scores = self._align(emission, tokenized_transcript)
        word_spans = self._extract_world_level(alignments, scores, transcript)
        num_frames = emission.size(1)

        if self.restore_punctuation_and_spaces:
            restored_transcript = (
                CTCForcedAlignment.process_text_w_punctuation_and_spaces(
                    original_transcript, language
                )
            )
        else:
            restored_transcript = transcript

        if len(word_spans) != len(restored_transcript):
            logging.warning(
                f"Number of words in the restored transcript w. punctuation and spaces ({len(restored_transcript)}) does not match the number of word_spans ({len(word_spans)})"
            )
            restored_transcript = transcript
            # TODO: Assuming that this happen rarely, we can implement a workaround
            # by removing words from the restored_transcript to match word_spans and vice versa
            # if len(restored_transcript) > len(word_spans):
            #     restored_transcript = restored_transcript[:len(word_spans)]
            # else:
            #     word_spans = word_spans[:len(restored_transcript)]

        outputs = [
            self.get_word(
                waveform, word_spans[i], num_frames, restored_transcript[i]
            )
            for i in range(len(word_spans))
        ]

        outputs[0]["x0"] = 0

        for i in range(len(outputs)):
            output = outputs[i]
            x0 = output["x0"]

            if i == len(outputs) - 1:
                x1 = output["x1"]
            else:
                x1 = outputs[i + 1]["x0"]

            outputs[i]["audio"] = waveform[:, x0:x1]

        return outputs

    def free(self):
        del self.model
