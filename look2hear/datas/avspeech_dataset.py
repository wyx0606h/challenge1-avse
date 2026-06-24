"""
Audio-Visual Speech Dataset for Speech Enhancement
"""
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from .transform import get_preprocessing_pipelines


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    """Normalize audio waveform"""
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class AVSpeechDataset(Dataset):
    """
    Audio-Visual Speech Dataset

    Args:
        json_dir: Directory containing mix.json and s1.json files
        n_src: Number of sources (1 for enhancement)
        sample_rate: Audio sample rate
        segment: Segment duration in seconds (None for full audio)
        normalize_audio: Whether to normalize audio
        is_train: Training mode (for different video augmentation)
    """
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 1,
        sample_rate: int = 16000,
        segment: float = 2.0,
        normalize_audio: bool = False,
        is_train: bool = True
    ):
        super().__init__()
        if json_dir is None or json_dir == "":
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]

        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * 25)  # 25 fps for video

        self.n_src = n_src
        self.test = self.seg_len is None

        # Load JSON files
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)

        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []

        # Filter out short utterances
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0

        if not self.test:
            for i in range(len(mix_infos) - 1, -1, -1):
                if mix_infos[i][1] < self.seg_len:
                    drop_utt += 1
                    drop_len += mix_infos[i][1]
                    del mix_infos[i]
                    for src_inf in sources_infos:
                        del src_inf[i]
                else:
                    self.mix.append(mix_infos[i])
                    self.sources.append(sources_infos[0][i])
        else:
            for i in range(len(mix_infos)):
                self.mix.append(mix_infos[i])
                self.sources.append(sources_infos[0][i])

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
            )
        )
        self.length = len(self.mix)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        self.EPS = 1e-8

        # Random start for training, fixed for test
        if self.test:
            rand_start = 0
            stop = None
        else:
            rand_start = 0
            stop = rand_start + self.seg_len

        # Load audio
        mix_source, _ = sf.read(
            self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
        )
        source = sf.read(
            self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
        )[0]

        # Load and preprocess video
        source_mouth = self.lipreading_preprocessing_func(
            np.load(self.sources[idx][1])["data"]
        )
        if self.fps_len is not None:
            source_mouth = source_mouth[:self.fps_len]

        # Convert to tensors
        source = torch.from_numpy(source)
        mixture = torch.from_numpy(mix_source)

        # Normalize audio
        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)

        # Extract filename
        filename = self.mix[idx][0].split("/")[-1]

        return mixture, source, source_mouth, filename


class AVSpeechDataModule(object):
    """
    DataModule for Audio-Visual Speech Enhancement
    """
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 1,
        sample_rate: int = 16000,
        segment: float = 2.0,
        normalize_audio: bool = False,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()

        if train_dir is None or valid_dir is None or test_dir is None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        """Setup train/val/test datasets"""
        self.data_train = AVSpeechDataset(
            json_dir=self.train_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=True
        )
        self.data_val = AVSpeechDataset(
            json_dir=self.valid_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=False
        )
        self.data_test = AVSpeechDataset(
            json_dir=self.test_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,  # Full audio for test
            normalize_audio=self.normalize_audio,
            is_train=False
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=1,  # Test one by one
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def make_loader(self):
        """Return train, val, test loaders"""
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        """Return train, val, test datasets"""
        return self.data_train, self.data_val, self.data_test
