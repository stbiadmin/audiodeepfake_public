"""
Audio Embedding Utilities
A module for extracting embeddings from audio files and text using Microsoft CLAP.
"""

import numpy as np
import librosa
import torch
from msclap import CLAP
from typing import List, Union, Tuple, Dict, Any, Optional


class AudioEmbeddingExtractor:
    """
    A class for extracting embeddings from audio files and text using Microsoft CLAP.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AudioEmbeddingExtractor with configuration settings.

        Args:
            config: A dictionary of configuration settings.
                   Default configuration if None is provided.
        """
        # Default configuration
        self.config = {
            'version': '2023',
            'use_cuda': False,
            'use_mps': True,  # Use Apple Silicon MPS by default
            'sample_rate': 48000,
            'use_tensor': False,
            'amodel': 'msclap',  # Future-proofing for other models
        }

        # Update configuration with user-provided values
        if config:
            self.config.update(config)

        # Determine device
        if self.config['use_cuda'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif self.config['use_mps'] and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Initialize model
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model based on the configuration."""
        if self.config['amodel'] == 'msclap':
            self.model = CLAP(
                version=self.config['version'],
                use_cuda=self.config['use_cuda']
            )
            # Move model to MPS if available and requested
            if self.device.type == 'mps':
                self.model.clap = self.model.clap.to(self.device)
                # Patch the _get_audio_embeddings method to move inputs to device
                original_get_audio_embeddings = self.model._get_audio_embeddings
                device = self.device

                def patched_get_audio_embeddings(preprocessed_audio):
                    with torch.no_grad():
                        preprocessed_audio = preprocessed_audio.reshape(
                            preprocessed_audio.shape[0], preprocessed_audio.shape[2])
                        preprocessed_audio = preprocessed_audio.to(device)
                        return self.model.clap.audio_encoder(preprocessed_audio)[0]

                self.model._get_audio_embeddings = patched_get_audio_embeddings
                print(f"MS-CLAP model moved to {self.device}")
        else:
            raise ValueError(f"Unsupported model: {self.config['amodel']}")

    def update_config(self, new_config: Dict[str, Any]):
        """
        Update the configuration settings.

        Args:
            new_config: A dictionary of configuration settings to update.
        """
        old_model = self.config.get('amodel')
        old_version = self.config.get('version')
        old_cuda = self.config.get('use_cuda')

        self.config.update(new_config)

        # Reload model if the model type, version, or cuda setting has changed
        if (old_model != self.config.get('amodel') or
            old_version != self.config.get('version') or
            old_cuda != self.config.get('use_cuda')):
            self._load_model()

    @staticmethod
    def int16_to_float32(x: np.ndarray) -> np.ndarray:
        """
        Convert 16-bit integer values to 32-bit float values.

        Args:
            x: Array of 16-bit integer values.

        Returns:
            Array of 32-bit float values.
        """
        return (x / 32767.0).astype(np.float32)

    @staticmethod
    def float32_to_int16(x: np.ndarray) -> np.ndarray:
        """
        Convert 32-bit float values to 16-bit integer values.

        Args:
            x: Array of 32-bit float values.

        Returns:
            Array of 16-bit integer values.
        """
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

    def extract_from_audio_files(self, file_paths: List[str]) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embeddings directly from audio files.

        Args:
            file_paths: List of paths to audio files.

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
        """
        embeddings = self.model.get_audio_embeddings(file_paths)

        # Handle tensor/numpy conversion based on config
        if self.config['use_tensor']:
            if not torch.is_tensor(embeddings):
                embeddings = torch.from_numpy(embeddings).float()
        else:
            if torch.is_tensor(embeddings):
                embeddings = embeddings.detach().cpu().numpy()

        return embeddings

    def extract_from_audio_data(self, audio_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embeddings from audio data directly (no temp file I/O).

        Args:
            audio_data: Audio data as numpy array or torch tensor.
                       Shape can be (T,) for single audio or (N, T) for batch.
                       Expected sample rate: self.config['sample_rate'] (48000 Hz)

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
            Shape: (N, embedding_dim) where N is the batch size.
        """
        # Ensure audio data is numpy array
        if torch.is_tensor(audio_data):
            audio_np = audio_data.detach().cpu().numpy()
        else:
            audio_np = audio_data

        # Handle both single audio (T,) and batch (N, T)
        if len(audio_np.shape) == 1:
            audio_np = audio_np.reshape(1, -1)

        n_samples = audio_np.shape[0]
        sample_rate = self.config['sample_rate']
        target_duration = self.model.args.duration  # Usually 7 seconds for MS-CLAP
        target_length = int(target_duration * sample_rate)

        # Preprocess each audio segment to target length
        audio_tensors = []
        for i in range(n_samples):
            audio = audio_np[i].reshape(-1)

            # Pad or trim to target length (same logic as MS-CLAP's load_audio_into_tensor)
            if target_length >= len(audio):
                # Repeat to fill
                repeat_factor = int(np.ceil(target_length / len(audio)))
                audio = np.tile(audio, repeat_factor)[:target_length]
            else:
                # Trim from start (deterministic, no random)
                audio = audio[:target_length]

            audio_tensors.append(torch.FloatTensor(audio).reshape(1, -1))

        # Stack into batch: (N, 1, T)
        preprocessed_audio = torch.stack(audio_tensors, dim=0)

        # Move to device and extract embeddings
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(n_samples, -1).to(self.device)
            embeddings = self.model.clap.audio_encoder(preprocessed_audio)[0]

        # Handle tensor/numpy conversion based on config
        if self.config['use_tensor']:
            return embeddings
        else:
            return embeddings.detach().cpu().numpy()

    def extract_from_audio_file_with_preprocessing(self, file_path: str, quantize: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Load audio file, preprocess, and extract embeddings.

        Args:
            file_path: Path to audio file.
            quantize: Whether to apply quantization.

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
        """
        if quantize:
            # Load audio file for preprocessing
            audio_data, _ = librosa.load(file_path, sr=self.config['sample_rate'])

            # Apply quantization
            audio_data = self.int16_to_float32(
                self.float32_to_int16(audio_data)
            )

            # Extract embeddings from processed audio data
            return self.extract_from_audio_data(audio_data)
        else:
            # Direct extraction from file
            return self.extract_from_audio_files([file_path])

    def extract_from_text(self, texts: List[str]) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embeddings from text.

        Args:
            texts: List of text strings.

        Returns:
            Text embeddings as numpy array or torch tensor depending on config.
        """
        embeddings = self.model.get_text_embeddings(texts)

        # Handle tensor/numpy conversion based on config
        if self.config['use_tensor']:
            if not torch.is_tensor(embeddings):
                embeddings = torch.from_numpy(embeddings).float()
        else:
            if torch.is_tensor(embeddings):
                embeddings = embeddings.detach().cpu().numpy()

        return embeddings