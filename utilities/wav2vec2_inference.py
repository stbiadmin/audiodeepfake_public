"""
Wav2Vec2 Audio Embedding Utilities
A module for extracting embeddings from audio using Facebook Wav2Vec2.

Wav2Vec2-XLS-R is a multilingual model trained on 128 languages,
making it useful for diverse audio deepfake detection.

References:
    - Model: https://huggingface.co/facebook/wav2vec2-xls-r-300m
    - Paper: https://arxiv.org/abs/2111.09296
"""

from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import torch


class AudioEmbeddingExtractor:
    """
    A class for extracting embeddings from audio using Facebook Wav2Vec2.

    Wav2Vec2-XLS-R is trained on 128 languages, providing good multilingual
    coverage for diverse deepfake detection scenarios.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AudioEmbeddingExtractor with configuration settings.

        Args:
            config: A dictionary of configuration settings.
                   Default configuration if None is provided.
        """
        # Default configuration
        # NOTE: MPS disabled due to memory leak - CPU is stable
        self.config = {
            'model_name': 'facebook/wav2vec2-xls-r-300m',
            'use_cuda': False,
            'use_mps': False,  # MPS has memory leak, use CPU instead
            'sample_rate': 16000,  # Wav2Vec2 expects 16kHz
            'use_tensor': False,
            'pooling': 'mean',  # 'mean', 'max', or 'last'
            'layer': -1,  # Which layer to use (-1 = last hidden state)
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
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the Wav2Vec2 model and processor."""
        try:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        model_name = self.config['model_name']
        print(f"Loading Wav2Vec2 model: {model_name}")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Wav2Vec2 model loaded on {self.device}")
        print(f"  Hidden size: {self.model.config.hidden_size}")
        print(f"  Num layers: {self.model.config.num_hidden_layers}")

    def update_config(self, new_config: Dict[str, Any]):
        """
        Update the configuration settings.

        Args:
            new_config: A dictionary of configuration settings to update.
        """
        old_model = self.config.get('model_name')

        self.config.update(new_config)

        # Reload model if the model name changed
        if old_model != self.config.get('model_name'):
            self._load_model()

    @staticmethod
    def int16_to_float32(x: np.ndarray) -> np.ndarray:
        """Convert 16-bit integer values to 32-bit float values."""
        return (x / 32767.0).astype(np.float32)

    @staticmethod
    def float32_to_int16(x: np.ndarray) -> np.ndarray:
        """Convert 32-bit float values to 16-bit integer values."""
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

    def _pool_embeddings(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden states across time dimension.

        Args:
            hidden_states: Shape (batch, time, hidden_size)

        Returns:
            Pooled embeddings of shape (batch, hidden_size)
        """
        pooling = self.config['pooling']

        if pooling == 'mean':
            return hidden_states.mean(dim=1)
        elif pooling == 'max':
            return hidden_states.max(dim=1)[0]
        elif pooling == 'last':
            return hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def extract_from_audio_files(self, file_paths: List[str]) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embeddings directly from audio files.

        Args:
            file_paths: List of paths to audio files.

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
            Shape: (N, hidden_size) where N is number of files
        """
        embeddings_list = []

        for file_path in file_paths:
            # Load and resample audio
            audio, sr = librosa.load(file_path, sr=self.config['sample_rate'])
            emb = self.extract_from_audio_data(audio)
            embeddings_list.append(emb)

        embeddings = np.vstack(embeddings_list)

        if self.config['use_tensor']:
            return torch.from_numpy(embeddings).float()
        return embeddings

    def extract_from_audio_data(
        self,
        audio_data: Union[np.ndarray, torch.Tensor],
        sr: int = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embeddings from audio data directly.

        Args:
            audio_data: Audio data as numpy array or torch tensor.
                       Shape can be (T,) for single audio or (N, T) for batch.
                       Expected sample rate: 16000 Hz (will resample if different)
            sr: Sample rate of input audio. If None, assumes 16000 Hz.

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
            Shape: (N, hidden_size) where N is the batch size.
        """
        # Ensure audio data is numpy array
        if torch.is_tensor(audio_data):
            audio_np = audio_data.detach().cpu().numpy()
        else:
            audio_np = audio_data

        # Handle both single audio (T,) and batch (N, T)
        if len(audio_np.shape) == 1:
            audio_np = audio_np.reshape(1, -1)

        # Resample if needed
        target_sr = self.config['sample_rate']
        if sr is not None and sr != target_sr:
            resampled = []
            for i in range(audio_np.shape[0]):
                audio_resampled = librosa.resample(
                    audio_np[i], orig_sr=sr, target_sr=target_sr
                )
                resampled.append(audio_resampled)
            # Pad to same length
            max_len = max(len(a) for a in resampled)
            audio_np = np.zeros((len(resampled), max_len), dtype=np.float32)
            for i, a in enumerate(resampled):
                audio_np[i, :len(a)] = a

        # Process with feature extractor
        # Wav2Vec2 processor expects list of arrays
        audio_list = [audio_np[i] for i in range(audio_np.shape[0])]

        inputs = self.processor(
            audio_list,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        input_values = inputs.input_values.to(self.device)

        # Delete processor output to free memory
        del inputs, audio_list

        # Extract embeddings
        with torch.no_grad():
            # Only get last hidden state (not all layers) to save memory
            outputs = self.model(input_values, output_hidden_states=False)
            hidden_states = outputs.last_hidden_state

            # Pool across time
            embeddings = self._pool_embeddings(hidden_states)

            # Move to CPU immediately and convert to numpy
            result = embeddings.detach().cpu().numpy()

            # Explicitly delete GPU tensors
            del outputs, hidden_states, embeddings, input_values

        # Aggressive memory cleanup for MPS
        if self.device.type == 'mps':
            torch.mps.empty_cache()

        # Force garbage collection
        import gc
        gc.collect()

        # Handle tensor/numpy conversion based on config
        if self.config['use_tensor']:
            return torch.from_numpy(result)
        else:
            return result

    def extract_from_audio_file_with_preprocessing(
        self,
        file_path: str,
        quantize: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Load audio file, preprocess, and extract embeddings.

        Args:
            file_path: Path to audio file.
            quantize: Whether to apply quantization.

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
        """
        # Load audio file
        audio_data, sr = librosa.load(file_path, sr=self.config['sample_rate'])

        if quantize:
            # Apply quantization
            audio_data = self.int16_to_float32(
                self.float32_to_int16(audio_data)
            )

        return self.extract_from_audio_data(audio_data)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.config.hidden_size
