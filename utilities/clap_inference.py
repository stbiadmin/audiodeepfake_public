"""
Audio Embedding Utilities
A module for extracting embeddings from audio files and text using LAION CLAP.
"""

import numpy as np
import librosa
import torch
import laion_clap
from typing import List, Union, Tuple, Dict, Any, Optional


class AudioEmbeddingExtractor:
    """
    A class for extracting embeddings from audio files and text using LAION CLAP.
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
            'enable_fusion': False,
            'sample_rate': 48000,
            'use_tensor': False,
            'amodel': 'laion_clap',  # Future-proofing for other models
        }

        # Update configuration with user-provided values
        if config:
            self.config.update(config)

        # Initialize model
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model based on the configuration."""
        if self.config['amodel'] == 'laion_clap':
            self.model = laion_clap.CLAP_Module(enable_fusion=self.config['enable_fusion'])
            self.model.load_ckpt()  # Download the default pretrained checkpoint
        else:
            raise ValueError(f"Unsupported model: {self.config['amodel']}")

    def update_config(self, new_config: Dict[str, Any]):
        """
        Update the configuration settings.

        Args:
            new_config: A dictionary of configuration settings to update.
        """
        old_model = self.config.get('amodel')
        self.config.update(new_config)

        # Reload model if the model type has changed
        if old_model != self.config.get('amodel'):
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

    def extract_from_audio_files(self, file_paths: List[str]) -> np.ndarray:
        """
        Extract embeddings directly from audio files.

        Args:
            file_paths: List of paths to audio files.

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
        """
        return self.model.get_audio_embedding_from_filelist(
            x=file_paths,
            use_tensor=self.config['use_tensor']
        )

    def extract_from_audio_data(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract embeddings from audio data.

        Args:
            audio_data: Audio data as numpy array or torch tensor.

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
        """
        # Ensure audio data has shape (N, T)
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)

        # Convert to tensor if needed
        if isinstance(audio_data, np.ndarray) and self.config['use_tensor']:
            audio_data = torch.from_numpy(audio_data).float()

        return self.model.get_audio_embedding_from_data(
            x=audio_data,
            use_tensor=self.config['use_tensor']
        )

    def extract_from_audio_file_with_preprocessing(self, file_path: str, quantize: bool = False) -> np.ndarray:
        """
        Load audio file, preprocess, and extract embeddings.

        Args:
            file_path: Path to audio file.
            quantize: Whether to apply quantization.

        Returns:
            Audio embeddings as numpy array or torch tensor depending on config.
        """
        # Load audio file
        audio_data, _ = librosa.load(file_path, sr=self.config['sample_rate'])

        # Reshape to (1, T)
        audio_data = audio_data.reshape(1, -1)

        # Apply quantization if requested
        if quantize:
            if self.config['use_tensor']:
                audio_data = torch.from_numpy(
                    self.int16_to_float32(
                        self.float32_to_int16(audio_data)
                    )
                ).float()
            else:
                audio_data = self.int16_to_float32(
                    self.float32_to_int16(audio_data)
                )
        elif self.config['use_tensor']:
            audio_data = torch.from_numpy(audio_data).float()

        return self.model.get_audio_embedding_from_data(
            x=audio_data,
            use_tensor=self.config['use_tensor']
        )

    def extract_from_text(self, texts: List[str]) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embeddings from text.

        Args:
            texts: List of text strings.

        Returns:
            Text embeddings as numpy array or torch tensor depending on config.
        """
        return self.model.get_text_embedding(
            texts,
            use_tensor=self.config['use_tensor']
        )


