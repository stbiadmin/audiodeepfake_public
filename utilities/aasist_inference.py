"""AASIST Audio Embedding Extractor.

Extracts penultimate-layer embeddings from AASIST for use in the temporal
coherence pipeline. Uses a forward hook to capture the graph pooling readout
(before the final classification linear layer).

Matches the AudioEmbeddingExtractor interface from wavlm_inference.py.

Requirements:
    - AASIST repo cloned to vendor/aasist/
    - einops package installed

Usage:
    from utilities.aasist_inference import AudioEmbeddingExtractor
    extractor = AudioEmbeddingExtractor()
    embeddings = extractor.extract_from_audio_data(audio_data, sr=16000)
"""

import numpy as np
import librosa
import torch
import sys
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class AudioEmbeddingExtractor:
    """Extract penultimate-layer embeddings from AASIST.

    Uses a forward hook on the final classification layer to capture
    the graph pooling readout - the learned spectro-temporal representation
    before final binary classification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'variant': 'AASIST',  # 'AASIST' or 'AASIST-L'
            'use_cuda': False,
            'use_mps': True,  # Use MPS on Apple Silicon by default
            'sample_rate': 16000,
            'use_tensor': False,
            'target_length': 64600,  # AASIST default input length
        }
        if config:
            self.config.update(config)

        # Determine device
        if self.config['use_cuda'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif self.config['use_mps'] and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = None
        self._embedding_dim = None
        self._hook_output = None
        self._hook_handle = None
        self._load_model()

    def _load_model(self):
        """Load AASIST model and register forward hook."""
        aasist_dir = PROJECT_ROOT / 'vendor' / 'aasist'
        if not aasist_dir.exists():
            raise FileNotFoundError(
                f"AASIST repo not found at {aasist_dir}. "
                f"Run: git clone https://github.com/clovaai/aasist.git {aasist_dir}"
            )

        # Import AASIST model directly via importlib to avoid conflict with project's models/ package
        import importlib.util
        spec = importlib.util.spec_from_file_location("AASIST", str(aasist_dir / "models" / "AASIST.py"))
        aasist_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aasist_module)
        AASISTModel = aasist_module.Model

        variant = self.config['variant']
        if variant == 'AASIST-L':
            config_path = aasist_dir / 'config' / 'AASIST-L.conf'
            weight_path = aasist_dir / 'models' / 'weights' / 'AASIST-L.pth'
        else:
            config_path = aasist_dir / 'config' / 'AASIST.conf'
            weight_path = aasist_dir / 'models' / 'weights' / 'AASIST.pth'

        if not weight_path.exists():
            raise FileNotFoundError(f"Weights not found: {weight_path}")

        import json
        with open(config_path, 'r') as f:
            model_config = json.load(f)['model_config']

        self.model = AASISTModel(model_config)
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Register forward hook on the last linear layer (output layer)
        # to capture its input (= the penultimate representation)
        self._register_hook()

        print(f"AASIST ({variant}) loaded on {self.device}")
        print(f"  Embedding dim: {self.embedding_dim}")

    def _register_hook(self):
        """Register forward hook on AASIST's output layer."""
        # AASIST's final layer is self.out_layer (nn.Linear)
        target_layer = None
        for name, module in self.model.named_modules():
            if name == 'out_layer' or (
                isinstance(module, torch.nn.Linear) and
                hasattr(module, 'out_features') and
                module.out_features == 2
            ):
                target_layer = module
                break

        if target_layer is None:
            # Fallback: find last Linear layer
            linear_layers = [(n, m) for n, m in self.model.named_modules()
                            if isinstance(m, torch.nn.Linear)]
            if linear_layers:
                target_layer = linear_layers[-1][1]

        if target_layer is None:
            raise RuntimeError("Could not find AASIST output layer for hook")

        self._embedding_dim = target_layer.in_features

        def hook_fn(module, input, output):
            # input is a tuple; first element is the embedding
            self._hook_output = input[0].detach()

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def _prepare_audio(self, audio_np):
        """Tile-pad/truncate audio to target length.

        Uses tile-padding (repeating audio) to match AASIST's training
        convention from data_utils.pad().
        """
        target_length = self.config['target_length']
        if len(audio_np) < target_length:
            num_repeats = int(target_length / len(audio_np)) + 1
            audio_np = np.tile(audio_np, num_repeats)[:target_length]
        else:
            audio_np = audio_np[:target_length]
        return audio_np

    def extract_from_audio_data(
        self,
        audio_data: Union[np.ndarray, torch.Tensor],
        sr: int = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Extract embeddings from audio data.

        Args:
            audio_data: Shape (T,) for single or (N, T) for batch.
            sr: Sample rate. If None, assumes 16000 Hz.

        Returns:
            Embeddings array of shape (N, embedding_dim).
        """
        if torch.is_tensor(audio_data):
            audio_np = audio_data.detach().cpu().numpy()
        else:
            audio_np = audio_data

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
            max_len = max(len(a) for a in resampled)
            audio_np = np.zeros((len(resampled), max_len), dtype=np.float32)
            for i, a in enumerate(resampled):
                audio_np[i, :len(a)] = a

        # Process each sample (pad/truncate to target length)
        embeddings = []
        for i in range(audio_np.shape[0]):
            audio_prepared = self._prepare_audio(audio_np[i])
            audio_tensor = torch.FloatTensor(audio_prepared).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _ = self.model(audio_tensor)
                embedding = self._hook_output.cpu().numpy()
                embeddings.append(embedding)

        result = np.vstack(embeddings)

        if self.config['use_tensor']:
            return torch.from_numpy(result)
        return result

    def extract_from_audio_files(self, file_paths: List[str]) -> Union[np.ndarray, torch.Tensor]:
        """Extract embeddings from audio files.

        Args:
            file_paths: List of paths to audio files.

        Returns:
            Embeddings array of shape (N, embedding_dim).
        """
        embeddings_list = []
        for file_path in file_paths:
            audio, sr = librosa.load(file_path, sr=self.config['sample_rate'])
            emb = self.extract_from_audio_data(audio)
            embeddings_list.append(emb)
        return np.vstack(embeddings_list)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        # Fallback: run a dummy forward pass
        dummy = torch.zeros(1, self.config['target_length']).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        if self._hook_output is not None:
            self._embedding_dim = self._hook_output.shape[-1]
        return self._embedding_dim or 160  # AASIST default

    def __del__(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
