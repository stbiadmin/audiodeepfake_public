"""
Improved Voice Isolation Utility

This script implements multiple advanced voice isolation techniques including:
1. Center channel extraction for stereo recordings
2. REPET-SIM method (librosa's official vocal separation)
3. Spectral subtraction for noise reduction
4. Harmonic-percussive separation with improved parameters

Based on proven signal processing methods from librosa documentation and research.
"""

import os
import librosa
import numpy as np
import soundfile as sf
from typing import List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Advanced separation libraries (optional imports)
try:
    import demucs.separate
    from demucs.pretrained import get_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    
try:
    from audio_separator.separator import Separator
    AUDIO_SEPARATOR_AVAILABLE = True
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False


class AdvancedVoiceIsolator:
    """
    Advanced voice isolation using multiple complementary techniques.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the AdvancedVoiceIsolator with configuration settings.
        
        Args:
            config: Dictionary of configuration parameters
        """
        # Default configuration optimized for voice isolation
        self.config = {
            # Approach selection
            'approach': 'librosa',     # Options: 'librosa', 'demucs', 'audio_separator'
            
            # Librosa parameters
            'sample_rate': 44100,      # Higher sample rate for better quality
            'hop_length': 512,         # Standard hop length
            'n_fft': 2048,            # FFT window size
            'win_length': 2048,       # Window length
            
            # Center channel extraction
            'center_isolation_strength': 1.0,  # Strength of center channel isolation
            
            # REPET-SIM parameters (from librosa documentation)
            'repet_filter_size': (1, 2048),   # Median filter size
            'repet_margin': (1, 1),           # Margin for soft masking
            'repet_power': 2,                 # Power for soft masking
            'repet_split_zeros': True,        # Handle zeros in filtering
            
            # Harmonic-percussive separation
            'hp_margin_h': 4,         # Harmonic margin
            'hp_margin_v': 2,         # Percussive (vocal) margin
            
            # Spectral subtraction
            'noise_reduction': 0.3,   # Noise reduction factor
            'spectral_floor': 0.1,    # Spectral floor to prevent artifacts
            
            # Post-processing
            'normalize_output': True,  # Normalize final output
            'output_gain': 0.8,       # Output gain factor
            
            # Demucs parameters
            'demucs_model': 'htdemucs',  # Model: 'htdemucs', 'htdemucs_ft', 'hdemucs_mmi', etc.
            'demucs_device': 'auto',     # Device: 'auto', 'cpu', 'cuda' (auto detects GPU)
            
            # Audio Separator parameters
            'separator_model': 'UVR-MDX-NET-Inst_HQ_3',  # Model name
            'separator_output_format': 'wav',  # Output format
        }
        
        # Update with user configuration
        if config:
            self.config.update(config)
            
        # Initialize approach-specific components
        self._initialize_approach()
    
    def load_audio(self, file_path: str) -> Tuple[Union[np.ndarray, None], Union[int, None], bool]:
        """
        Load audio file, preserving stereo if available.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate, is_stereo)
        """
        try:
            # Load without forcing mono to preserve stereo information
            audio, sr = librosa.load(file_path, sr=self.config['sample_rate'], mono=False)
            
            # Check if stereo
            is_stereo = len(audio.shape) == 2 and audio.shape[0] == 2
            
            if is_stereo:
                print(f"Loaded stereo file: {file_path} ({len(audio[0])/sr:.2f}s, {sr}Hz)")
            else:
                print(f"Loaded mono file: {file_path} ({len(audio)/sr:.2f}s, {sr}Hz)")
                # Ensure mono audio is 1D
                if len(audio.shape) == 2:
                    audio = audio.flatten()
            
            return audio, sr, is_stereo
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None, False
    
    def center_channel_extraction(self, stereo_audio: np.ndarray) -> np.ndarray:
        """
        Extract center channel (vocals) from stereo audio.
        This works well when vocals are mixed in the center of the stereo field.
        
        Args:
            stereo_audio: Stereo audio array (2, N)
            
        Returns:
            Center channel (vocals) as mono audio
        """
        if len(stereo_audio.shape) != 2 or stereo_audio.shape[0] != 2:
            raise ValueError("Input must be stereo audio (2, N)")
        
        left_channel = stereo_audio[0]
        right_channel = stereo_audio[1]
        
        # Method 1: Simple center extraction (vocals)
        # Vocals are typically in both channels (center), so we add them
        center = (left_channel + right_channel) / 2
        
        # Method 2: Side information (instruments are often panned)
        # Subtract to get difference signal (usually instruments)
        sides = (left_channel - right_channel) / 2
        
        # Combine: emphasize center, reduce sides
        strength = self.config['center_isolation_strength']
        vocals_enhanced = center + (sides * (1 - strength))
        
        return vocals_enhanced
    
    def repet_sim_vocal_separation(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        REPET-SIM vocal separation based on librosa's official implementation.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (vocals, accompaniment)
        """
        # Compute the short-time Fourier transform
        stft = librosa.stft(
            audio, 
            hop_length=self.config['hop_length'],
            n_fft=self.config['n_fft'],
            win_length=self.config['win_length']
        )
        
        # Decompose to magnitude and phase
        magnitude, phase = librosa.magphase(stft)
        
        # Apply nearest-neighbor filtering
        # This identifies and removes repeating background (accompaniment)
        # Calculate safe width based on the spectrogram size
        max_width = (magnitude.shape[-1] - 1) // 2
        desired_width = int(librosa.time_to_frames(2, sr=self.config['sample_rate']))
        safe_width = max(1, min(desired_width, max_width))
        
        accompaniment_mag = librosa.decompose.nn_filter(
            magnitude,
            aggregate=np.median,
            metric='cosine',
            width=safe_width,
        )
        
        # Create masks using soft masking
        # Ensure non-negative values for soft masking
        vocals_component = np.maximum(magnitude - accompaniment_mag, 0.01 * magnitude)
        accompaniment_component = np.maximum(accompaniment_mag, 0.01 * magnitude)
        
        vocals_mask = librosa.util.softmask(
            vocals_component,
            accompaniment_component,
            power=self.config['repet_power']
        )
        
        accompaniment_mask = librosa.util.softmask(
            accompaniment_component,
            vocals_component,
            power=self.config['repet_power']
        )
        
        # Apply masks
        vocals_stft = vocals_mask * stft
        accompaniment_stft = accompaniment_mask * stft
        
        # Convert back to time domain
        vocals = librosa.istft(
            vocals_stft, 
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length']
        )
        
        accompaniment = librosa.istft(
            accompaniment_stft, 
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length']
        )
        
        return vocals, accompaniment
    
    def harmonic_percussive_vocal_extraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract vocals using improved harmonic-percussive separation.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Vocal component
        """
        # Separate harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(
            audio,
            margin=(self.config['hp_margin_h'], self.config['hp_margin_v'])
        )
        
        # Vocals are typically in the harmonic component
        # But may have some percussive elements (consonants)
        # Combine with emphasis on harmonic
        vocals = 0.8 * harmonic + 0.2 * percussive
        
        return vocals
    
    def spectral_subtraction_denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction to reduce background noise.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Denoised audio
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            hop_length=self.config['hop_length'],
            n_fft=self.config['n_fft']
        )
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from the quieter portions
        # Use lower percentile as noise estimate
        noise_magnitude = np.percentile(magnitude, 20, axis=1, keepdims=True)
        
        # Spectral subtraction
        clean_magnitude = magnitude - self.config['noise_reduction'] * noise_magnitude
        
        # Apply spectral floor to prevent artifacts
        spectral_floor = self.config['spectral_floor'] * magnitude
        clean_magnitude = np.maximum(clean_magnitude, spectral_floor)
        
        # Reconstruct
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(
            clean_stft,
            hop_length=self.config['hop_length']
        )
        
        return clean_audio
    
    def isolate_vocals_combined(self, audio: np.ndarray, is_stereo: bool = False) -> np.ndarray:
        """
        Main vocal isolation pipeline combining multiple techniques.
        
        Args:
            audio: Input audio (mono or stereo)
            is_stereo: Whether input is stereo
            
        Returns:
            Isolated vocals
        """
        print("  - Starting vocal isolation pipeline...")
        
        # Step 1: Handle stereo audio with center channel extraction
        if is_stereo:
            print("  - Applying center channel extraction...")
            mono_audio = self.center_channel_extraction(audio)
        else:
            mono_audio = audio
        
        # Step 2: Apply REPET-SIM vocal separation
        print("  - Applying REPET-SIM vocal separation...")
        try:
            vocals_repet, accompaniment = self.repet_sim_vocal_separation(mono_audio)
        except Exception as e:
            print(f"    REPET-SIM failed: {e}, falling back to harmonic-percussive")
            vocals_repet = self.harmonic_percussive_vocal_extraction(mono_audio)
        
        # Step 3: Apply harmonic-percussive separation for refinement
        print("  - Refining with harmonic-percussive separation...")
        vocals_hp = self.harmonic_percussive_vocal_extraction(vocals_repet)
        
        # Step 4: Apply spectral subtraction for noise reduction
        print("  - Applying spectral subtraction denoising...")
        vocals_clean = self.spectral_subtraction_denoise(vocals_hp)
        
        # Step 5: Post-processing
        if self.config['normalize_output'] and np.max(np.abs(vocals_clean)) > 0:
            vocals_clean = vocals_clean / np.max(np.abs(vocals_clean)) * self.config['output_gain']
        
        return vocals_clean
    
    def _initialize_approach(self):
        """
        Initialize the selected approach.
        """
        approach = self.config['approach'].lower()
        
        if approach == 'demucs':
            if not DEMUCS_AVAILABLE:
                raise ImportError("Demucs not available. Install with: pip install demucs")
            
            # Auto-detect GPU availability for Demucs
            if self.config['demucs_device'] == 'auto':
                import torch
                self.config['demucs_device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            print(f"Initialized Demucs approach with model: {self.config['demucs_model']}, device: {self.config['demucs_device']}")
            
        elif approach == 'audio_separator':
            if not AUDIO_SEPARATOR_AVAILABLE:
                raise ImportError("Audio Separator not available. Install with: pip install audio-separator")
            
            # Check for GPU availability and enable CUDA if available
            import torch
            use_cuda = torch.cuda.is_available()
            
            # Initialize separator with GPU support if available
            separator_kwargs = {
                'output_format': self.config['separator_output_format']
            }
            
            self.separator = Separator(**separator_kwargs)
            print(f"Initialized Audio Separator with output format: {self.config['separator_output_format']}, CUDA: {use_cuda}")
            
        elif approach == 'librosa':
            print("Initialized Librosa approach")
            
        else:
            raise ValueError(f"Unknown approach: {approach}. Options: 'librosa', 'demucs', 'audio_separator'")
    
    def isolate_vocals_demucs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Vocal isolation using Demucs.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Isolated vocals
        """
        print("  - Using Demucs for vocal isolation...")
        
        try:
            # Load Demucs model
            import torch
            model = get_model(self.config['demucs_model'])
            model.eval()
            
            # Move model to appropriate device
            device = self.config['demucs_device']
            if device == 'cuda' and torch.cuda.is_available():
                model = model.cuda()
            elif device == 'cpu':
                model = model.cpu()
            
            # Convert to torch tensor and add batch dimension
            import torch
            from demucs.apply import apply_model
            
            if len(audio.shape) == 1:
                # Mono to stereo for Demucs
                audio_stereo = np.stack([audio, audio])
            else:
                audio_stereo = audio
                
            audio_tensor = torch.tensor(audio_stereo, dtype=torch.float32).unsqueeze(0)
            
            # Move tensor to same device as model
            device = self.config['demucs_device']
            if device == 'cuda' and torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # Separate audio using apply_model
            with torch.no_grad():
                sources = apply_model(model, audio_tensor)
            
            # Extract vocals (typically index 3: drums, bass, other, vocals)
            if sources.shape[1] >= 4:
                vocals = sources[0, 3].mean(dim=0).cpu().numpy()  # Average stereo channels, move to CPU
            else:
                # Fallback if model output is different
                vocals = sources[0, -1].mean(dim=0).cpu().numpy()
            
            return vocals
            
        except Exception as e:
            print(f"    Demucs failed: {e}, falling back to librosa")
            return self.isolate_vocals_combined(audio, len(audio.shape) == 2 and audio.shape[0] == 2)
    
    def isolate_vocals_audio_separator(self, file_path: str, output_dir: str) -> str:
        """
        Vocal isolation using Audio Separator.
        
        Args:
            file_path: Path to input audio file
            output_dir: Output directory
            
        Returns:
            Path to isolated vocals file
        """
        print("  - Using Audio Separator for vocal isolation...")
        
        # Try multiple models in case one fails
        model_candidates = [
            self.config['separator_model'],
            'UVR_MDXNET_KARA_2.onnx',
            'Kim_Vocal_2.onnx', 
            'UVR-MDX-NET-Inst_HQ_4.onnx',
            'UVR_MDXNET_Main.onnx'
        ]
        
        for model_name in model_candidates:
            try:
                print(f"    Trying model: {model_name}")
                # Load the model and separate audio
                self.separator.load_model(model_filename=model_name)
                output_files = self.separator.separate(file_path)
                
                # Find vocals file (usually contains 'Vocals' in name)
                vocals_file = None
                for file in output_files:
                    if 'vocals' in file.lower() or 'vocal' in file.lower():
                        vocals_file = file
                        break
                
                if not vocals_file:
                    # If no vocals file found, take first output
                    vocals_file = output_files[0] if output_files else None
                
                if vocals_file:
                    print(f"    Success with model: {model_name}")
                    return vocals_file
                    
            except Exception as e:
                print(f"    Model {model_name} failed: {e}")
                continue
        
        print("    All Audio Separator models failed")
        return None
    
    def isolate_vocals_adaptive(self, audio: np.ndarray, sr: int, file_path: str = None) -> np.ndarray:
        """
        Main vocal isolation using the configured approach.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            file_path: Original file path (needed for audio_separator)
            
        Returns:
            Isolated vocals
        """
        approach = self.config['approach'].lower()
        
        if approach == 'demucs':
            return self.isolate_vocals_demucs(audio, sr)
            
        elif approach == 'audio_separator' and file_path:
            # Audio separator works with files, not arrays
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input = os.path.join(temp_dir, 'input.wav')
                
                # Handle both mono and stereo audio correctly
                if len(audio.shape) == 2 and audio.shape[0] == 2:
                    # Stereo audio - transpose to (samples, channels)
                    audio_to_write = audio.T
                else:
                    # Mono audio
                    audio_to_write = audio
                
                # Write with proper format
                sf.write(temp_input, audio_to_write, sr, subtype='PCM_16')
                
                vocals_file = self.isolate_vocals_audio_separator(temp_input, temp_dir)
                
                if vocals_file and os.path.exists(vocals_file):
                    vocals, _ = librosa.load(vocals_file, sr=sr)
                    return vocals
                else:
                    print("    Audio Separator failed, falling back to librosa")
                    return self.isolate_vocals_combined(audio, len(audio.shape) == 2 and audio.shape[0] == 2)
        
        else:
            # Default to librosa approach
            return self.isolate_vocals_combined(audio, len(audio.shape) == 2 and audio.shape[0] == 2)
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Process a single file to isolate vocals using advanced techniques.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save isolated vocals
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Processing: {os.path.basename(input_path)}")
            
            # Load audio
            audio, sr, is_stereo = self.load_audio(input_path)
            if audio is None:
                return False
            
            # Isolate vocals using selected approach
            vocals_isolated = self.isolate_vocals_adaptive(audio, sr, input_path)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save isolated vocals
            sf.write(output_path, vocals_isolated, sr)
            print(f"  Saved: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False


def find_audio_files(directory: str) -> List[str]:
    """
    Find all audio files in a directory and subdirectories.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of audio file paths
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    return sorted(audio_files)


def main():
    """
    Main function with advanced vocal isolation processing.
    """
    
    #
    # CONFIGURATION - Modify these paths as needed
    #
    # Determine the correct paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from utilities/
    
    # Source directory containing music files to process
    SOURCE_DIR = os.path.join(project_root, "data", "music-samples")
    
    # Destination directory for isolated vocal tracks
    DEST_DIR = os.path.join(project_root, "data", "extracted-vocals")
    
    # Advanced processing configuration
    config = {
        #
        # APPROACH SELECTION - Choose your voice isolation method
        #
        'approach': 'audio_separator',              # Options: 'librosa', 'demucs', 'audio_separator'
        
        #
        # LIBROSA PARAMETERS (used when approach='librosa')
        #
        'sample_rate': 44100,               # Higher quality sample rate
        'hop_length': 512,                  # Standard hop length
        'n_fft': 2048,                     # FFT size
        'win_length': 2048,                # Window length
        
        # Center channel extraction (for stereo files)
        'center_isolation_strength': 0.8,   # How much to isolate center vs sides
        
        # REPET-SIM parameters
        'repet_power': 2,                   # Soft masking power
        
        # Harmonic-percussive separation
        'hp_margin_h': 4,                   # Harmonic margin
        'hp_margin_v': 2,                   # Percussive margin
        
        # Spectral subtraction
        'noise_reduction': 0.4,             # Noise reduction strength
        'spectral_floor': 0.15,             # Prevent over-subtraction
        
        # Output settings
        'normalize_output': True,           # Normalize output
        'output_gain': 0.8,                # Output gain
        
        #
        # DEMUCS PARAMETERS (used when approach='demucs')
        #
        'demucs_model': 'htdemucs',         # Model: 'htdemucs', 'htdemucs_ft', 'hdemucs_mmi'
        'demucs_device': 'auto',            # Device: 'auto', 'cpu', 'cuda' (auto detects GPU)
        
        #
        # AUDIO SEPARATOR PARAMETERS (used when approach='audio_separator')
        #
        'separator_model': 'UVR-MDX-NET-Inst_HQ_3.onnx',  # Pre-trained model
        'separator_output_format': 'wav',   # Output format
    }
    
    #
    # PROCESSING
    #
    print("Advanced Voice Isolation Utility")
    print("=" * 60)
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Destination directory: {DEST_DIR}")
    approach_name = {
        'librosa': 'Librosa (Center Channel + REPET-SIM + Harmonic-Percussive + Spectral Subtraction)',
        'demucs': 'Demucs (Deep Learning Source Separation)',
        'audio_separator': 'Audio Separator (Multiple Pre-trained Models)'
    }.get(config['approach'], 'Unknown')
    print(f"Approach: {approach_name}")
    print()
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        print("Please create the directory and add audio files to process.")
        return
    
    # Find audio files
    audio_files = find_audio_files(SOURCE_DIR)
    
    if not audio_files:
        print(f"No audio files found in '{SOURCE_DIR}'.")
        print("Supported formats: .wav, .mp3, .flac, .m4a, .aac, .ogg")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process:")
    for file in audio_files:
        print(f"  - {os.path.relpath(file, SOURCE_DIR)}")
    print()
    
    # Initialize advanced voice isolator
    isolator = AdvancedVoiceIsolator(config)
    
    # Process each file
    successful = 0
    failed = 0
    
    for input_file in audio_files:
        # Generate output path maintaining directory structure
        rel_path = os.path.relpath(input_file, SOURCE_DIR)
        output_file = os.path.join(DEST_DIR, rel_path)
        
        # Add approach suffix to filename
        approach_suffix = {
            'librosa': '_vocals_librosa',
            'demucs': '_vocals_demucs', 
            'audio_separator': '_vocals_separator'
        }.get(config['approach'], '_vocals_advanced')
        
        output_file = os.path.splitext(output_file)[0] + approach_suffix + '.wav'
        
        # Process file
        if isolator.process_file(input_file, output_file):
            successful += 1
        else:
            failed += 1
        print()
    
    # Summary
    print("Processing Complete")
    print("=" * 60)
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Output directory: {DEST_DIR}")
    print()
    if config['approach'] == 'librosa':
        print("Techniques used:")
        print("1. Center channel extraction (for stereo files)")
        print("2. REPET-SIM vocal separation (librosa official method)")
        print("3. Harmonic-percussive separation refinement")
        print("4. Spectral subtraction denoising")
    elif config['approach'] == 'demucs':
        print("Model used:")
        print(f"- Demucs model: {config['demucs_model']}")
        print(f"- Device: {config['demucs_device']}")
    elif config['approach'] == 'audio_separator':
        print("Model used:")
        print(f"- Audio Separator model: {config['separator_model']}")
        print(f"- Output format: {config['separator_output_format']}")
    
    print("\nNote: If the selected approach fails, the system will fall back to librosa.")


if __name__ == "__main__":
    main()