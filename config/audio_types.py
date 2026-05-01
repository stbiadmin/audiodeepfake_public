"""Audio type configurations for different content types."""

from typing import Any, Dict, List

AUDIO_TYPE_CONFIGS: Dict[str, Dict[str, Any]] = {
    'single_voice': {
        'description': 'Single speaker voice/speech',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 1.5,  # Option C: shorter segments
        'segment_hop': 0.75,  # 50% overlap
        'min_segments': 2,  # Reduced to keep more samples
        'min_duration': 2.25,  # Minimum for 2 segments
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize', 'trim_silence'],
    },
    'single_voice_wild': {
        'description': 'Single speaker voice in the wild (with background noise)',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 1.5,  # Option C: shorter segments
        'segment_hop': 0.75,  # 50% overlap
        'min_segments': 2,  # Reduced to keep more samples
        'min_duration': 2.25,  # Minimum for 2 segments
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
    },
    'multi_voice': {
        'description': 'Multiple speakers (conversations, podcasts)',
        'use_diarization': True,
        'use_stem_separation': False,
        'segment_duration': 2.0,
        'min_duration': 10.0,
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
        'min_speakers': 2,
    },
    'multi_voice_wild': {
        'description': 'Multiple speakers in the wild',
        'use_diarization': True,
        'use_stem_separation': False,
        'segment_duration': 2.0,
        'min_duration': 10.0,
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
        'min_speakers': 2,
    },
    'music_instrumental': {
        'description': 'Instrumental music only (no vocals)',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 1.5,  # Shorter segments for MUSDB18 7s samples
        'min_duration': 5.0,      # Adjusted for sample dataset
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
    },
    'music_with_vocals': {
        'description': 'Music with singing/vocals',
        'use_diarization': False,
        'use_stem_separation': False,  # Disabled for now - stem separation not needed for full mixtures
        'segment_duration': 1.5,  # Shorter segments for MUSDB18 7s samples
        'min_duration': 5.0,      # Adjusted for sample dataset
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
    },
    'isolated_vocals': {
        'description': 'Vocals isolated from music',
        'use_diarization': False,
        'use_stem_separation': False,  # Already separated
        'segment_duration': 2.0,
        'min_duration': 10.0,
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
    },
    'isolated_instrument': {
        'description': 'Single instrument isolated from music',
        'use_diarization': False,
        'use_stem_separation': False,  # Already separated
        'segment_duration': 3.0,
        'min_duration': 10.0,
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
    },
    'deepspeak_v2': {
        'description': 'DeepSpeak v2 talking head audio (single speaker)',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 2.0,
        'min_duration': 5.0,
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize', 'trim_silence'],
    },
    'deepspeak_v2_train': {
        'description': 'DeepSpeak v2 training split (single speaker)',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 1.5,  # Reduced for short clips (Option C)
        'segment_hop': 0.75,  # 50% overlap
        'min_segments': 2,  # Reduced to keep more short clips
        'min_duration': 2.25,  # Minimum for 2 segments with 1.5s duration
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize', 'trim_silence'],
    },
    'deepspeak_v2_test': {
        'description': 'DeepSpeak v2 test split (single speaker)',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 1.5,  # Reduced for short clips (Option C)
        'segment_hop': 0.75,  # 50% overlap
        'min_segments': 2,  # Reduced to keep more short clips
        'min_duration': 2.25,  # Minimum for 2 segments with 1.5s duration
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize', 'trim_silence'],
    },
    'mlaad_english': {
        'description': 'MLAAD English TTS (84 models) + M-AILABS real speech',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 2.0,
        'min_duration': 4.0,
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize', 'trim_silence'],
    },
    'atadd_sound': {
        'description': 'Environmental/synthetic sounds (AT-ADD benchmark)',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 2.0,
        'segment_hop': 1.0,
        'min_segments': 2,
        'min_duration': 3.0,
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
    },
    'atadd_singing': {
        'description': 'Singing voice (AT-ADD benchmark)',
        'use_diarization': False,
        'use_stem_separation': False,
        'segment_duration': 2.0,
        'segment_hop': 1.0,
        'min_segments': 2,
        'min_duration': 3.0,
        'embedding_model': 'laion_clap',
        'preprocessing': ['normalize'],
    },
}


# Combined configurations for training data ablation
# Maps config name -> list of audio types to combine
COMBINED_CONFIGS: Dict[str, List[str]] = {
    # Single voice combinations (existing data)
    'sv_ds': ['single_voice', 'deepspeak_v2_train'],

    # Single voice + MLAAD (84 TTS models for diversity)
    'sv_ds_mla': ['single_voice', 'deepspeak_v2_train', 'mlaad_english'],

    # Music combinations
    'music_all': ['music_instrumental', 'music_with_vocals'],

    # Future combinations (when FakeAVCeleb available)
    # 'sv_ds_fav': ['single_voice', 'deepspeak_v2_train', 'fakeavceleb'],
    # 'sv_ds_fav_mla': ['single_voice', 'deepspeak_v2_train', 'fakeavceleb', 'mlaad_english'],
    # 'universal_v3': ['single_voice', 'deepspeak_v2_train', 'music_instrumental',
    #                  'music_with_vocals', 'fakeavceleb'],
    # 'universal_full': ['single_voice', 'deepspeak_v2_train', 'music_instrumental',
    #                    'music_with_vocals', 'fakeavceleb', 'mlaad_english'],
}


def get_combined_audio_types(combined_name: str) -> List[str]:
    """Get list of audio types for a combined configuration.

    Args:
        combined_name: Name from COMBINED_CONFIGS

    Returns:
        List of audio type names to combine

    Raises:
        ValueError: If combined_name is not recognized
    """
    if combined_name not in COMBINED_CONFIGS:
        valid_names = list(COMBINED_CONFIGS.keys())
        raise ValueError(
            f"Unknown combined config: {combined_name}. "
            f"Valid names are: {valid_names}"
        )
    return COMBINED_CONFIGS[combined_name].copy()


def list_combined_configs() -> list:
    """List all available combined configurations."""
    return list(COMBINED_CONFIGS.keys())


def get_audio_type_config(audio_type: str) -> Dict[str, Any]:
    """Get configuration for a specific audio type.

    Args:
        audio_type: One of the keys in AUDIO_TYPE_CONFIGS

    Returns:
        Configuration dictionary for the audio type

    Raises:
        ValueError: If audio_type is not recognized
    """
    if audio_type not in AUDIO_TYPE_CONFIGS:
        valid_types = list(AUDIO_TYPE_CONFIGS.keys())
        raise ValueError(
            f"Unknown audio type: {audio_type}. "
            f"Valid types are: {valid_types}"
        )
    return AUDIO_TYPE_CONFIGS[audio_type].copy()


def list_audio_types() -> list:
    """List all available audio types."""
    return list(AUDIO_TYPE_CONFIGS.keys())


def requires_diarization(audio_type: str) -> bool:
    """Check if an audio type requires speaker diarization."""
    config = get_audio_type_config(audio_type)
    return config.get('use_diarization', False)


def requires_stem_separation(audio_type: str) -> bool:
    """Check if an audio type requires stem separation."""
    config = get_audio_type_config(audio_type)
    return config.get('use_stem_separation', False)
