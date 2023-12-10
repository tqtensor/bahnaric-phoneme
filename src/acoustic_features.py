import constants as c
import librosa
import numpy as np
import pandas as pd


def _extract_mfcc_feature_means(
    signal: np.ndarray,
    sr: float,
    number_of_mfcc: int,
) -> pd.DataFrame:
    mfcc_alt = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    mfcc_features = {}

    for i in range(0, number_of_mfcc):
        # MFCC coefficient
        key_name = "".join(["mfcc_", str(i).zfill(2)])
        mfcc_value = np.mean(mfcc_alt[i])
        mfcc_features.update({key_name: mfcc_value})

        # MFCC delta coefficient
        key_name = "".join(["mfcc_delta_", str(i).zfill(2)])
        mfcc_value = np.mean(delta[i])
        mfcc_features.update({key_name: mfcc_value})

        # MFCC accelerate coefficient
        key_name = "".join(["mfcc_accelerate_", str(i).zfill(2)])
        mfcc_value = np.mean(accelerate[i])
        mfcc_features.update({key_name: mfcc_value})

    df = pd.DataFrame.from_records(data=[mfcc_features])
    return df


def extract_feature_means(signal: np.ndarray, sr: float) -> pd.DataFrame:
    # Config settings
    number_of_mfcc = c.NUMBER_OF_MFCC

    # Trim leading and trailing silence from an audio signal (silence before
    # and after the actual audio)
    signal, _ = librosa.effects.trim(signal)

    # 1. Fourier Transform
    n_fft = c.NUMBER_FFT  # FFT window size
    hop_length = (
        c.HOP_LENGTH
    )  # number audio of frames between STFT columns (looks like a good default)

    # Short-time Fourier transform (STFT)
    d_audio = np.abs(
        librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length)
    )

    # 2. Spectrogram
    db_audio = librosa.amplitude_to_db(S=d_audio, ref=np.max)

    # 3. Create the Mel Spectrograms
    s_audio = librosa.feature.melspectrogram(y=signal, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    # 4. Harmonics and Perceptrual
    y_harm, y_perc = librosa.effects.hpss(signal)

    # 5. Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    spectral_centroids_accelerate = librosa.feature.delta(
        spectral_centroids, order=2
    )

    # 6. Chroma Frequencies
    hop_length = c.HOP_LENGTH
    chromagram = librosa.feature.chroma_stft(
        y=signal, sr=sr, hop_length=hop_length
    )

    # 7. Tempo BPM (beats per minute)
    tempo_y, _ = librosa.beat.beat_track(y=signal, sr=sr)

    # 8. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

    # 9. Spectral Bandwidth
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, p=2
    )[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, p=3
    )[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, p=4
    )[0]

    audio_features = {
        "zero_crossing_rate": np.mean(
            librosa.feature.zero_crossing_rate(signal)[0]
        ),
        "zero_crossings": np.sum(librosa.zero_crossings(signal, pad=False)),
        "spectrogram": np.mean(db_audio[0]),
        "mel_spectrogram": np.mean(s_db_audio[0]),
        "harmonics": np.mean(y_harm),
        "perceptual_shock_wave": np.mean(y_perc),
        "spectral_centroids": np.mean(spectral_centroids),
        "spectral_centroids_delta": np.mean(spectral_centroids_delta),
        "spectral_centroids_accelerate": np.mean(
            spectral_centroids_accelerate
        ),
        "chroma_01": np.mean(chromagram[0]),
        "chroma_02": np.mean(chromagram[1]),
        "chroma_03": np.mean(chromagram[2]),
        "chroma_04": np.mean(chromagram[3]),
        "chroma_05": np.mean(chromagram[4]),
        "chroma_06": np.mean(chromagram[5]),
        "chroma_07": np.mean(chromagram[6]),
        "chroma_08": np.mean(chromagram[7]),
        "chroma_09": np.mean(chromagram[8]),
        "chroma_10": np.mean(chromagram[9]),
        "chroma_11": np.mean(chromagram[10]),
        "chroma_12": np.mean(chromagram[11]),
        "tempo_bpm": tempo_y,
        "spectral_rolloff": np.mean(spectral_rolloff),
        "spectral_flux": np.mean(onset_env),
        "spectral_bandwidth_02": np.mean(spectral_bandwidth_2),
        "spectral_bandwidth_03": np.mean(spectral_bandwidth_3),
        "spectral_bandwidth_04": np.mean(spectral_bandwidth_4),
    }

    # Extract MFCC feature
    mfcc_df = _extract_mfcc_feature_means(
        signal=signal, sr=sr, number_of_mfcc=number_of_mfcc
    )

    df = pd.DataFrame.from_records(data=[audio_features])
    df = pd.concat([df, mfcc_df], axis=1)
    return df
