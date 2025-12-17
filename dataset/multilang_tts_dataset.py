"""
Multilingual TTS Dataset for DIA model training.

This dataset loads audio files and converts them to DAC codes for training.
"""
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio


class MultilangTTSDataset(Dataset):
    """Dataset that loads audio and prepares it for DIA training.
    
    Audio is kept as waveforms - DAC encoding happens in the training loop
    to avoid memory issues with pre-encoding.
    """
    
    def __init__(self, manifest_path, lang_vocab, sample_rate=44100, max_audio_len=15.0):
        self.data = json.load(open(manifest_path))
        self.lang_vocab = lang_vocab
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.max_samples = int(max_audio_len * sample_rate)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        path = Path(sample["audio"])
        
        # Load audio
        wav, sr = torchaudio.load(path)
        
        # Resample to 44.1kHz (DAC's native sample rate)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Truncate if too long
        if wav.shape[1] > self.max_samples:
            wav = wav[:, :self.max_samples]
        
        # Prepare text input (byte encoding like the real Dia)
        text = sample.get("text", sample.get("phonemes", ""))
        lang = sample.get("lang", "el")
        
        # Encode text as bytes with speaker markers
        text_with_speaker = f"[S1] {text}"
        byte_text = text_with_speaker.encode("utf-8")
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)

        return {
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "waveform": wav.squeeze(0),  # (samples,)
            "lang": lang,
            "path": str(path)
        }


def collate_fn(batch):
    """Collate function that pads text and audio separately."""
    from torch.nn.utils.rnn import pad_sequence
    
    text_seqs = [b["text_tokens"] for b in batch]
    waveforms = [b["waveform"] for b in batch]
    
    # Pad text sequences
    padded_text = pad_sequence(text_seqs, batch_first=True, padding_value=0)
    
    # Pad waveforms
    max_len = max(w.shape[0] for w in waveforms)
    padded_wavs = torch.zeros(len(waveforms), max_len)
    wav_lens = []
    for i, w in enumerate(waveforms):
        padded_wavs[i, :w.shape[0]] = w
        wav_lens.append(w.shape[0])
    
    return {
        "text_tokens": padded_text,  # (B, S)
        "waveforms": padded_wavs,    # (B, samples)
        "wav_lens": wav_lens,
        "langs": [b["lang"] for b in batch],
        "paths": [b["path"] for b in batch]
    }
