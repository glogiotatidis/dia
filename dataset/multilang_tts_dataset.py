import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import torchaudio.transforms as T

class MultilangTTSDataset(Dataset):
    def __init__(self, manifest_path, lang_vocab, sample_rate=22050, n_mels=80):
        self.data = json.load(open(manifest_path))
        self.lang_vocab = lang_vocab
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.phoneme_tokenizer = self._build_tokenizer()
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney"
        )

    def _build_tokenizer(self):
        from collections import Counter
        chars = Counter()
        for sample in self.data:
            chars.update(sample["phonemes"])
        vocab = {c: i+10 for i, c in enumerate(sorted(chars))}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        self.phoneme_vocab = vocab
        return lambda p: [vocab.get(c, 1) for c in p]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        path = Path(sample["audio"])
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        
        # Compute mel spectrogram: (1, n_mels, time) -> (time, n_mels)
        mel = self.mel_transform(wav)
        mel = mel.squeeze(0).T  # (time, n_mels)
        
        # Log mel spectrogram
        mel = torch.log(mel.clamp(min=1e-5))

        phoneme_ids = self.phoneme_tokenizer(sample["phonemes"])
        lang_token = f"<{sample['lang']}>"
        lang_token_id = self.lang_vocab.get(lang_token, self.lang_vocab.get("<unk>", 0))
        input_ids = [lang_token_id] + phoneme_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "mel": mel,  # (time, n_mels)
            "lang": sample["lang"],
            "lang_token_id": lang_token_id,
            "path": str(path)
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_seqs = [b["input_ids"] for b in batch]
    mels = [b["mel"] for b in batch]  # List of (time, n_mels) tensors
    
    padded_inputs = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    mel_lens = [m.shape[0] for m in mels]
    padded_mels = pad_sequence(mels, batch_first=True)  # (B, max_time, n_mels)

    return {
        "input_ids": padded_inputs,
        "audio": padded_mels,  # (B, T, 80) - mel spectrograms
        "audio_lens": mel_lens,
        "langs": [b["lang"] for b in batch],
        "lang_token_ids": torch.tensor([b["lang_token_id"] for b in batch]),
        "paths": [b["path"] for b in batch]
    }