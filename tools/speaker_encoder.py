import torch
import torchaudio

# Handle SpeechBrain 1.0 API change
try:
    from speechbrain.inference import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier

class SpeakerEncoder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )

    def encode(self, wav_path):
        signal, fs = torchaudio.load(wav_path)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
        # Move signal to device
        signal = signal.to(self.device)
        embedding = self.model.encode_batch(signal).squeeze()
        # Ensure 192-dim output (ECAPA-TDNN outputs 192-dim embeddings)
        if embedding.dim() == 0:
            embedding = embedding.unsqueeze(0)
        return embedding.detach().cpu()

# Usage:
# enc = SpeakerEncoder()
# spk_vec = enc.encode("sample.wav")