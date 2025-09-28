"""Video transcription utility.

Bu betik, verilen bir video dosyasının sesini ayıklar ve Whisper modeli ile
Türkçe de dahil olmak üzere metni çıkarır. Varsayılan olarak OpenAI'nin resmi
Whisper uygulamasını kullanır.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from faster_whisper import WhisperModel


def resolve_ffmpeg_binary(custom_binary: str | None = None) -> str:
    """FFmpeg yürütülebilir dosyasının yolunu tespit et."""

    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates: List[str] = []

    def add_candidate(candidate: str | None) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add_candidate(custom_binary)
    add_candidate(os.environ.get("FFMPEG_BINARY"))
    add_candidate(os.environ.get("FFMPEG_PATH"))

    default_dirs = [
        r"C:\ffmpeg\bin",
        r"C:\ffmpeg",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files\ffmpeg",
        r"C:\Program Files (x86)\ffmpeg\bin",
    ]
    for directory in default_dirs:
        add_candidate(directory)

    add_candidate(exe_name)
    add_candidate("ffmpeg")

    tried: List[str] = []

    for candidate in candidates:
        resolved: str | None = None
        tried.append(candidate)

        path_candidate = Path(candidate)
        if path_candidate.is_dir():
            exe_path = path_candidate / exe_name
            if exe_path.exists():
                resolved = str(exe_path)
        elif path_candidate.exists():
            resolved = str(path_candidate)

        if resolved is None:
            located = shutil.which(candidate)
            if located:
                resolved = located

        if resolved:
            return resolved

    raise RuntimeError(
        "ffmpeg bulunamadı. Denenen yollar: "
        + ", ".join(tried)
        + ". PATH'e ekleyin veya '--ffmpeg' parametresiyle tam yolu belirtin."
    )


def extract_audio(
    video_path: Path,
    audio_path: Path,
    sample_rate: int = 16000,
    ffmpeg_binary: str | None = None,
) -> None:
    """ffmpeg ile videodan ses kanalını çıkar."""

    binary = resolve_ffmpeg_binary(ffmpeg_binary)

    command = [
        binary,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(audio_path),
    ]

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg ile ses çıkarma başarısız oldu:\n"
            f"STDOUT: {completed.stdout}\nSTDERR: {completed.stderr}"
        )


_MODEL_CACHE: Dict[Tuple[str, str, str], WhisperModel] = {}

PRELOAD_MODEL_NAME = os.getenv("WHISPER_PRELOAD_MODEL", "turbo").strip() or None
PRELOAD_DEVICE = os.getenv("WHISPER_PRELOAD_DEVICE")
PRELOAD_COMPUTE_TYPE = os.getenv("WHISPER_PRELOAD_COMPUTE_TYPE")


def _resolve_compute_type(device: str | None, compute_type: str | None) -> str:
    if compute_type:
        return compute_type
    if device in {None, "cpu"}:
        return "int8"
    return "float16"


def _load_model(model_name: str, device: str | None, compute_type: str) -> WhisperModel:
    key = (model_name, device or "auto", compute_type)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(
            model_name,
            device=device or "auto",
            compute_type=compute_type,
        )
    return _MODEL_CACHE[key]


def _preload_default_model() -> None:
    if not PRELOAD_MODEL_NAME:
        return

    try:
        compute_type = _resolve_compute_type(PRELOAD_DEVICE, PRELOAD_COMPUTE_TYPE)
        _load_model(PRELOAD_MODEL_NAME, PRELOAD_DEVICE, compute_type)
    except Exception as exc:
        warnings.warn(
            f"Ön yükleme başarısız oldu (model={PRELOAD_MODEL_NAME}): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


_preload_default_model()

def transcribe_audio(
    audio_path: Path,
    model_name: str = "small",
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = False,
    verbose: bool = False,
) -> dict:
    """Faster-Whisper ile ses dosyasını yazılı metne çevir."""

    resolved_compute_type = _resolve_compute_type(device, compute_type)
    model = _load_model(model_name, device, resolved_compute_type)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    segments = []
    texts: list[str] = []
    for idx, segment in enumerate(segments_iter):
        segments.append(
            {
                "id": idx,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "temperature": segment.temperature,
            }
        )
        texts.append(segment.text.strip())

    result = {
        "text": " ".join(texts).strip(),
        "segments": segments,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": getattr(info, "duration_after_vad", info.duration),
        "model": model_name,
        "device": device or "auto",
        "compute_type": resolved_compute_type,
    }
    if verbose:
        result["raw"] = {
            "vad": vad_filter,
            "beam_size": beam_size,
            "num_segments": len(segments),
        }
    return result


def save_transcript(result: dict, output_path: Path, output_format: str = "txt") -> None:
    """Transkripsiyonu belirtilen formatta diske kaydet."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "txt":
        output_path.write_text(result["text"], encoding="utf-8")
    elif output_format == "json":
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise ValueError("Desteklenmeyen format. 'txt' veya 'json' seçebilirsiniz.")


ProgressCallback = Callable[[str, float, str | None], None]


def transcribe_video(
    video_path: Path,
    model_name: str = "small",
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    ffmpeg_binary: str | None = None,
    output_format: str = "txt",
    output_path: Path | None = None,
    keep_audio: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, dict]:
    """Videodan transkript üret.

    Dönen değer: (transkript dosya yolu, Whisper sonucu dict)
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")

    if output_format not in {"txt", "json"}:
        raise ValueError("output_format yalnızca 'txt' veya 'json' olabilir")

    if output_path is None:
        suffix = ".json" if output_format == "json" else ".txt"
        output_path = video_path.with_suffix(".transcript" + suffix)

    def notify(stage: str, progress: float, message: str | None = None) -> None:
        if progress_callback is not None:
            progress_callback(stage, progress, message)

    notify("hazırlık", 0.05, "Video dosyası kontrol ediliyor")

    with tempfile.TemporaryDirectory() as tmpdir:
        t_extract_start = time.perf_counter()
        tmp_audio = Path(tmpdir) / "extracted.wav"
        notify("ses_çıkarma", 0.2, "Ses kanalı çıkarılıyor")
        extract_audio(video_path, tmp_audio, ffmpeg_binary=ffmpeg_binary)
        extract_duration = time.perf_counter() - t_extract_start

        notify("transkripsiyon", 0.6, "Whisper modeli çalıştırılıyor")
        t_transcribe_start = time.perf_counter()
        result = transcribe_audio(
            tmp_audio,
            model_name=model_name,
            language=language,
            device=device,
            compute_type=compute_type,
            verbose=verbose,
        )
        transcribe_duration = time.perf_counter() - t_transcribe_start

        notify("kaydetme", 0.85, "Çıktı dosyası kaydediliyor")
        t_save_start = time.perf_counter()
        save_transcript(result, output_path, output_format)
        save_duration = time.perf_counter() - t_save_start

        if keep_audio:
            notify("ses_kaydetme", 0.9, "Ara ses dosyası saklanıyor")
            permanent_audio = output_path.with_suffix(".wav")
            permanent_audio.write_bytes(tmp_audio.read_bytes())

    total_duration = extract_duration + transcribe_duration + save_duration

    metrics = {
        "extract_seconds": extract_duration,
        "transcribe_seconds": transcribe_duration,
        "save_seconds": save_duration,
        "total_seconds": total_duration,
    }

    timeline = []
    for segment in result.get("segments", []):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        timeline.append(
            {
                "index": segment.get("id", len(timeline)) + 1,
                "start": start,
                "end": end,
                "duration": max(end - start, 0.0),
                "text": segment.get("text", ""),
            }
        )

    notify("tamamlandı", 1.0, "İşlem tamamlandı")
    return output_path, result, metrics, timeline


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Videolardan metin çıkaran Whisper uygulaması")
    parser.add_argument("video", type=Path, help="Transkript çıkarılacak video dosyası")
    parser.add_argument(
        "--model",
        default="small",
        help="Kullanılacak Whisper modeli (tiny, base, small, medium, large, large-v3, turbo vb.)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Konuşulan dil. Belirtilmezse Whisper otomatik tespit etmeye çalışır (ör. 'tr').",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA/CPU tercihi. Örnek: 'cuda', 'cpu'. Belirtilmezse Whisper otomatik karar verir.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "json"],
        default="txt",
        help="Çıktı formatı",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Çıkış dosyası yolu. Belirtilmezse video adıyla aynı klasöre kaydeder.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Ara ses dosyasını silme",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whisper transkripsiyon sürecini ayrıntılı göster",
    )
    parser.add_argument(
        "--ffmpeg",
        dest="ffmpeg",
        default=None,
        help="FFmpeg yürütülebilir dosyasının yolu (örn. C:/ffmpeg/bin/ffmpeg.exe)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        output_path, _, _, _ = transcribe_video(
            args.video,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=None,
            ffmpeg_binary=args.ffmpeg,
            output_format=args.format,
            output_path=args.output,
            keep_audio=args.keep_audio,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Transkript kaydedildi: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

