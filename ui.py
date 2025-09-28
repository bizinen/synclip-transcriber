"""Modern Streamlit arayüzü ile video transkripsiyon uygulaması.

Komut satırı yerine bu arayüzü kullanarak videonuzu yükleyebilir, ilerlemeyi
takip edebilir ve sonuçları görüntüleyebilirsiniz.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from transcriber import transcribe_video


st.set_page_config(
    page_title="Video Transkripsiyon Asistanı",
    page_icon="🎧",
    layout="wide",
    menu_items={
        "About": "Bu arayüz, Whisper modeli ile videoların metne dönüştürülmesini sağlar.",
    },
)


THEME_COLOR = "#6C63FF"


def show_header() -> None:
    st.markdown(
        f"""
        <style>
        .css-1vencpc {{
            background: linear-gradient(135deg, {THEME_COLOR}, #928DFF);
        }}
        .stApp {{
            background-color: #101223;
            color: #F5F7FF;
        }}
        .stFileUploader label::before {{
            content: "📁";
            margin-right: 0.5rem;
        }}
        .stProgress .st-bo {{
            background-color: {THEME_COLOR};
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.06);
            border-radius: 1rem;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px -12px rgba(108, 99, 255, 0.35);
        }}
        .metric-card h3 {{
            color: {THEME_COLOR};
            margin-bottom: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🎧 Akıllı Video Transkripsiyon")
    st.markdown(
        "Modern, şık ve gerçek zamanlı izlenebilen bir arayüzle Whisper gücünü kullanın."
    )


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getbuffer())
    temp.flush()
    return Path(temp.name)


class ProgressTracker:
    def __init__(self) -> None:
        self.progress_bar = st.progress(0)
        self.status_placeholder = st.empty()

    def __call__(self, stage: str, progress: float, message: str | None) -> None:
        labels = {
            "hazırlık": "Hazırlık",
            "ses_çıkarma": "Ses çıkarma",
            "transkripsiyon": "Whisper transkripsiyon",
            "kaydetme": "Çıktı kaydetme",
            "ses_kaydetme": "Ses dosyası kaydetme",
            "tamamlandı": "Tamamlandı",
        }
        label = labels.get(stage, stage)
        display_message = message or "İşlem devam ediyor"
        self.status_placeholder.markdown(f"**{label}** · {display_message}")
        self.progress_bar.progress(min(max(progress, 0.0), 1.0))


def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Ayarlar")
    model = st.sidebar.selectbox(
        "Whisper modeli",
        options=["tiny", "base", "small", "medium", "large", "large-v3", "turbo"],
        index=2,
        help="Daha büyük modeller daha doğru fakat daha yavaş çalışır.",
    )
    language = st.sidebar.text_input(
        "Dil (opsiyonel)",
        value="tr",
        help="Konuşma dilini belirtmezseniz Whisper otomatik tespit eder.",
    )
    output_format = st.sidebar.radio("Çıktı formatı", options=["txt", "json"], index=0)
    keep_audio = st.sidebar.checkbox("Ara ses dosyasını sakla", value=False)
    device = st.sidebar.selectbox(
        "Çalıştırma ortamı",
        options=["auto", "cpu", "cuda"],
        index=1,
        help="GPU'nuz varsa 'cuda' seçeneğini kullanabilirsiniz.",
    )
    ffmpeg_path = st.sidebar.text_input(
        "FFmpeg yolu",
        value="",
        help="Örn. C:/ffmpeg/bin/ffmpeg.exe. Boş bırakırsanız PATH üzerinde arama yapılır.",
    )

    return {
        "model_name": model,
        "language": language or None,
        "output_format": output_format,
        "keep_audio": keep_audio,
        "device": None if device == "auto" else device,
        "ffmpeg": ffmpeg_path or None,
    }


def render_transcript(result: dict, output_path: Path, output_format: str) -> None:
    st.subheader("📝 Transkript")
    if output_format == "txt":
        st.text_area("Metin", result["text"], height=300)
    else:
        st.json(result)

    st.download_button(
        label="Transkripti indir",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="application/json" if output_format == "json" else "text/plain",
    )


def render_metrics(result: dict) -> None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h3>Uzunluk</h3>
                <p>{:.2f} dk</p>
            </div>
            """.format(result.get("duration", 0) / 60),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>Algılanan Dil</h3>
                <p>{}</p>
            </div>
            """.format(result.get("language", "-")),
            unsafe_allow_html=True,
        )

    with col3:
        tokens = len(result.get("text", "").split())
        st.markdown(
            """
            <div class="metric-card">
                <h3>Kelime Sayısı</h3>
                <p>{}</p>
            </div>
            """.format(tokens),
            unsafe_allow_html=True,
        )


def render_timing_metrics(metrics: dict) -> None:
    st.subheader("⏱️ Süreler")
    col1, col2, col3, col4 = st.columns(4)

    def format_seconds(value: float) -> str:
        return f"{value:.2f} s"

    with col1:
        st.metric("Ses Çıkarma", format_seconds(metrics["extract_seconds"]))
    with col2:
        st.metric("Transkripsiyon", format_seconds(metrics["transcribe_seconds"]))
    with col3:
        st.metric("Kaydetme", format_seconds(metrics["save_seconds"]))
    with col4:
        st.metric("Toplam", format_seconds(metrics["total_seconds"]))


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def render_timeline(timeline: list[dict]) -> None:
    st.subheader("🕒 Zaman Çizelgesi")
    if not timeline:
        st.info("Segment bilgisi bulunamadı.")
        return

    df = pd.DataFrame(
        [
            {
                "Segment": item["index"],
                "Başlangıç": format_timestamp(item["start"]),
                "Bitiş": format_timestamp(item["end"]),
                "Süre (sn)": f"{item['duration']:.2f}",
                "Metin": item["text"],
            }
            for item in timeline
        ]
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Zaman çizelgesini CSV olarak indir",
        df.to_csv(index=False).encode("utf-8"),
        file_name="timeline.csv",
        mime="text/csv",
    )


def main() -> None:
    show_header()
    config = render_sidebar()

    uploaded_video = st.file_uploader(
        "Video yükle",
        type=["mp4", "mov", "mkv", "avi", "flv", "webm"],
        accept_multiple_files=False,
        help="Maksimum dosya boyutu Streamlit limitiyle sınırlıdır.",
    )

    if uploaded_video is None:
        st.info("Başlamak için bir video dosyası yükleyin.")
        return

    if st.button("🚀 Transkripsiyonu Başlat"):
        with st.spinner("İşlem hazırlanıyor..."):
            temp_video_path = save_uploaded_file(uploaded_video)

        tracker = ProgressTracker()

        try:
            output_path, result, metrics, timeline = transcribe_video(
                temp_video_path,
                model_name=config["model_name"],
                language=config["language"],
                device=config["device"],
                ffmpeg_binary=config["ffmpeg"],
                output_format=config["output_format"],
                keep_audio=config["keep_audio"],
                progress_callback=tracker,
            )
        except Exception as exc:
            st.error(f"Bir hata oluştu: {exc}")
            return

        st.success("Transkripsiyon tamamlandı!")
        render_metrics(result)
        render_timing_metrics(metrics)
        render_transcript(result, output_path, config["output_format"])
        render_timeline(timeline)

        if config["keep_audio"]:
            audio_path = output_path.with_suffix(".wav")
            if audio_path.exists():
                st.audio(audio_path.read_bytes(), format="audio/wav")


if __name__ == "__main__":
    main()

