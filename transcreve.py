#!/usr/bin/env python3
"""transcreve.py — MP4 (ou áudio) -> Markdown com timestamps, 100% local.

Cadeia: ffmpeg extrai o áudio -> whisper.cpp transcreve -> este script agrupa
o SRT em frases e escreve o .md ao lado do vídeo.

Uso:
    python3 transcreve.py video.mp4 [outro.mp4 ...]
    python3 transcreve.py pasta/            # todos os vídeos da pasta
    python3 transcreve.py video.mp4 --saida ~/dev/nata/content/transcricoes

Opções:
    --modelo NOME     nome do modelo ggml (padrão: large-v3-turbo)
    --idioma XX       código do idioma (padrão: pt)
    --saida DIR       onde escrever o .md (padrão: ao lado do vídeo)
    --prompt TEXTO    prompt inicial que ensina nomes próprios ao modelo
    --srt             guarda também o .srt bruto ao lado do .md

Só usa a stdlib do Python. Depende de ffmpeg e whisper-cli no PATH.
"""

from __future__ import annotations  # anotações novas no python3 do sistema (3.9)

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "whisper-models"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin"

# Nomes próprios e jargão que o Natã fala nos vídeos. Sem isso o Whisper escreve
# "Albatroz" como "Alba Trós" e "n8n" como "n oito n".
PROMPT_PADRAO = (
    "Transcrição em português do Brasil. Termos que aparecem: Albatroz Studio, "
    "Natã Gomes, n8n, WhatsApp, CRM, IA, inteligência artificial, follow-up, "
    "lead, funil, automação, agente de IA, Supabase, prompt."
)

EXTENSOES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
             ".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}

# Uma frase fecha aqui. Se passar do teto sem fechar, corta assim mesmo.
FIM_DE_FRASE = re.compile(r"[.!?…]['\"”’)]?$")
TETO_FRASE = 220
# Pausa (em segundos) que vira quebra de parágrafo no texto corrido.
PAUSA_PARAGRAFO = 0.8


class Falha(Exception):
    pass


# ---------------------------------------------------------------- utilidades


def exige(binario: str, dica: str) -> str:
    caminho = shutil.which(binario)
    if not caminho:
        raise Falha(f"'{binario}' não está no PATH. Instale com: {dica}")
    return caminho


def garante_modelo(nome: str) -> Path:
    destino = MODEL_DIR / f"ggml-{nome}.bin"
    if destino.exists() and destino.stat().st_size > 1_000_000:
        return destino
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(nome)
    print(f"  baixando modelo {nome} (só na primeira vez)...", flush=True)
    parcial = destino.with_suffix(".bin.parcial")
    try:
        urllib.request.urlretrieve(url, parcial)
    except Exception as erro:  # rede caiu, nome de modelo errado, 404
        parcial.unlink(missing_ok=True)
        raise Falha(f"não consegui baixar o modelo '{nome}': {erro}")
    parcial.replace(destino)
    return destino


def duracao_segundos(arquivo: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    saida = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(arquivo)],
        capture_output=True, text=True,
    )
    try:
        return float(saida.stdout.strip())
    except ValueError:
        return 0.0


def hms(segundos: float) -> str:
    s = int(segundos)
    h, resto = divmod(s, 3600)
    m, s = divmod(resto, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ------------------------------------------------------------------- pipeline


def extrai_audio(video: Path, wav: Path) -> None:
    """Whisper quer PCM 16 kHz mono. Qualquer outra coisa ele recusa."""
    subprocess.run(
        [exige("ffmpeg", "brew install ffmpeg"), "-y", "-loglevel", "error",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )


def roda_whisper(wav: Path, modelo: Path, idioma: str, prompt: str,
                 prefixo: Path) -> Path:
    subprocess.run(
        [exige("whisper-cli", "brew install whisper-cpp"),
         "-m", str(modelo), "-f", str(wav), "-l", idioma,
         "--prompt", prompt, "-sns", "-np",
         # Corta curto e no limite da palavra. Sozinho o modelo emite blocos de
         # 10 s, e o timestamp fica grosso demais pra achar o corte na edição.
         # O agrupamento por frase, mais abaixo, remonta o texto.
         "-ml", "60", "-sow",
         "-osrt", "-of", str(prefixo)],
        check=True,
    )
    srt = prefixo.with_suffix(".srt")
    if not srt.exists():
        raise Falha("o whisper terminou sem escrever o .srt")
    return srt


def le_srt(srt: Path) -> list[tuple[float, float, str]]:
    """Devolve os blocos crus do SRT: (início, fim, texto)."""
    padrao = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    blocos = []
    for bruto in srt.read_text(encoding="utf-8").strip().split("\n\n"):
        linhas = [l for l in bruto.splitlines() if l.strip()]
        if len(linhas) < 2:
            continue
        marca = next((l for l in linhas if padrao.search(l)), None)
        if not marca:
            continue
        m = padrao.search(marca)
        ini = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
        fim = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
        texto = " ".join(linhas[linhas.index(marca) + 1:]).strip()
        texto = re.sub(r"\[.*?\]|\(.*?\)", "", texto).strip()  # [música], (risos)
        if texto:
            blocos.append((ini, fim, texto))
    return blocos


def divide_por_frase(blocos):
    """O `-ml` corta o bloco a cada 60 caracteres, então o ponto final cai no
    meio do bloco. Aqui o bloco é partido na pontuação e o tempo de cada pedaço
    sai por regra de três sobre o comprimento (a fala é regular dentro de 60
    caracteres, o erro fica em fração de segundo)."""
    corte = re.compile(r"[.!?…]['\"”’)]?\s+")
    saida = []
    for ini, fim, texto in blocos:
        pontos = [m.end() for m in corte.finditer(texto)]
        if not pontos:
            saida.append((ini, fim, texto))
            continue
        span, total = fim - ini, max(len(texto), 1)
        anterior = 0
        for p in pontos + [len(texto)]:
            pedaco = texto[anterior:p].strip()
            if pedaco:
                saida.append((ini + span * anterior / total,
                              ini + span * p / total, pedaco))
            anterior = p
    return saida


def agrupa_em_frases(blocos) -> list[dict]:
    """O SRT quebra a cada 5-8 palavras. Isso não serve de insumo pra /reels:
    remonta em frases inteiras, guardando o início de cada uma."""
    frases, buffer, inicio, fim_anterior = [], "", None, None
    for ini, fim, texto in blocos:
        if buffer:
            buffer += " " + texto
        else:
            buffer, inicio = texto, ini
            pausa = 0.0 if fim_anterior is None else ini - fim_anterior
        fecha = FIM_DE_FRASE.search(buffer) or len(buffer) >= TETO_FRASE
        if fecha:
            frases.append({"inicio": inicio, "texto": buffer.strip(),
                           "pausa": pausa})
            buffer, inicio = "", None
        fim_anterior = fim
    if buffer:
        frases.append({"inicio": inicio, "texto": buffer.strip(),
                       "pausa": pausa})
    return frases


def monta_markdown(video: Path, frases: list[dict], modelo: str,
                   duracao: float) -> str:
    paragrafos, atual = [], []
    for f in frases:
        if atual and f["pausa"] >= PAUSA_PARAGRAFO:
            paragrafos.append(" ".join(atual))
            atual = []
        atual.append(f["texto"])
    if atual:
        paragrafos.append(" ".join(atual))

    linhas = [
        f"# Transcrição — {video.stem}",
        "",
        f"- Arquivo: `{video.name}`",
        f"- Duração: {hms(duracao)}",
        f"- Modelo: whisper.cpp `{modelo}`",
        f"- Gerado em: {date.today().isoformat()}",
        "",
        "## Texto corrido",
        "",
    ]
    linhas += [p + "\n" for p in paragrafos] or ["_(sem fala detectada)_\n"]
    linhas += ["## Com timestamps", ""]
    linhas += [f"- **[{hms(f['inicio'])}]** {f['texto']}" for f in frases]
    linhas.append("")
    return "\n".join(linhas)


def transcreve(video: Path, modelo_nome: str, idioma: str, prompt: str,
               saida_dir: Path | None, guardar_srt: bool) -> Path:
    modelo = garante_modelo(modelo_nome)
    destino_dir = saida_dir or video.parent
    destino_dir.mkdir(parents=True, exist_ok=True)
    md = destino_dir / f"{video.stem}.md"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = tmp / "audio.wav"
        extrai_audio(video, wav)
        srt = roda_whisper(wav, modelo, idioma, prompt, tmp / video.stem)
        if guardar_srt:
            shutil.copy2(srt, destino_dir / f"{video.stem}.srt")
        blocos = le_srt(srt)

    frases = agrupa_em_frases(divide_por_frase(blocos))
    md.write_text(
        monta_markdown(video, frases, modelo_nome, duracao_segundos(video)),
        encoding="utf-8",
    )
    return md


# ----------------------------------------------------------------------- CLI


def alvos(entradas: list[str]) -> list[Path]:
    achados = []
    for bruto in entradas:
        p = Path(bruto).expanduser()
        if p.is_dir():
            achados += sorted(f for f in p.iterdir()
                              if f.suffix.lower() in EXTENSOES)
        elif p.is_file():
            achados.append(p)
        else:
            print(f"  aviso: não achei '{bruto}', pulando", file=sys.stderr)
    return achados


def main() -> int:
    ap = argparse.ArgumentParser(description="MP4 -> Markdown com timestamps")
    ap.add_argument("entradas", nargs="+", help="arquivos de vídeo/áudio ou pasta")
    ap.add_argument("--modelo", default=os.environ.get("WHISPER_MODELO",
                                                       "large-v3-turbo"))
    ap.add_argument("--idioma", default="pt")
    ap.add_argument("--saida", default=None, help="pasta de destino do .md")
    ap.add_argument("--prompt", default=PROMPT_PADRAO)
    ap.add_argument("--srt", action="store_true", help="guardar também o .srt")
    args = ap.parse_args()

    videos = alvos(args.entradas)
    if not videos:
        print("nenhum vídeo ou áudio encontrado.", file=sys.stderr)
        return 1

    saida_dir = Path(args.saida).expanduser() if args.saida else None
    erros = 0
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}", flush=True)
        try:
            md = transcreve(video, args.modelo, args.idioma, args.prompt,
                            saida_dir, args.srt)
            print(f"  -> {md}", flush=True)
        except (Falha, subprocess.CalledProcessError) as erro:
            print(f"  FALHOU: {erro}", file=sys.stderr)
            erros += 1
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
