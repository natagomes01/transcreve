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
    --palavras        grava também <nome>.palavras.json com o tempo de cada
                      palavra (DTW), para legenda palavra a palavra

Só usa a stdlib do Python. Depende de ffmpeg e whisper-cli no PATH.
"""

from __future__ import annotations  # anotações novas no python3 do sistema (3.9)

import argparse
import json
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

# Tempo por palavra (--palavras). O DTW do whisper.cpp marca cada palavra uns
# 0,15 s depois do começo real da fala; medido contra silencedetect, com o
# deslocamento o p90 do erro cai de ~0,3-0,46 s para 0,14 s. Em centésimos de
# segundo, que é a unidade do t_dtw, para a conta sair exata.
DTW_DESLOCAMENTO_CS = -15
PALAVRAS_VERSAO = 1


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


# Presets de alignment heads que o whisper-cli aceita em -dtw. Cada modelo tem
# as suas cabeças de atenção que acompanham o tempo do áudio.
PRESETS_DTW = frozenset({
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large.v1", "large.v2", "large.v3",
    "large.v3.turbo",
})


def preset_dtw(modelo_nome: str) -> str:
    """large-v3-turbo -> large.v3.turbo. Sufixo de quantização (-q5_0, -q8_0)
    não muda as cabeças de atenção e sai do nome."""
    base = re.sub(r"-q\d.*$", "", modelo_nome)
    preset = base.replace("-", ".")
    if preset not in PRESETS_DTW:
        raise Falha(
            f"o modelo '{modelo_nome}' não tem preset de DTW, então --palavras "
            "não funciona com ele. Use um modelo padrão (ex.: large-v3-turbo)."
        )
    return preset


def comando_whisper(binario: str, modelo: Path, wav: Path, idioma: str,
                    prompt: str, prefixo: Path, palavras: bool,
                    modelo_nome: str) -> list[str]:
    cmd = [binario, "-m", str(modelo), "-f", str(wav), "-l", idioma,
           "--prompt", prompt, "-sns", "-np",
           # Corta curto e no limite da palavra. Sozinho o modelo emite blocos de
           # 10 s, e o timestamp fica grosso demais pra achar o corte na edição.
           # O agrupamento por frase, mais abaixo, remonta o texto.
           "-ml", "60", "-sow",
           "-osrt"]
    if palavras:
        # -nfa é obrigatório: com flash attention ligada (padrão) o whisper
        # pula o DTW sem avisar e todo t_dtw volta -1.
        cmd += ["-nfa", "-ojf", "-dtw", preset_dtw(modelo_nome)]
    return cmd + ["-of", str(prefixo)]


def roda_whisper(wav: Path, modelo: Path, idioma: str, prompt: str,
                 prefixo: Path, palavras: bool = False,
                 modelo_nome: str = "large-v3-turbo") -> Path:
    subprocess.run(
        comando_whisper(exige("whisper-cli", "brew install whisper-cpp"),
                        modelo, wav, idioma, prompt, prefixo, palavras,
                        modelo_nome),
        check=True,
    )
    # O whisper acrescenta a extensão ao prefixo (-of x -> x.srt). Montado por
    # string, não por with_suffix: em "aula 1.2" o with_suffix trocaria o ".2".
    srt = Path(f"{prefixo}.srt")
    if not srt.exists():
        raise Falha("o whisper terminou sem escrever o .srt")
    if palavras and not Path(f"{prefixo}.json").exists():
        raise Falha("o whisper terminou sem escrever o .json (-ojf)")
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


# Pontuação que, sozinha num token com espaço antes, abre a PRÓXIMA palavra
# (" (" + "ris" + "os" + ")" = "(risos)"), em vez de grudar na anterior.
ABERTURA = "([{“«¿¡\"'"


def _texto_utf8(s: str) -> str:
    """O whisper às vezes parte um caractere de 2 bytes entre dois tokens. O
    arquivo é lido com surrogateescape, então cada byte solto vira \\udcXX;
    juntando os tokens da palavra e refazendo o UTF-8, o "ç" volta inteiro.
    Meio caractere que sobrar vira U+FFFD em vez de derrubar o arquivo."""
    return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def _tem_letra(texto: str) -> bool:
    # byte solto (\udc80-\udcff) é pedaço de letra acentuada, não pontuação
    return any(c.isalnum() or "\udc80" <= c <= "\udcff" for c in texto)


def palavras_de_json(dados: dict) -> list[dict]:
    """Junta os tokens do JSON completo do whisper (-ojf) em palavras.

    Token que começa com espaço abre palavra nova; o que vem sem espaço gruda
    na anterior (" re" + "i" = "rei"). Pontuação solta, mesmo com espaço antes,
    também gruda, exceto a de abertura, que vira prefixo da próxima palavra.
    Tokens especiais ([_BEG_], [_TT_150]) ficam de fora.

    O tempo é o t_dtw do primeiro token com letra da palavra, menos o
    deslocamento, nunca negativo e nunca antes da palavra anterior (o DTW não
    garante ordem). A confiança é o menor p entre os tokens da palavra.

    Palavra sem t_dtw (valor -1) derruba o arquivo: isso só acontece com a
    flash attention ligada, e cair para os offsets daria legenda fora do tempo
    sem ninguém perceber."""
    palavras: list[dict] = []
    cs_anterior = 0
    pendente = ""        # abertura esperando a próxima palavra
    p_pendente = 1.0
    for segmento in dados.get("transcription", []):
        for token in segmento.get("tokens", []):
            texto = token.get("text", "")
            if texto.startswith("[_") or not texto.strip():
                continue
            limpo = texto.strip()
            p = float(token.get("p", 1.0))
            letra = _tem_letra(limpo)
            if (not letra and texto.startswith(" ") and limpo[0] in ABERTURA
                    and all(c in ABERTURA for c in limpo)):
                pendente += limpo
                p_pendente = min(p_pendente, p)
                continue
            if palavras and not pendente and (not texto.startswith(" ") or not letra):
                atual = palavras[-1]
                atual["w"] += limpo
                atual["p"] = min(atual["p"], p)
                continue
            dtw = token.get("t_dtw", -1)
            if dtw is None or dtw < 0:
                raise Falha(
                    f"a palavra '{_texto_utf8(limpo)}' (perto de "
                    f"{token.get('offsets', {}).get('from', 0) / 1000:.1f} s) "
                    "veio sem tempo de DTW. Isso acontece quando a flash "
                    "attention fica ligada: o whisper-cli precisa de -nfa "
                    "junto com -dtw. Nada foi gravado para este arquivo."
                )
            cs = max(0, int(dtw) + DTW_DESLOCAMENTO_CS, cs_anterior)
            cs_anterior = cs
            palavras.append({"t": cs / 100, "p": min(p, p_pendente),
                             "w": pendente + limpo})
            pendente, p_pendente = "", 1.0
    if pendente and palavras:
        palavras[-1]["w"] += pendente
    for palavra in palavras:
        palavra["w"] = _texto_utf8(palavra["w"])
    return palavras


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


def monta_palavras_json(palavras: list[dict], modelo: str, idioma: str,
                        duracao: float) -> dict:
    return {
        "version": PALAVRAS_VERSAO,
        "model": modelo,
        "language": idioma,
        "dtw_shift_s": DTW_DESLOCAMENTO_CS / 100,
        "duration_s": round(duracao, 2),
        "words": palavras,
    }


def transcreve(video: Path, modelo_nome: str, idioma: str, prompt: str,
               saida_dir: Path | None, guardar_srt: bool,
               palavras: bool = False) -> Path:
    modelo = garante_modelo(modelo_nome)
    destino_dir = saida_dir or video.parent
    destino_dir.mkdir(parents=True, exist_ok=True)
    md = destino_dir / f"{video.stem}.md"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = tmp / "audio.wav"
        extrai_audio(video, wav)
        srt = roda_whisper(wav, modelo, idioma, prompt, tmp / video.stem,
                           palavras, modelo_nome)
        if guardar_srt:
            shutil.copy2(srt, destino_dir / f"{video.stem}.srt")
        blocos = le_srt(srt)
        lista = None
        if palavras:
            # Lido e validado ainda dentro do tmp: se o DTW veio quebrado, a
            # Falha sobe antes de qualquer arquivo de palavras ser gravado.
            # surrogateescape: o whisper pode gravar meio caractere num token;
            # palavras_de_json remonta o UTF-8 quando junta os pedaços.
            bruto = json.loads(Path(f"{tmp / video.stem}.json").read_bytes()
                               .decode("utf-8", "surrogateescape"))
            lista = palavras_de_json(bruto)

    duracao = duracao_segundos(video)
    frases = agrupa_em_frases(divide_por_frase(blocos))
    md.write_text(monta_markdown(video, frases, modelo_nome, duracao),
                  encoding="utf-8")
    if lista is not None:
        destino = destino_dir / f"{video.stem}.palavras.json"
        destino.write_text(
            json.dumps(monta_palavras_json(lista, modelo_nome, idioma, duracao),
                       ensure_ascii=False, indent=1),
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
    ap.add_argument("--palavras", action="store_true",
                    help="gravar também <nome>.palavras.json com o tempo de "
                         "cada palavra (DTW)")
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
                            saida_dir, args.srt, args.palavras)
            print(f"  -> {md}", flush=True)
        except (Falha, subprocess.CalledProcessError) as erro:
            print(f"  FALHOU: {erro}", file=sys.stderr)
            erros += 1
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
