#!/usr/bin/env bash
# install.sh — instala o transcreve: vídeo/áudio -> Markdown com timestamps.
#
# Só macOS. Roda tudo na sua máquina, offline, sem custo por minuto.
# Idempotente: pode rodar de novo sem estragar nada.
#
#   bash install.sh

set -euo pipefail

MODELO="${MODELO:-large-v3-turbo}"
DIR_MODELOS="$HOME/.cache/whisper-models"
DIR_APP="$HOME/.local/share/transcreve"
FONTE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

titulo() { printf '\n\033[1m%s\033[0m\n' "$1"; }
ok()     { printf '  ok    %s\n' "$1"; }
erro()   { printf '  ERRO  %s\n' "$1" >&2; }

# 1. Sistema ------------------------------------------------------------------
titulo "1/5  Sistema"
if [[ "$(uname -s)" != "Darwin" ]]; then
  erro "isto é só para macOS. No Linux dá para adaptar; no Windows, não."
  exit 1
fi
ok "macOS $(sw_vers -productVersion) ($(uname -m))"
[[ "$(uname -m)" == "arm64" ]] || echo "  aviso: Mac Intel transcreve bem mais devagar (sem Metal)."

if ! command -v brew >/dev/null 2>&1; then
  erro "Homebrew não encontrado. Instale primeiro, em https://brew.sh, e rode de novo."
  exit 1
fi
ok "Homebrew"

command -v python3 >/dev/null 2>&1 || { erro "python3 não encontrado (vem no macOS)."; exit 1; }
ok "python3 $(python3 -V 2>&1 | cut -d' ' -f2)  (só biblioteca padrão, nada a instalar)"

# 2. Dependências -------------------------------------------------------------
titulo "2/5  ffmpeg e whisper.cpp"
for pacote in ffmpeg whisper-cpp; do
  if brew list --formula "$pacote" >/dev/null 2>&1; then
    ok "$pacote (já instalado)"
  else
    echo "  instalando $pacote..."
    brew install "$pacote"
  fi
done

# 3. Modelo -------------------------------------------------------------------
titulo "3/5  Modelo de transcrição ($MODELO)"
mkdir -p "$DIR_MODELOS"
ALVO="$DIR_MODELOS/ggml-$MODELO.bin"
if [[ -f "$ALVO" ]] && [[ "$(stat -f%z "$ALVO")" -gt 1000000 ]]; then
  ok "já baixado ($(du -h "$ALVO" | cut -f1))"
else
  echo "  baixando ~1,5 GB, uma vez só. Pode demorar alguns minutos."
  # Baixa para .parcial e só renomeia no fim: rede caindo não deixa modelo truncado.
  curl -L --fail --progress-bar \
    -o "$ALVO.parcial" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$MODELO.bin"
  mv "$ALVO.parcial" "$ALVO"
  ok "baixado em $ALVO"
fi

# 4. Script -------------------------------------------------------------------
titulo "4/5  Script"
mkdir -p "$DIR_APP"
cp "$FONTE/transcreve.py" "$DIR_APP/transcreve.py"
chmod +x "$DIR_APP/transcreve.py"
ok "$DIR_APP/transcreve.py"

# 5. Atalho no shell ----------------------------------------------------------
titulo "5/5  Atalho 'trs' no terminal"
# Nome é trs e não tr porque /usr/bin/tr é comando POSIX: sobrescrever quebra script.
RC="$HOME/.zshrc"
[[ "${SHELL##*/}" == "bash" ]] && RC="$HOME/.bash_profile"

if grep -q "transcreve.py" "$RC" 2>/dev/null; then
  ok "já existe em $RC"
else
  cat >> "$RC" <<EOF

# trs <arquivo|pasta> — transcreve vídeo/áudio para Markdown com timestamps.
trs() {
  if [ \$# -eq 0 ]; then
    echo "uso: trs <arquivo.mp4|pasta> [--saida DIR] [--srt] [--prompt \"nomes próprios\"]" >&2
    return 1
  fi
  python3 "$DIR_APP/transcreve.py" "\$@"
}
EOF
  ok "adicionado em $RC"
fi

titulo "Pronto"
cat <<EOF
Abra uma aba nova do terminal (ou rode: source $RC) e use:

  trs video.mp4              # o .md nasce do lado do vídeo
  trs ~/Videos/              # a pasta inteira
  trs video.mp4 --srt        # guarda a legenda .srt também

Erra nome próprio? Ensine, em vez de corrigir na mão:

  trs video.mp4 --prompt "Nomes que aparecem: Mateus, Albatroz Studio, n8n."
EOF
