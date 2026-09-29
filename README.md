# transcreve

Vídeo ou áudio vira Markdown com timestamp de cada frase. Roda inteiro na sua
máquina: sem conta, sem chave de API, sem custo por minuto, sem mandar o arquivo
para servidor nenhum.

Um vídeo de 2 minutos sai em cerca de 40 segundos num MacBook M2.

**Só macOS.** Depende de Homebrew e do Metal da Apple. No Linux dá para adaptar
trocando a instalação das dependências; no Windows, não tente por este caminho.

## Instalar

```bash
git clone https://github.com/natagomes01/transcreve.git
cd transcreve
bash install.sh
```

O instalador põe `ffmpeg` e `whisper-cpp` pelo Homebrew, baixa o modelo de
transcrição (1,5 GB, uma vez só, fica em `~/.cache/whisper-models/`) e cria o
atalho `trs` no seu terminal. Rode de novo quando quiser: nada é duplicado.

Python não precisa instalar. O script usa só a biblioteca padrão e roda no
`python3` que já vem no macOS.

## Usar

```bash
trs video.mp4              # o .md nasce ao lado do vídeo, mesmo nome
trs ~/Videos/              # a pasta inteira, um .md por arquivo
trs a.mp4 b.mov c.m4a      # vários de uma vez
```

| Flag | Para quê |
|------|----------|
| `--saida DIR` | escreve os `.md` noutra pasta |
| `--srt` | guarda também a legenda `.srt` |
| `--palavras` | grava também `nome.palavras.json`, com o tempo de cada palavra |
| `--prompt "..."` | ensina nomes próprios ao modelo |
| `--idioma en` | outro idioma (padrão `pt`) |
| `--modelo NOME` | outro modelo (padrão `large-v3-turbo`) |

### Nome próprio saindo errado

Não corrija o `.md` na mão, senão o erro volta no próximo arquivo. Ensine:

```bash
trs video.mp4 --prompt "Nomes que aparecem: Mateus Ribeiro, Albatroz Studio, n8n, Supabase."
```

## Formatos que aceita

| | |
|---|---|
| **Vídeo** | `.mp4` `.mov` `.m4v` `.avi` `.mkv` `.webm` |
| **Áudio** | `.mp3` `.m4a` `.wav` `.aac` `.ogg` `.flac` |

Quem abre o arquivo é o `ffmpeg`, que lê muito mais coisa que isso. A lista
acima é só o filtro de extensão que o script usa para varrer uma pasta. Se você
tem um formato exótico e aponta o arquivo direto, é provável que funcione.

## O que sai

Um `.md` com duas seções, de propósito:

```markdown
## Texto corrido
Presta atenção aqui porque esse cara é brasileiro e tá revolucionando...

## Com timestamps
- **[00:00]** Presta atenção aqui porque esse cara é brasileiro.
- **[00:12]** Você pega qualquer ideia e o sistema transforma em conteúdo.
```

O texto corrido é para jogar num modelo de linguagem escrever legenda, roteiro
ou resumo. A lista com timestamp é para achar o corte na hora de editar.

A transcrição **guarda** gaguejo, repetição e take refeito. É de propósito: se
você gravou o mesmo gancho três vezes, os três aparecem com o segundo de cada
um, e é isso que ajuda a escolher na edição.

## Tempo por palavra (`--palavras`)

```bash
trs video.mp4 --palavras
```

Além do `.md`, nasce um `video.palavras.json` com o início de cada palavra.
Serve para legenda que acende palavra por palavra e para cortar vídeo no
limite exato da fala.

```json
{
  "version": 1, "model": "large-v3-turbo", "language": "pt",
  "dtw_shift_s": -0.15, "duration_s": 600.2,
  "words": [ {"t": 79.02, "p": 0.99, "w": "Então"}, {"t": 79.12, "p": 1.0, "w": "vamos"} ]
}
```

- `t` é o segundo em que a palavra começa; `p` é a confiança do modelo (0 a 1);
  `w` é a palavra, com a pontuação grudada.
- O tempo vem do **DTW** do whisper.cpp: o modelo "olha" para o trecho do áudio
  em que cada palavra foi dita, e um algoritmo de alinhamento acha esse ponto.
  Medido contra o início real da fala, 90% das palavras caem a menos de
  0,14 s do lugar certo.
- O DTW marca a palavra um pouco atrasada, então o script adianta tudo em
  0,15 s. O valor vai gravado no próprio JSON (`dtw_shift_s`).
- Custa uns 30% a mais de tempo, porque o DTW exige desligar a flash attention
  do whisper. Sem a flag, nada muda.
- Se alguma palavra vier sem tempo, o arquivo falha com erro em vez de gravar
  um JSON errado.
- Só funciona com os modelos padrão do whisper.cpp (`tiny` até
  `large-v3-turbo`).

## O que ele não faz

- Não separa quem falou o quê. Vídeo com duas pessoas sai como fala contínua.
- Não limpa o texto nem corrige o que você disse.
- Música alta por cima da voz degrada bastante o resultado.
- Em trecho longo de silêncio, o Whisper às vezes inventa uma frase de
  encerramento. Se aparecer frase fantasma no fim, é isso.

## Por dentro

```
MP4  ->  ffmpeg  ->  WAV 16 kHz mono  ->  whisper.cpp  ->  SRT  ->  Markdown
```

O SRT do Whisper quebra a cada 5 a 8 palavras, porque nasceu para virar legenda
na tela. O script força um corte ainda mais fino, divide cada bloco na
pontuação, estima o tempo de cada pedaço e remonta em frase inteira. É daí que
sai o timestamp por frase em vez de um a cada 10 segundos.

O modelo é o `large-v3-turbo`, do
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) de Georgi Gerganov, que
é a implementação em C++ do [Whisper](https://github.com/openai/whisper) da
OpenAI. Fica fora do repositório de propósito: 1,5 GB não entra em git.

## Desinstalar

```bash
brew uninstall whisper-cpp          # o ffmpeg talvez você use para outra coisa
rm -rf ~/.cache/whisper-models ~/.local/share/transcreve
```

E apague a função `trs` do seu `~/.zshrc`.

---

Feito por [Albatroz Studio](https://albatroz.studio). Licença MIT: use, modifique
e distribua à vontade.
