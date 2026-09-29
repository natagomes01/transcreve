import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import transcreve  # noqa: E402

ARGS = dict(binario="whisper-cli", modelo=Path("m.bin"), wav=Path("a.wav"),
            idioma="pt", prompt="nomes", prefixo=Path("saida/x"),
            modelo_nome="large-v3-turbo")

COMANDO_DE_HOJE = [
    "whisper-cli", "-m", "m.bin", "-f", "a.wav", "-l", "pt",
    "--prompt", "nomes", "-sns", "-np", "-ml", "60", "-sow",
    "-osrt", "-of", "saida/x",
]


class ComandoWhisper(unittest.TestCase):
    def test_caminho_normal_fica_igual_ao_de_hoje(self):
        self.assertEqual(transcreve.comando_whisper(palavras=False, **ARGS),
                         COMANDO_DE_HOJE)

    def test_palavras_desliga_flash_attention_e_pede_dtw(self):
        cmd = transcreve.comando_whisper(palavras=True, **ARGS)
        self.assertIn("-nfa", cmd)
        self.assertIn("-ojf", cmd)
        i = cmd.index("-dtw")
        self.assertEqual(cmd[i + 1], "large.v3.turbo")
        self.assertIn("-osrt", cmd)  # o .md continua saindo do SRT

    def test_palavras_mantem_o_resto_do_comando(self):
        cmd = transcreve.comando_whisper(palavras=True, **ARGS)
        for pedaco in COMANDO_DE_HOJE:
            self.assertIn(pedaco, cmd)


class PresetDtw(unittest.TestCase):
    def test_nomes_conhecidos(self):
        casos = {
            "large-v3-turbo": "large.v3.turbo",
            "large-v3": "large.v3",
            "small": "small",
            "base.en": "base.en",
            "medium-en": "medium.en",
            "large-v3-turbo-q5_0": "large.v3.turbo",
            "large-v2-q8_0": "large.v2",
        }
        for nome, esperado in casos.items():
            with self.subTest(nome=nome):
                self.assertEqual(transcreve.preset_dtw(nome), esperado)

    def test_preset_desconhecido_falha(self):
        with self.assertRaises(transcreve.Falha) as ctx:
            transcreve.preset_dtw("meu-modelo-caseiro")
        self.assertIn("--palavras", str(ctx.exception))


class RodaWhisperNomeComPonto(unittest.TestCase):
    """Exercita o roda_whisper de verdade (só o subprocess é falso): prova que
    "aula 1.2" acha "aula 1.2.srt" e não "aula 1.srt"."""

    def test_roda_whisper_nome_com_ponto(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            prefixo = tmp / "aula 1.2"

            def whisper_escreve(cmd, check):
                alvo = cmd[cmd.index("-of") + 1]
                Path(f"{alvo}.srt").write_text("", encoding="utf-8")
                if "-ojf" in cmd:
                    Path(f"{alvo}.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(transcreve, "exige", return_value="whisper-cli"), \
                    mock.patch.object(transcreve.subprocess, "run", whisper_escreve):
                srt = transcreve.roda_whisper(Path("a.wav"), Path("m.bin"), "pt",
                                              "p", prefixo, palavras=True)
            self.assertEqual(srt, tmp / "aula 1.2.srt")
            self.assertTrue((tmp / "aula 1.2.json").exists())
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
