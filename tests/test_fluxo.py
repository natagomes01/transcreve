import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import transcreve  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "whisper_ojf.json"
SRT = "1\n00:00:00,000 --> 00:00:01,000\nO rei chegou.\n\n"


def whisper_falso(json_origem: Path):
    """Substitui roda_whisper: escreve o .srt e, se pedido, o .json."""
    def falso(wav, modelo, idioma, prompt, prefixo, palavras=False,
              modelo_nome="large-v3-turbo"):
        srt = Path(f"{prefixo}.srt")
        srt.write_text(SRT, encoding="utf-8")
        if palavras:
            shutil.copy(json_origem, Path(f"{prefixo}.json"))
        return srt
    return falso


class Fluxo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.video = self.tmp / "aula.mp4"
        self.video.write_bytes(b"")
        self.patches = [
            mock.patch.object(transcreve, "garante_modelo",
                              return_value=Path("m.bin")),
            mock.patch.object(transcreve, "extrai_audio"),
            mock.patch.object(transcreve, "duracao_segundos",
                              return_value=3.5),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def test_com_palavras_grava_json_ao_lado_do_md(self):
        with mock.patch.object(transcreve, "roda_whisper",
                               whisper_falso(FIXTURE)):
            md = transcreve.transcreve(self.video, "large-v3-turbo", "pt",
                                       "p", None, False, palavras=True)
        self.assertTrue(md.exists())
        destino = self.tmp / "aula.palavras.json"
        dados = json.loads(destino.read_text(encoding="utf-8"))
        self.assertEqual(list(dados), ["version", "model", "language",
                                       "dtw_shift_s", "duration_s", "words"])
        self.assertEqual(dados["version"], 1)
        self.assertEqual(dados["model"], "large-v3-turbo")
        self.assertEqual(dados["language"], "pt")
        self.assertEqual(dados["dtw_shift_s"], -0.15)
        self.assertEqual(dados["duration_s"], 3.5)
        self.assertEqual(dados["words"][1], {"t": 0.25, "p": 0.9, "w": "rei"})

    def test_json_sai_em_utf8_legivel(self):
        origem = self.tmp / "acento.json"
        dados = json.loads(FIXTURE.read_text(encoding="utf-8"))
        dados["transcription"][0]["tokens"][4]["text"] = " coração"
        origem.write_text(json.dumps(dados), encoding="utf-8")
        with mock.patch.object(transcreve, "roda_whisper",
                               whisper_falso(origem)):
            transcreve.transcreve(self.video, "large-v3-turbo", "pt", "p",
                                  None, False, palavras=True)
        texto = (self.tmp / "aula.palavras.json").read_text(encoding="utf-8")
        self.assertIn("coração", texto)

    def test_json_com_bytes_partidos(self):
        # "ç" (C3 A7) partido entre dois tokens, gravado como bytes crus
        dados = json.loads(FIXTURE.read_text(encoding="utf-8"))
        toks = dados["transcription"][0]["tokens"]
        toks[4]["text"] = " coraXPARTE1"
        toks[5]["text"] = "XPARTE2ão"
        cru = (json.dumps(dados, ensure_ascii=False).encode("utf-8")
               .replace(b"XPARTE1", b"\xc3").replace(b"XPARTE2", b"\xa7"))
        origem = self.tmp / "partido.json"
        origem.write_bytes(cru)
        with mock.patch.object(transcreve, "roda_whisper",
                               whisper_falso(origem)):
            transcreve.transcreve(self.video, "large-v3-turbo", "pt", "p",
                                  None, False, palavras=True)
        texto = (self.tmp / "aula.palavras.json").read_text(encoding="utf-8")
        self.assertIn("coração", texto)

    def test_respeita_saida(self):
        saida = self.tmp / "outra"
        with mock.patch.object(transcreve, "roda_whisper",
                               whisper_falso(FIXTURE)):
            transcreve.transcreve(self.video, "large-v3-turbo", "pt", "p",
                                  saida, False, palavras=True)
        self.assertTrue((saida / "aula.palavras.json").exists())
        self.assertTrue((saida / "aula.md").exists())

    def test_sem_palavras_nao_grava_json(self):
        with mock.patch.object(transcreve, "roda_whisper",
                               whisper_falso(FIXTURE)):
            transcreve.transcreve(self.video, "large-v3-turbo", "pt", "p",
                                  None, False)
        self.assertFalse((self.tmp / "aula.palavras.json").exists())

    def test_sem_fala_grava_lista_vazia(self):
        vazio = self.tmp / "vazio.json"
        vazio.write_text(json.dumps({"transcription": []}), encoding="utf-8")
        with mock.patch.object(transcreve, "roda_whisper",
                               whisper_falso(vazio)):
            transcreve.transcreve(self.video, "large-v3-turbo", "pt", "p",
                                  None, False, palavras=True)
        dados = json.loads((self.tmp / "aula.palavras.json")
                           .read_text(encoding="utf-8"))
        self.assertEqual(dados["words"], [])

    def test_dtw_quebrado_nao_grava_nada_de_palavras(self):
        ruim = self.tmp / "ruim.json"
        dados = json.loads(FIXTURE.read_text(encoding="utf-8"))
        dados["transcription"][1]["tokens"][1]["t_dtw"] = -1
        ruim.write_text(json.dumps(dados), encoding="utf-8")
        with mock.patch.object(transcreve, "roda_whisper",
                               whisper_falso(ruim)):
            with self.assertRaises(transcreve.Falha):
                transcreve.transcreve(self.video, "large-v3-turbo", "pt",
                                      "p", None, False, palavras=True)
        self.assertFalse((self.tmp / "aula.palavras.json").exists())

    def test_nome_com_ponto(self):
        video = self.tmp / "aula 1.2.mp4"
        video.write_bytes(b"")
        with mock.patch.object(transcreve, "roda_whisper",
                               whisper_falso(FIXTURE)):
            transcreve.transcreve(video, "large-v3-turbo", "pt", "p",
                                  None, False, palavras=True)
        self.assertTrue((self.tmp / "aula 1.2.palavras.json").exists())
        self.assertTrue((self.tmp / "aula 1.2.md").exists())

    def test_pasta_com_um_arquivo_falhando_continua(self):
        ruim = self.tmp / "ruim.json"
        dados = json.loads(FIXTURE.read_text(encoding="utf-8"))
        dados["transcription"][1]["tokens"][1]["t_dtw"] = -1
        ruim.write_text(json.dumps(dados), encoding="utf-8")
        (self.tmp / "b.mp4").write_bytes(b"")
        chamadas = {"n": 0}

        def alterna(wav, modelo, idioma, prompt, prefixo, palavras=False,
                    modelo_nome="large-v3-turbo"):
            chamadas["n"] += 1
            origem = ruim if chamadas["n"] == 1 else FIXTURE
            return whisper_falso(origem)(wav, modelo, idioma, prompt,
                                         prefixo, palavras, modelo_nome)

        argv = ["transcreve.py", str(self.tmp), "--palavras"]
        with mock.patch.object(transcreve, "roda_whisper", alterna), \
                mock.patch.object(sys, "argv", argv):
            codigo = transcreve.main()
        self.assertEqual(codigo, 1)
        self.assertEqual(chamadas["n"], 2)
        self.assertFalse((self.tmp / "aula.palavras.json").exists())
        self.assertTrue((self.tmp / "b.palavras.json").exists())


if __name__ == "__main__":
    unittest.main()
