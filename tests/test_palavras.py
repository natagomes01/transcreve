import copy
import json
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import transcreve  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "whisper_ojf.json"


def carrega():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class PalavrasDeJson(unittest.TestCase):
    def test_junta_tokens_em_palavras(self):
        palavras = transcreve.palavras_de_json(carrega())
        self.assertEqual([p["w"] for p in palavras],
                         ["O", "rei", "chegou.", "Bom", "dia,", "amigos"])

    def test_tempo_vem_do_dtw_menos_15_centesimos(self):
        palavras = transcreve.palavras_de_json(carrega())
        self.assertEqual([p["t"] for p in palavras],
                         [0.0, 0.25, 0.55, 1.95, 2.25, 2.55])

    def test_primeira_palavra_nao_fica_negativa(self):
        # t_dtw 12 cs - 15 cs = -3 cs: vira 0.0, nunca negativo
        self.assertEqual(transcreve.palavras_de_json(carrega())[0]["t"], 0.0)

    def test_confianca_e_o_menor_p_dos_tokens(self):
        palavras = transcreve.palavras_de_json(carrega())
        por_texto = {p["w"]: p["p"] for p in palavras}
        self.assertEqual(por_texto["rei"], 0.9)      # min(0.95, 0.90)
        self.assertEqual(por_texto["chegou."], 0.97)  # min(0.99, 0.97)
        self.assertEqual(por_texto["dia,"], 0.7)      # min(0.80, 0.70); o token vazio é pulado

    def test_pontuacao_solta_gruda_na_anterior(self):
        palavras = transcreve.palavras_de_json(carrega())
        textos = [p["w"] for p in palavras]
        self.assertNotIn(",", textos)
        self.assertNotIn("", textos)

    def test_ignora_tokens_especiais(self):
        palavras = transcreve.palavras_de_json(carrega())
        self.assertFalse(any(p["w"].startswith("[_") for p in palavras))

    def test_sem_dtw_aborta_sem_cair_nos_offsets(self):
        dados = carrega()
        dados["transcription"][0]["tokens"][4]["t_dtw"] = -1  # " chegou"
        with self.assertRaises(transcreve.Falha) as ctx:
            transcreve.palavras_de_json(dados)
        self.assertIn("-nfa", str(ctx.exception))
        self.assertIn("chegou", str(ctx.exception))

    def test_sem_fala_devolve_lista_vazia(self):
        dados = {"transcription": [
            {"offsets": {"from": 0, "to": 1000}, "text": "", "tokens": [
                {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0},
                 "p": 0.9, "t_dtw": -1}]}]}
        self.assertEqual(transcreve.palavras_de_json(dados), [])

    def test_nao_altera_o_dicionario_de_entrada(self):
        dados = carrega()
        antes = copy.deepcopy(dados)
        transcreve.palavras_de_json(dados)
        self.assertEqual(dados, antes)

    def test_tempo_nunca_volta(self):
        dados = carrega()
        # " chegou" com DTW antes de " re": 30 cs - 15 = 15 cs < 25 cs de "rei"
        dados["transcription"][0]["tokens"][4]["t_dtw"] = 30
        tempos = [p["t"] for p in transcreve.palavras_de_json(dados)]
        self.assertEqual(tempos[:3], [0.0, 0.25, 0.25])
        self.assertEqual(tempos, sorted(tempos))

    def test_caractere_partido_em_dois_tokens(self):
        # "ç" = bytes C3 A7; o whisper pode partir entre dois tokens, e a
        # leitura com surrogateescape guarda cada byte como \udcXX.
        dados = {"transcription": [{"offsets": {"from": 0, "to": 900},
                                    "text": "", "tokens": [
            {"text": " cora", "offsets": {"from": 0, "to": 300}, "p": 0.9, "t_dtw": 40},
            {"text": "\udcc3", "offsets": {"from": 300, "to": 400}, "p": 0.8, "t_dtw": 60},
            {"text": "\udca7ão", "offsets": {"from": 400, "to": 900}, "p": 0.9, "t_dtw": 70},
        ]}]}
        palavras = transcreve.palavras_de_json(dados)
        self.assertEqual([p["w"] for p in palavras], ["coração"])

    def test_meio_caractere_sobrando_vira_substituto(self):
        dados = {"transcription": [{"offsets": {"from": 0, "to": 900},
                                    "text": "", "tokens": [
            {"text": " fim\udcc3", "offsets": {"from": 0, "to": 300}, "p": 0.9, "t_dtw": 40},
        ]}]}
        self.assertEqual(transcreve.palavras_de_json(dados)[0]["w"], "fim�")

    def test_abertura_vira_prefixo_da_proxima(self):
        dados = {"transcription": [{"offsets": {"from": 0, "to": 3000},
                                    "text": "", "tokens": [
            {"text": " anterior", "offsets": {"from": 0, "to": 500}, "p": 0.9, "t_dtw": 30},
            {"text": " (", "offsets": {"from": 500, "to": 600}, "p": 0.6, "t_dtw": 60},
            {"text": "ris", "offsets": {"from": 600, "to": 800}, "p": 0.9, "t_dtw": 70},
            {"text": "os", "offsets": {"from": 800, "to": 900}, "p": 0.9, "t_dtw": 80},
            {"text": ")", "offsets": {"from": 900, "to": 950}, "p": 0.9, "t_dtw": 90},
            {"text": " \"", "offsets": {"from": 950, "to": 1000}, "p": 0.9, "t_dtw": 100},
            {"text": " Deus", "offsets": {"from": 1000, "to": 1400}, "p": 0.9, "t_dtw": 110},
            {"text": "\"", "offsets": {"from": 1400, "to": 1450}, "p": 0.9, "t_dtw": 140},
        ]}]}
        palavras = transcreve.palavras_de_json(dados)
        self.assertEqual([p["w"] for p in palavras],
                         ["anterior", "(risos)", "\"Deus\""])
        # o tempo é o do primeiro token com letra, não o do "("
        self.assertEqual(palavras[1]["t"], 0.55)
        self.assertEqual(palavras[1]["p"], 0.6)

    def test_abertura_no_fim_gruda_na_anterior(self):
        dados = {"transcription": [{"offsets": {"from": 0, "to": 900},
                                    "text": "", "tokens": [
            {"text": " fim", "offsets": {"from": 0, "to": 300}, "p": 0.9, "t_dtw": 40},
            {"text": " (", "offsets": {"from": 300, "to": 400}, "p": 0.9, "t_dtw": 60},
        ]}]}
        self.assertEqual([p["w"] for p in transcreve.palavras_de_json(dados)],
                         ["fim("])


if __name__ == "__main__":
    unittest.main()
