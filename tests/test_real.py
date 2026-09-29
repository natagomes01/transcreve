"""Roda o whisper de verdade. Pulado a menos que SHORTS_SAMPLE aponte para um
vídeo ou áudio curto (alguns minutos) na máquina:

    SHORTS_SAMPLE=/caminho/amostra.mp4 python3 -m unittest tests.test_real -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import transcreve  # noqa: E402

AMOSTRA = os.environ.get("SHORTS_SAMPLE", "")


@unittest.skipUnless(AMOSTRA and Path(AMOSTRA).is_file(),
                     "defina SHORTS_SAMPLE com um vídeo curto para rodar")
class RodadaReal(unittest.TestCase):
    def test_palavras_com_tempo_valido_e_em_ordem(self):
        saida = Path(tempfile.mkdtemp())
        try:
            transcreve.transcreve(Path(AMOSTRA), "large-v3-turbo", "pt",
                                  transcreve.PROMPT_PADRAO, saida, False,
                                  palavras=True)
            destino = saida / f"{Path(AMOSTRA).stem}.palavras.json"
            dados = json.loads(destino.read_text(encoding="utf-8"))
            tempos = [p["t"] for p in dados["words"]]
            self.assertGreater(len(tempos), 0)
            self.assertTrue(all(t >= 0 for t in tempos))
            fora_de_ordem = [(a, b) for a, b in zip(tempos, tempos[1:])
                             if b < a]
            self.assertEqual(fora_de_ordem, [])
            self.assertLessEqual(tempos[-1], dados["duration_s"] + 1)
            self.assertEqual(dados["dtw_shift_s"], -0.15)
        finally:
            shutil.rmtree(saida)


if __name__ == "__main__":
    unittest.main()
