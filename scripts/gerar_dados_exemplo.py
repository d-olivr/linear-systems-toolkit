"""
Script auxiliar util para testar a Questao 1 sem precisar montar os arquivos .xlsx manualmente

Gera 4 arquivos .xlsx de exemplo na pasta ../dados_exemplo/:
  matriz_A.xlsx, vetor_b.xlsx, vetor_c.xlsx, matriz_P.xlsx

Esses arquivos correspondem ao mesmo cenario usado nos testes automatizados
e na demonstracao do questao1_sistemas_lineares.py, entao servem como dados validos de exemplo.

Uso:
    python gerar_dados_exemplo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PASTA_SAIDA = Path(__file__).resolve().parent.parent / "dados_exemplo"

A = np.array([
    [4., 1., 0.],
    [1., 3., 1.],
    [0., 1., 2.],
])

b = np.array([1., 2., 3.])
c = np.array([3., 1., -1.])
P = np.eye(3)


def salvar(matriz, nome_arquivo):
    PASTA_SAIDA.mkdir(exist_ok=True)
    caminho = PASTA_SAIDA / nome_arquivo
    pd.DataFrame(matriz).to_excel(caminho, header=False, index=False)
    print(f"  Gerado: {caminho}  (shape={np.array(matriz).shape})")


if __name__ == "__main__":
    print("Gerando arquivos de exemplo para a Questao 1...")
    salvar(A, "matriz_A.xlsx")
    salvar(b, "vetor_b.xlsx")
    salvar(c, "vetor_c.xlsx")
    salvar(P, "matriz_P.xlsx")
    print("Pronto. Use main() em questao1_sistemas_lineares.py para ler esses arquivos")