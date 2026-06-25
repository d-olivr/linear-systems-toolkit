<div align="center">

# 🧮 Linear Systems Toolkit
### *Substituição, LU, Jacobi/Gauss-Seidel, Normas, Condicionamento & QR*

**Projeto de Álgebra Linear Computacional · UFRRJ**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Álgebra_Linear-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-I%2FO_Excel-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Notebooks-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Status](https://img.shields.io/badge/Status-Completo-brightgreen?style=flat-square)]()

</div>

---

## 📌 Overview

Implementação de um conjunto de algoritmos numéricos para resolver sistemas lineares `Ax = b` e analisar suas propriedades, feita para a Questão 1 do Trabalho de Álgebra Linear Computacional

Tudo gira em torno da classe `SistemaLinear`, que recebe a matriz `A`, o vetor `b` e os auxiliares `c` (vetor) e `P` (matriz) uma única vez, e expõe um método para cada item do enunciado. O código foi pensado assim para que os dados não precisem ser passados repetidamente entre chamadas

Disponível em duas versões equivalentes:
- **`scripts/sistemas_lineares.py`** 
- **`notebooks/sistemas_lineares.ipynb`**

O código foi desenvolvido inicialmente como um script Python no VSCode, para trabalhar no ambiente de desenvolvimento padrão e ter espaço para testar e estudar o que foi necessário ao longo do projeto. Só depois desse script estar completo, testado e entendido, traduzi o conteúdo no notebook, para atender ao formato .ipynb solicitado pelo professor.

---

## 🧠 O que foi implementado

| Item | Descrição |
|------|-----------|
| **(a)** | Substituição para frente (`Ly=b`) e para trás (`Ux=y`), implementadas do zero |
| **(b)** | Fatoração LU via eliminação gaussiana sem pivoteamento |
| **(c)** | Métodos de Jacobi e Gauss-Seidel, com verificação de dominância diagonal estrita |
| **(d)** | Normas matriciais de `A`: 1, ∞, Frobenius, 2 (espectral) |
| **(e)** | Normas vetoriais de `b`: 1, 2, p genérica, ∞, induzida por `P` (com checagem de SPD) |
| **(f)** | Número de condição exato (norma 2, via valores singulares) e estimado (norma 1) |
| **(g)** | Produto interno `<b,c>` e distância euclidiana `d(b,c)` |
| **(h)** | Distância entre `A` e `P` em três normas (Frobenius, 1, ∞) |
| **(i)** | Fatoração QR via Gram-Schmidt clássico |

### Decisões de design

- A fatoração QR usa Gram-Schmidt **clássico** (não modificado) — menos estável numericamente que a versão modificada ou Householder, mas mais direto de explicar passo a passo no vídeo da prova;
- Jacobi e Gauss-Seidel verificam dominância diagonal estrita antes de rodar e **avisam** o usuário quando a convergência não é garantida pela teoria, mas tentam executar mesmo assim;
- A leitura dos `.xlsx` assume arquivos **sem cabeçalho** (a primeira linha já é dado numérico, não um rótulo de coluna);
- A norma "induzida por P" (`sqrt(bᵀPb)`) exige que `P` seja simétrica positiva definida — o código testa isso via tentativa de fatoração de Cholesky e avisa quando a matriz fornecida não satisfaz a condição, mas calcula o valor de qualquer forma.

---

## 📐 Hipóteses assumidas

- `A` é quadrada (n×n) e não-singular;
- `b`, `c` e `P` têm dimensões compatíveis com `A` (`b`, `c` ∈ ℝⁿ; `P` ∈ ℝⁿˣⁿ);
- A fatoração LU sem pivoteamento é suficiente para os casos de teste propostos — matrizes com pivô nulo (ou quase) durante a eliminação geram um erro tratado, explicando o problema;
- Os arquivos `.xlsx` de entrada não têm linha de cabeçalho.

A documentação completa de hipóteses está nos comentários no topo de cada arquivo (`.py` e primeira célula do `.ipynb`).

---

## 🗂️ Estrutura do Projeto

```
linear-systems-toolkit/
│
├── notebooks/
│   └── 📓 sistemas_lineares.ipynb
│
├── scripts/
│   ├── 🐍 sistemas_lineares.py
│   └── 🐍 gerar_dados_exemplo.py
│
├── dados_exemplo/
│   ├── matriz_A.xlsx
│   ├── vetor_b.xlsx
│   ├── vetor_c.xlsx
│   └── matriz_P.xlsx
│
├── requirements.txt
└── README.md
```

> Os arquivos em `dados_exemplo/` são gerados por `scripts/gerar_dados_exemplo.py` e servem como exemplo de entrada válida

---

## ✅ Testes e validação

O script roda uma bateria de 6 testes automatizados (`testes()`) como ponto de entrada padrão, sem precisar de input do usuário:

1. Comparação dos resultados (LU, Jacobi, Gauss-Seidel) com `numpy.linalg.solve` como gabarito;
2. Matriz 4×4 genérica, validando `A = L@U` e `A = Q@R` com `Q` ortogonal;
3. Produto interno de vetores ortogonais (deve ser ≈ 0);
4. Matriz propositalmente mal condicionada (κ deve ser grande);
5. Matriz `P` inválida (não-SPD) — deve avisar mas não quebrar o cálculo;
6. Matriz com pivô nulo — deve lançar um erro tratado e explicativo, não travar silenciosamente.

O notebook já foi executado de ponta a ponta no Jupyter/Colab antes da entrega — todas as células rodam sem erro.

---

## 🛠️ Tecnologias

| Tool | Finalidade |
|------|------------|
| **Python 3.11+** | Linguagem principal |
| **NumPy** | Álgebra linear (normas, número de condição, Cholesky, SVD) |
| **Pandas** | Leitura/escrita dos arquivos `.xlsx` |
| **openpyxl** | Engine usada pelo Pandas para ler/escrever `.xlsx` |
| **Google Colab** | Ambiente de execução do notebook |

---

## 🚀 Como rodar

```bash
# Clonar o repositório
git clone https://github.com/d-olivr/linear-systems-toolkit.git
cd linear-systems-toolkit

# Instalar dependências
pip install -r requirements.txt

# Gerar os dados de exemplo
python scripts/gerar_dados_exemplo.py

# Rodar o script (executa os testes automaticamente e depois a demonstração)
python scripts/sitemas_lineares.py
```

Ou abra `notebooks/sistemas_lineares.ipynb` direto no Google Colab (`Arquivo -> Fazer upload de notebook`).

Para rodar com seus próprios dados em vez dos arquivos de exemplo, use a função `main()` no final do script — ela lê `matriz_A.xlsx`, `vetor_b.xlsx`, `vetor_c.xlsx` e `matriz_P.xlsx` da pasta `dados_exemplo/`.

---

## 🤖 Uso de LLM

Conforme exigido no enunciado da prova, o uso de LLM (Claude, da Anthropic) está declarado diretamente na primeira célula do notebook, detalhando especificamente em que pontos a IA foi usada (sugestão de estruturação do código em classe, revisão de corretude matemática das fórmulas, escrita dos testes e da documentação). O raciocínio matemático foi compreendido e validado manualmente antes da entrega.

Também utilizei LLM para gerar este README! Mandei um README que gostei bastante de outro projeto meu, e pedi que fosse adaptado para esse projeto. Com revisão manual posteriormente, claro. :)


