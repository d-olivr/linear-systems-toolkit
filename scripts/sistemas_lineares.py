import numpy as np
import pandas as pd


# ENTRADA/SAIDA

def carregar_matriz_excel(caminho: str) -> np.ndarray:
    """Le uma matriz numerica de um arquivo .xlsx sem cabecalho."""
    df = pd.read_excel(caminho, header=None)
    return df.to_numpy(dtype=float)


def carregar_vetor_excel(caminho: str) -> np.ndarray:
    """Le um vetor (linha ou coluna) de um arquivo .xlsx sem cabecalho."""
    df = pd.read_excel(caminho, header=None)
    return df.to_numpy(dtype=float).ravel()


def linha(char="-", n=78):
    print(char * n)


def titulo(texto):
    linha("=")
    print(texto)
    linha("=")


def subtitulo(letra, texto):
    print(f"\n>> Item ({letra}) {texto}")
    linha("-")


def tabela(matriz: np.ndarray, nome: str = "") -> pd.DataFrame:
    """Devolve a matriz como DataFrame para impressao tabular legivel."""
    n_linhas = matriz.shape[0]
    n_colunas = matriz.shape[1] if matriz.ndim > 1 else 1
    df = pd.DataFrame(
        matriz.reshape(n_linhas, n_colunas) if matriz.ndim == 1 else matriz,
        index=[f"L{i}" for i in range(n_linhas)],
        columns=[f"C{j}" for j in range(n_colunas)],
    )
    if nome:
        print(f"\n{nome} =")
    print(df.round(6).to_string())
    return df


# ============================================================================
# CLASSE PRINCIPAL: encapsula A, b, c, P e todos os metodos pedidos
# ============================================================================

class SistemaLinear:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, P: np.ndarray):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).ravel()
        c = np.asarray(c, dtype=float).ravel()
        P = np.asarray(P, dtype=float)

        n, m = A.shape
        if n != m:
            raise ValueError(f"A precisa ser quadrada, recebida com formato {A.shape}.")
        if b.shape[0] != n:
            raise ValueError(f"b precisa ter {n} elementos, recebido com {b.shape[0]}.")
        if c.shape[0] != n:
            raise ValueError(f"c precisa ter {n} elementos, recebido com {c.shape[0]}.")
        if P.shape != (n, n):
            raise ValueError(f"P precisa ser {n}x{n}, recebida com formato {P.shape}.")

        self.A = A
        self.b = b
        self.c = c
        self.P = P
        self.n = n

    # (a) Substituicao para frente e para tras
    @staticmethod
    def substituicao_frente(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Resolve L y = b para L triangular inferior, varrendo i = 0..n-1.
        Em cada passo, x[i] depende apenas de valores ja calculados
        (x[0..i-1]), entao isolamos x[i] diretamente:

            x[i] = ( b[i] - sum_{j<i} L[i,j]*x[j] ) / L[i,i]
        """
        n = b.shape[0]
        x = np.zeros(n)
        for i in range(n):
            acumulado = 0.0
            for j in range(i):
                acumulado += L[i, j] * x[j]
            x[i] = (b[i] - acumulado) / L[i, i]
        return x

    @staticmethod
    def substituicao_tras(U: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Resolve U x = y para U triangular superior, varrendo i = n-1..0.
        Espelha a logica da substituicao para frente, mas de baixo para
        cima, ja que aqui x[i] depende dos valores x[i+1..n-1]:

            x[i] = ( y[i] - sum_{j>i} U[i,j]*x[j] ) / U[i,i]
        """
        n = y.shape[0]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            acumulado = 0.0
            for j in range(i + 1, n):
                acumulado += U[i, j] * x[j]
            x[i] = (y[i] - acumulado) / U[i, i]
        return x

    # (b) Fatoracao LU (eliminacao gaussiana sem pivoteamento)
    def fatorar_lu(self):
        """
        Decompoe A = L @ U.

        A ideia e eliminar, coluna por coluna, os elementos abaixo da
        diagonal usando combinacoes lineares de linhas. O multiplicador
        usado para zerar a posicao (i, k) e guardado em L[i, k] -- e
        exatamente esse numero que, se "desfeito", reconstroi a linha
        original. U comeca como copia de A e vai sendo escalonada;
        L comeca como identidade (diagonal 1) e recebe os multiplicadores.
        """
        A = self.A
        n = self.n
        U = A.copy()
        L = np.eye(n)

        for coluna_pivo in range(n - 1):
            pivo = U[coluna_pivo, coluna_pivo]
            if abs(pivo) < 1e-13:
                raise ZeroDivisionError(
                    f"Pivo nulo (ou quase nulo) na coluna {coluna_pivo}. "
                    "Fatoracao LU sem pivoteamento nao e aplicavel a esta matriz."
                )
            for linha_alvo in range(coluna_pivo + 1, n):
                multiplicador = U[linha_alvo, coluna_pivo] / pivo
                L[linha_alvo, coluna_pivo] = multiplicador
                U[linha_alvo, coluna_pivo:] -= multiplicador * U[coluna_pivo, coluna_pivo:]

        return L, U

    def resolver_via_lu(self):
        """Combina fatoracao LU + as duas substituicoes para obter x em Ax=b."""
        L, U = self.fatorar_lu()
        y = self.substituicao_frente(L, self.b)
        x = self.substituicao_tras(U, y)
        return x, L, U, y

    # (c) Jacobi e Gauss-Seidel
    def _diagonal_dominante(self) -> bool:
        """Testa dominancia diagonal estrita por linha: |a_ii| > soma dos demais |a_ij|."""
        A = self.A
        soma_resto = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
        return bool(np.all(np.abs(np.diag(A)) > soma_resto))

    def jacobi(self, tol=1e-9, max_iter=500, x0=None, verbose=True):
        """
        Metodo de Jacobi: toda a atualizacao da iteracao k+1 usa SOMENTE
        valores da iteracao k (atualizacao "em bloco", todos simultaneos).

            x_novo[i] = ( b[i] - sum_{j != i} A[i,j] * x_velho[j] ) / A[i,i]

        Convergencia garantida (pela teoria classica) se A for
        estritamente diagonal dominante; o codigo avisa quando isso falha
        mas roda do mesmo jeito.
        """
        A, b, n = self.A, self.b, self.n
        if not self._diagonal_dominante() and verbose:
            print("  [aviso] A nao e estritamente diagonal dominante -> "
                  "convergencia de Jacobi nao e garantida pela teoria.")

        x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
        historico = [x.copy()]

        for it in range(1, max_iter + 1):
            x_novo = np.zeros(n)
            for i in range(n):
                soma = 0.0
                for j in range(n):
                    if j != i:
                        soma += A[i, j] * x[j]
                x_novo[i] = (b[i] - soma) / A[i, i]

            historico.append(x_novo.copy())
            erro = np.linalg.norm(x_novo - x, ord=np.inf)
            x = x_novo
            if erro < tol:
                if verbose:
                    print(f"  Jacobi convergiu em {it} iteracoes (erro={erro:.2e}).")
                return x, historico, it
        if verbose:
            print(f"  [aviso] Jacobi NAO convergiu em {max_iter} iteracoes.")
        return x, historico, max_iter

    def gauss_seidel(self, tol=1e-9, max_iter=500, x0=None, verbose=True):
        """
        Metodo de Gauss-Seidel: igual a Jacobi, mas cada x[i] e atualizado
        "no lugar" e usado IMEDIATAMENTE nas componentes seguintes da MESMA
        iteracao. Isso costuma acelerar a convergencia em relacao a Jacobi:

            x[i] = ( b[i] - sum_{j<i} A[i,j]*x_novo[j]
                          - sum_{j>i} A[i,j]*x_velho[j] ) / A[i,i]
        """
        A, b, n = self.A, self.b, self.n
        if not self._diagonal_dominante() and verbose:
            print("  [aviso] A nao e estritamente diagonal dominante -> "
                  "convergencia de Gauss-Seidel nao e garantida pela teoria.")

        x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
        historico = [x.copy()]

        for it in range(1, max_iter + 1):
            x_anterior = x.copy()
            for i in range(n):
                soma = 0.0
                for j in range(n):
                    if j != i:
                        soma += A[i, j] * x[j]
                x[i] = (b[i] - soma) / A[i, i]

            historico.append(x.copy())
            erro = np.linalg.norm(x - x_anterior, ord=np.inf)
            if erro < tol:
                if verbose:
                    print(f"  Gauss-Seidel convergiu em {it} iteracoes (erro={erro:.2e}).")
                return x, historico, it
        if verbose:
            print(f"  [aviso] Gauss-Seidel NAO convergiu em {max_iter} iteracoes.")
        return x, historico, max_iter

    # (d) Normas matriciais de A
    def normas_matriciais(self) -> dict:
        """
        Calcula as principais normas matriciais de A:
          - norma 1        : maior soma de modulos por COLUNA
          - norma infinito  : maior soma de modulos por LINHA
          - norma Frobenius : raiz da soma dos quadrados de todas as entradas
          - norma 2 (espectral): maior valor singular de A
        Usa numpy.linalg.norm internamente -- a questao pede para "retornar"
        essas normas, nao para reimplementar a algebra linear ja consolidada
        do numpy.
        """
        A = self.A
        return {
            "norma_1": float(np.linalg.norm(A, 1)),
            "norma_infinito": float(np.linalg.norm(A, np.inf)),
            "norma_frobenius": float(np.linalg.norm(A, "fro")),
            "norma_2_espectral": float(np.linalg.norm(A, 2)),
        }

    # (e) Normas vetoriais de b
    def _eh_positiva_definida(self, M: np.ndarray) -> bool:
        """Testa positividade definida via tentativa de fatoracao de Cholesky."""
        if not np.allclose(M, M.T, atol=1e-10):
            return False
        try:
            np.linalg.cholesky(M)
            return True
        except np.linalg.LinAlgError:
            return False

    def normas_vetoriais(self, p: int = 3) -> dict:
        """
        Calcula as normas vetoriais de b:
          - norma 1        : soma dos modulos
          - norma 2        : norma euclidiana
          - norma p        : generalizacao (Σ|b_i|^p)^(1/p), p configuravel
          - norma infinito : maior modulo entre as componentes
          - norma induzida por P: sqrt(b^T P b), exige P simetrica positiva
            definida (SPD). Caso a P fornecida nao seja SPD, o resultado
            ainda e calculado mas o usuario e avisado da inconsistencia.
        """
        b, P = self.b, self.P
        spd = self._eh_positiva_definida(P)
        if not spd:
            print("  [aviso] A matriz P fornecida nao e simetrica positiva definida; "
                  "a 'norma induzida por P' calculada abaixo nao tem garantia "
                  "de ser uma norma valida (pode ser negativa sob a raiz, por exemplo).")

        valor_quadratico = float(b @ P @ b)
        norma_inducida = float(np.sqrt(valor_quadratico)) if valor_quadratico >= 0 else float("nan")

        return {
            "norma_1": float(np.linalg.norm(b, 1)),
            "norma_2": float(np.linalg.norm(b, 2)),
            f"norma_p (p={p})": float(np.linalg.norm(b, p)),
            "norma_infinito": float(np.linalg.norm(b, np.inf)),
            "norma_induzida_por_P": norma_inducida,
            "P_e_SPD": spd,
        }

    # (f) Numero de condicao de A (exato + estimativa)
    def numero_condicao(self) -> dict:
        """
        kappa(A) mede a sensibilidade da solucao de Ax=b a perturbacoes em
        A ou b. Quanto maior, mais "instavel" numericamente o sistema.

        - exato (norma 2): kappa_2(A) = sigma_max / sigma_min (razao entre
          o maior e o menor valor singular de A). Calculado via
          np.linalg.cond, que e o jeito direto de obter isso.
        - estimativa (norma 1): kappa_1(A) = ||A||_1 * ||A^-1||_1. E chamada
          de "estimativa" porque, na pratica numerica, evitar a inversa
          explicita (e usar tecnicas de estimativa de condicao, como a usada
          internamente em LAPACK) e mais barato computacionalmente do que
          inverter A de fato -- aqui inverter A diretamente serve para
          ilustrar o conceito, mas em sistemas grandes seria substituido
          por um estimador iterativo.
        """
        A = self.A
        kappa_exato = float(np.linalg.cond(A, 2))
        A_inv = np.linalg.inv(A)
        kappa_estimado = float(np.linalg.norm(A, 1) * np.linalg.norm(A_inv, 1))

        if kappa_exato < 10:
            classificacao = "bem condicionada"
        elif kappa_exato < 1e4:
            classificacao = "moderadamente condicionada"
        else:
            classificacao = "mal condicionada (resultados sensiveis a erros numericos)"

        return {
            "kappa_exato_norma2": kappa_exato,
            "kappa_estimado_norma1": kappa_estimado,
            "classificacao": classificacao,
        }

    # (g) Produto interno e distancia entre b e c
    def produto_interno_bc(self) -> float:
        """<b, c> = soma(b_i * c_i). Produto interno canonico de R^n."""
        return float(np.dot(self.b, self.c))

    def distancia_bc(self) -> float:
        """d(b, c) = ||b - c||_2, distancia euclidiana entre os dois vetores."""
        return float(np.linalg.norm(self.b - self.c, 2))

    # (h) Distancia entre A e P
    def distancia_matrizes(self) -> dict:
        """
        Distancia entre A e P usando a diferenca D = A - P avaliada em
        tres normas matriciais distintas, cada uma com um significado
        proprio:
          - Frobenius: "distancia euclidiana" entre as matrizes vistas
            como vetores empilhados -- captura a diferenca GLOBAL.
          - norma 1: maior diferenca acumulada em uma UNICA coluna.
          - norma infinito: maior diferenca acumulada em uma UNICA linha.
        """
        D = self.A - self.P
        return {
            "distancia_frobenius": float(np.linalg.norm(D, "fro")),
            "distancia_norma_1": float(np.linalg.norm(D, 1)),
            "distancia_norma_infinito": float(np.linalg.norm(D, np.inf)),
        }

    # (i) Fatoracao QR via Gram-Schmidt
    def fatorar_qr(self):
        """
        Decompoe A = Q @ R via Gram-Schmidt classico:
          - Cada coluna a_k de A e ortogonalizada contra as colunas
            ortonormais q_0..q_{k-1} JA calculadas (remove as projecoes);
          - o que resta e normalizado para virar q_k;
          - os coeficientes de projecao formam a coluna k de R (que por
            construcao fica triangular superior).
        Resultado: Q tem colunas ortonormais (Q^T Q = I) e R e triangular
        superior, satisfazendo A = Q @ R.
        """
        A = self.A
        n = self.n
        Q = np.zeros((n, n))
        R = np.zeros((n, n))

        for k in range(n):
            v = A[:, k].copy()
            for j in range(k):
                R[j, k] = np.dot(Q[:, j], A[:, k])
                v -= R[j, k] * Q[:, j]
            norma_v = np.linalg.norm(v)
            if norma_v < 1e-13:
                raise ValueError(
                    f"Coluna {k} de A e linearmente dependente das anteriores "
                    "(ou quase) -- Gram-Schmidt nao consegue gerar uma base "
                    "ortonormal completa para esta matriz."
                )
            R[k, k] = norma_v
            Q[:, k] = v / norma_v

        return Q, R


# ============================================================================
# RELATORIO COMPLETO (roda todos os itens a-i em sequencia, com explicacoes)
# ============================================================================

def relatorio_completo(sistema: SistemaLinear):
    A, b, c, P, n = sistema.A, sistema.b, sistema.c, sistema.P, sistema.n

    titulo("QUESTAO 1 -- SISTEMAS LINEARES, FATORACOES E NORMAS")
    tabela(A, "Matriz A")
    tabela(b.reshape(-1, 1), "Vetor b")
    tabela(c.reshape(-1, 1), "Vetor c")
    tabela(P, "Matriz P")

    # --- (a) ---
    subtitulo("a", "Substituicao para frente e para tras")
    print("Estrategia: fatora A = L*U (item b) e resolve em duas etapas: Ly=b, depois Ux=y.")
    x_sub, L, U, y = sistema.resolver_via_lu()
    tabela(y.reshape(-1, 1), "y intermediario (Ly = b)")
    tabela(x_sub.reshape(-1, 1), "x final (Ux = y)")
    residuo = np.linalg.norm(A @ x_sub - b)
    print(f"Verificacao ||Ax - b|| = {residuo:.3e} (esperado: proximo de 0)")

    # --- (b) ---
    subtitulo("b", "Fatoracao LU")
    L, U = sistema.fatorar_lu()
    tabela(L, "L (triangular inferior, diagonal = 1)")
    tabela(U, "U (triangular superior)")
    print(f"Verificacao ||A - L@U||_F = {np.linalg.norm(A - L @ U):.3e} (esperado: proximo de 0)")

    # --- (c) ---
    subtitulo("c", "Metodo de Jacobi")
    x_jac, hist_jac, it_jac = sistema.jacobi()
    tabela(x_jac.reshape(-1, 1), "Solucao (Jacobi)")
    print(f"Numero de iteracoes: {it_jac}")

    subtitulo("c", "Metodo de Gauss-Seidel")
    x_gs, hist_gs, it_gs = sistema.gauss_seidel()
    tabela(x_gs.reshape(-1, 1), "Solucao (Gauss-Seidel)")
    print(f"Numero de iteracoes: {it_gs}")
    if it_gs <= it_jac:
        print(f"-> Gauss-Seidel convergiu em {it_jac - it_gs} iteracao(oes) a menos que Jacobi "
              "(comportamento esperado, pois reaproveita valores atualizados na mesma iteracao).")

    # --- (d) ---
    subtitulo("d", "Normas matriciais de A")
    normas_A = sistema.normas_matriciais()
    for nome, valor in normas_A.items():
        print(f"  {nome:22s} = {valor:.6f}")

    # --- (e) ---
    subtitulo("e", "Normas vetoriais de b")
    normas_b = sistema.normas_vetoriais()
    for nome, valor in normas_b.items():
        if isinstance(valor, bool):
            print(f"  {nome:22s} = {valor}")
        else:
            print(f"  {nome:22s} = {valor:.6f}")

    # --- (f) ---
    subtitulo("f", "Numero de condicao de A")
    cond = sistema.numero_condicao()
    print(f"  kappa exato (norma 2)     = {cond['kappa_exato_norma2']:.6e}")
    print(f"  kappa estimado (norma 1)  = {cond['kappa_estimado_norma1']:.6e}")
    print(f"  classificacao             = {cond['classificacao']}")

    # --- (g) ---
    subtitulo("g", "Produto interno e distancia entre b e c")
    pi = sistema.produto_interno_bc()
    dist_bc = sistema.distancia_bc()
    print(f"  <b, c>   = {pi:.6f}" + ("  (vetores ortogonais)" if abs(pi) < 1e-9 else ""))
    print(f"  d(b, c)  = {dist_bc:.6f}")

    # --- (h) ---
    subtitulo("h", "Distancia entre A e P")
    dist_AP = sistema.distancia_matrizes()
    for nome, valor in dist_AP.items():
        print(f"  {nome:24s} = {valor:.6f}")

    # --- (i) ---
    subtitulo("i", "Fatoracao QR (Gram-Schmidt)")
    Q, R = sistema.fatorar_qr()
    tabela(Q, "Q (colunas ortonormais)")
    tabela(R, "R (triangular superior)")
    print(f"Verificacao ||A - Q@R||_F   = {np.linalg.norm(A - Q @ R):.3e} (esperado: proximo de 0)")
    print(f"Verificacao ||Q^T@Q - I||_F = {np.linalg.norm(Q.T @ Q - np.eye(n)):.3e} (esperado: proximo de 0)")

    linha("=")
    print("FIM DO RELATORIO -- QUESTAO 1")
    linha("=")


# ============================================================================
# TESTES AUTOMATIZADOS (varios cenarios, sem depender de arquivo Excel)
# ============================================================================

def testes():
    print("\n" + "#" * 78)
    print("# BLOCO DE TESTES AUTOMATIZADOS -- QUESTAO 1")
    print("#" * 78)

    # Teste 1: caso classico, diagonal dominante (Jacobi/Gauss-Seidel convergem)
    print("\n--- Teste 1: matriz 3x3 diagonal dominante ---")
    A1 = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
    b1 = np.array([1., 2., 3.])
    c1 = np.array([3., 1., -1.])
    P1 = np.eye(3)
    s1 = SistemaLinear(A1, b1, c1, P1)
    x_lu, *_ = s1.resolver_via_lu()
    x_jac, _, _ = s1.jacobi(verbose=False)
    x_gs, _, _ = s1.gauss_seidel(verbose=False)
    x_np = np.linalg.solve(A1, b1)
    print(f"  x (LU)          = {np.round(x_lu, 6)}")
    print(f"  x (Jacobi)      = {np.round(x_jac, 6)}")
    print(f"  x (Gauss-Seidel)= {np.round(x_gs, 6)}")
    print(f"  x (numpy.solve) = {np.round(x_np, 6)}  <- gabarito de referencia")
    assert np.allclose(x_lu, x_np, atol=1e-8), "FALHOU: LU nao bate com numpy.solve"
    assert np.allclose(x_jac, x_np, atol=1e-6), "FALHOU: Jacobi nao convergiu para o valor certo"
    assert np.allclose(x_gs, x_np, atol=1e-6), "FALHOU: Gauss-Seidel nao convergiu para o valor certo"
    print("  OK: LU, Jacobi e Gauss-Seidel concordam com numpy.linalg.solve.")

    # Teste 2: matriz 4x4 generica (testa LU e QR em dimensao maior)
    print("\n--- Teste 2: matriz 4x4 generica ---")
    rng = np.random.default_rng(42)
    A2 = rng.uniform(1, 10, size=(4, 4)) + np.eye(4) * 20  # garante boa condicao
    b2 = rng.uniform(-5, 5, size=4)
    c2 = rng.uniform(-5, 5, size=4)
    P2 = np.eye(4) * 2
    s2 = SistemaLinear(A2, b2, c2, P2)
    L2, U2 = s2.fatorar_lu()
    assert np.allclose(A2, L2 @ U2, atol=1e-8), "FALHOU: A != L@U"
    Q2, R2 = s2.fatorar_qr()
    assert np.allclose(A2, Q2 @ R2, atol=1e-8), "FALHOU: A != Q@R"
    assert np.allclose(Q2.T @ Q2, np.eye(4), atol=1e-8), "FALHOU: Q nao e ortogonal"
    print("  OK: LU (A=LU) e QR (A=QR, Q ortogonal) corretos para matriz 4x4.")

    # Teste 3: vetores ortogonais (produto interno deve ser ~0)
    print("\n--- Teste 3: produto interno de vetores ortogonais ---")
    b3 = np.array([1., 0., 0.])
    c3 = np.array([0., 1., 0.])
    s3 = SistemaLinear(np.eye(3), b3, c3, np.eye(3))
    pi3 = s3.produto_interno_bc()
    assert abs(pi3) < 1e-12, "FALHOU: produto interno de vetores ortogonais deveria ser 0"
    print(f"  <b, c> = {pi3:.10f}  -> OK, vetores ortogonais confirmados.")

    # Teste 4: matriz mal condicionada (numero de condicao deve ser grande)
    print("\n--- Teste 4: matriz proxima de singular (mal condicionada) ---")
    A4 = np.array([[1., 1.], [1., 1.0001]])
    b4 = np.array([2., 2.0001])
    s4 = SistemaLinear(A4, b4, np.array([1., 1.]), np.eye(2))
    cond4 = s4.numero_condicao()
    assert cond4["kappa_exato_norma2"] > 1e3, "FALHOU: matriz deveria ser mal condicionada"
    print(f"  kappa = {cond4['kappa_exato_norma2']:.2e} -> OK, classificada como "
          f"'{cond4['classificacao']}'.")

    # Teste 5: P nao-SPD deve disparar aviso mas nao quebrar o calculo
    print("\n--- Teste 5: matriz P invalida (nao-SPD) ---")
    P5 = np.array([[0., 1.], [1., 0.]])  # simetrica, mas nao positiva definida
    s5 = SistemaLinear(np.eye(2), np.array([1., 2.]), np.array([2., 1.]), P5)
    normas5 = s5.normas_vetoriais()
    assert normas5["P_e_SPD"] is False, "FALHOU: deveria detectar que P nao e SPD"
    print("  OK: codigo detectou corretamente que P nao e SPD e avisou o usuario.")

    # Teste 6: pivo nulo deve ser detectado e reportado, nao travar silenciosamente
    print("\n--- Teste 6: matriz com pivo nulo (deve lancar erro tratado) ---")
    A6 = np.array([[0., 1.], [1., 1.]])
    s6 = SistemaLinear(A6, np.array([1., 2.]), np.array([1., 1.]), np.eye(2))
    try:
        s6.fatorar_lu()
        raise AssertionError("FALHOU: deveria ter lancado ZeroDivisionError")
    except ZeroDivisionError as e:
        print(f"  OK: erro tratado corretamente -> {e}")

    print("\n" + "#" * 78)
    print("# TODOS OS TESTES PASSARAM")
    print("#" * 78 + "\n")


# PONTO DE ENTRADA

def main():
    """
    Modo de uso com arquivos reais (.xlsx). Por padrao busca os 4 arquivos
    descritos no cabecalho do modulo na pasta ../dados_exemplo (relativa a
    este script)
    """
    from pathlib import Path
    pasta_dados = Path(__file__).resolve().parent.parent / "dados_exemplo"

    sistema = SistemaLinear(
        A=carregar_matriz_excel(pasta_dados / "matriz_A.xlsx"),
        b=carregar_vetor_excel(pasta_dados / "vetor_b.xlsx"),
        c=carregar_vetor_excel(pasta_dados / "vetor_c.xlsx"),
        P=carregar_matriz_excel(pasta_dados / "matriz_P.xlsx"),
    )
    relatorio_completo(sistema)


if __name__ == "__main__":
    testes()

main()