import numpy as np
import cmath
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTES NUMÉRICAS
# ==============================================================================

TOL_DEGENERADO = 1e-15  # s*t próximo de zero
TOL_ANGULO_90  = 1e-10  # proteção para ângulo de incidência próximo de 90°
TOL_K_ZERO     = 1e-12  # k_global considerado nulo
TOL_OMEGA_ZERO = 1e-12  # omega considerado nulo
TOL_DENOM      = 1e-30  # denominadores (D, sin) considerados nulos

# ==============================================================================
# 1. LEITURA E CONFIGURAÇÃO
# ==============================================================================

def ler_arquivo_entrada(nome_arquivo):
    """
    Lê e valida o arquivo JSON com a configuração do perfil de solo.

    Estrutura esperada::

        {
            "analise": {
                "omega"        : 10.0,
                "modo"         : "angulos",
                "valores"      : [30, 45, 60],
                "arquivo_saida": "meu_resultado.txt"   (opcional)
            },
            "camadas": [
                {"id": 1, "d": 10.0, "G": 20000.0, "nu": 0.33,
                 "zeta_p": 0.02, "zeta_s": 0.02, "rho": 2.0}
            ],
            "semi_espaco": {
                "G": 100000.0, "nu": 0.25,
                "zeta_p": 0.01, "zeta_s": 0.01, "rho": 2.4
            }
        }

    O campo ``valores`` aceita lista explícita ou dict de range::

        {"inicio": 0.001, "fim": 8.0, "passo": 0.001}

    Args:
        nome_arquivo (str): Caminho do arquivo JSON.

    Returns:
        dict: Dados completos do modelo.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        json.JSONDecodeError: Se o JSON for inválido.
        ValueError: Se algum campo obrigatório estiver ausente ou com tipo errado.
    """
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo '{nome_arquivo}' não encontrado.")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"JSON inválido em '{nome_arquivo}': {e.msg}", e.doc, e.pos
        )

    if "camadas" not in dados or not isinstance(dados["camadas"], list):
        raise ValueError("'camadas' ausente ou não é uma lista.")

    analise = dados.get("analise")
    if analise is None:
        raise ValueError("Bloco 'analise' ausente no JSON.")

    for campo, tipos in {"omega": (int, float), "modo": str}.items():
        if campo not in analise:
            raise ValueError(f"'analise.{campo}' ausente.")
        if not isinstance(analise[campo], tipos):
            raise ValueError(
                f"Tipo inválido em 'analise.{campo}': "
                f"esperado {tipos}, recebido {type(analise[campo]).__name__}."
            )

    if "valores" not in analise:
        raise ValueError("'analise.valores' ausente.")

    valores_raw = analise["valores"]
    if not isinstance(valores_raw, (list, dict)):
        raise ValueError("'analise.valores' deve ser list ou dict.")

    if isinstance(valores_raw, dict):
        for sub in ("inicio", "fim", "passo"):
            if sub not in valores_raw:
                raise ValueError(f"'analise.valores.{sub}' ausente.")
        if valores_raw["passo"] <= 0:
            raise ValueError("'analise.valores.passo' deve ser maior que zero.")
        if valores_raw["inicio"] >= valores_raw["fim"]:
            raise ValueError("'analise.valores.inicio' deve ser menor que 'fim'.")
    else:
        if len(valores_raw) == 0:
            raise ValueError("'analise.valores' não pode ser lista vazia.")

    if analise["modo"] not in ("angulos", "k_global"):
        raise ValueError(
            f"'analise.modo' inválido: '{analise['modo']}'. Use 'angulos' ou 'k_global'."
        )

    return dados


def _resolver_valores(analise, modo):
    """
    Converte ``analise.valores`` (lista ou range) em array NumPy concreto.


    Args:
        analise (dict): Bloco 'analise' do JSON.
        modo (str): ``'angulos'`` ou ``'k_global'``.

    Returns:
        tuple[list, list]: ``(angulos_teste, k_globais_teste)``.
            Um dos dois será sempre vazio dependendo do modo.
    """
    valores_raw = analise["valores"]

    if isinstance(valores_raw, dict):
        inicio = float(valores_raw["inicio"])
        fim    = float(valores_raw["fim"])
        passo  = float(valores_raw["passo"])
        valores_expandidos = np.arange(inicio, fim + passo / 2, passo)
    else:
        valores_expandidos = valores_raw

    if modo == "k_global":
        out = []
        for v in valores_expandidos:
            if isinstance(v, list):
                if len(v) != 2:
                    raise ValueError(f"k complexo inválido: {v}. Use [real, imag].")
                out.append(complex(v[0], v[1]))
            else:
                out.append(complex(v))
        return [], out

    return [float(v) for v in valores_expandidos], []


def configurar_analise(dados):
    """
    Extrai os parâmetros de análise do JSON e calcula o número de graus de liberdade.

    Args:
        dados (dict): Saída de :func:`ler_arquivo_entrada`.

    Returns:
        dict: Configuração com chaves ``omega``, ``modo``, ``angulos_teste``,
              ``k_globais_teste``, ``arquivo_saida``, ``num_gl``.
              O campo ``arquivo_saida`` é lido de ``analise.arquivo_saida`` no
              JSON; se ausente, usa ``'SAIDA_Rigidez_Global.txt'``.
    """
    analise = dados["analise"]
    num_gl  = 2 * (len(dados['camadas']) + 1)
    omega   = float(analise["omega"])
    modo    = analise["modo"]

    angulos_teste, k_globais_teste = _resolver_valores(analise, modo)

    return {
        'omega'          : omega,
        'modo'           : modo,
        'angulos_teste'  : angulos_teste,
        'k_globais_teste': k_globais_teste,
        'arquivo_saida'  : analise.get('arquivo_saida', 'SAIDA_Rigidez_Global.txt'),
        'num_gl'         : num_gl,
    }

# ==============================================================================
# 2. PRÉ-CÁLCULO DE PROPRIEDADES DO MATERIAL
# ==============================================================================

def _calcular_props_material(props):
    """
    Núcleo de cálculo das propriedades dinâmicas complexas de uma camada.

    Função base reutilizada por :func:`precalcular_propriedades_materiais` e
    :func:`calcular_propriedades_dinamicas_locais`, eliminando duplicação de
    lógica e garantindo consistência entre os dois caminhos de execução.

    Args:
        props (dict): Propriedades do material com chaves
                      ``G``, ``nu``, ``zeta_p``, ``zeta_s``, ``rho``, ``d``.

    Returns:
        tuple: ``(G_complex, cp_star, cs_star, d)`` onde os três primeiros
               são complexos e ``d`` é float.
    """
    G_real = props.get('G',      20000.0)
    nu     = props.get('nu',     0.33)
    zeta_p = props.get('zeta_p', 0.02)
    zeta_s = props.get('zeta_s', 0.02)
    rho    = props.get('rho',    2.0)
    d      = props.get('d',      0.0)

    lam_real  = (2 * G_real * nu) / (1 - 2 * nu)
    M_real    = lam_real + 2 * G_real
    M_complex = M_real * complex(1, 2 * zeta_p)
    G_complex = G_real * complex(1, 2 * zeta_s)
    cp_star   = cmath.sqrt(M_complex / rho)
    cs_star   = cmath.sqrt(G_complex / rho)

    return G_complex, cp_star, cs_star, d


def precalcular_propriedades_materiais(dados_json):
    """
    Calcula as propriedades dinâmicas complexas de todas as camadas e do
    semi-espaço em uma única passagem.

    Esta função deve ser chamada uma vez antes do loop paramétrico. Os arrays
    retornados são indexados por camada e reutilizados em todas as iterações,
    eliminando o recálculo redundante de G*, cp*, cs* a cada ângulo ou k.

    Args:
        dados_json (dict): Dados do modelo com chaves ``'camadas'`` e
                           opcionalmente ``'semi_espaco'``.

    Returns:
        dict:
            - ``G_complex`` (ndarray complexo, shape (n,)): Módulo de
              cisalhamento complexo G* [kPa].
            - ``cp_star`` (ndarray complexo, shape (n,)): Velocidade de
              onda P complexa [m/s].
            - ``cs_star`` (ndarray complexo, shape (n,)): Velocidade de
              onda S complexa [m/s].
            - ``d`` (ndarray float, shape (n,)): Espessuras [m].
              Para o semi-espaço, ``d=0``.
            - ``n_camadas`` (int): Número de camadas (sem o semi-espaço).
    """
    camadas     = dados_json['camadas']
    semi_espaco = dados_json.get('semi_espaco')
    todas       = list(camadas) + ([semi_espaco] if semi_espaco else [])
    n           = len(todas)

    G_arr  = np.zeros(n, dtype=complex)
    cp_arr = np.zeros(n, dtype=complex)
    cs_arr = np.zeros(n, dtype=complex)
    d_arr  = np.zeros(n, dtype=float)

    for i, props in enumerate(todas):
        G_complex, cp_star, cs_star, d = _calcular_props_material(props)
        G_arr[i]  = G_complex
        cp_arr[i] = cp_star
        cs_arr[i] = cs_star
        d_arr[i]  = d

    return {
        'G_complex': G_arr,
        'cp_star'  : cp_arr,
        'cs_star'  : cs_arr,
        'd'        : d_arr,
        'n_camadas': len(camadas),
    }

# ==============================================================================
# 3. CÁLCULO VETORIZADO DE s E t
# ==============================================================================

def calcular_s_t_vetorizado(mat_props, omega, k_arr):
    """
    Calcula os parâmetros de propagação vertical s e t para todas as camadas
    e todos os valores de k_global simultaneamente via broadcasting NumPy.

    Os parâmetros são definidos como (Wolf 1985, Eq. 5.121)::

        s = -i * sqrt(1 - (c / cp*)^2)
        t = -i * sqrt(1 - (c / cs*)^2)

    onde c = omega / k é a velocidade de fase horizontal.

    O broadcasting opera sobre shapes ``(N, 1)`` x ``(1, n_total)``
    produzindo resultados ``(N, n_total)`` sem alocação de loops Python.

    Args:
        mat_props (dict): Saída de :func:`precalcular_propriedades_materiais`.
        omega (float): Frequência angular [rad/s].
        k_arr (ndarray complexo, shape (N,)): Valores de k_global [rad/m].

    Returns:
        tuple[ndarray, ndarray]: ``(s_all, t_all)``, ambos com shape ``(N, n_total)``.
    """
    cp = mat_props['cp_star']  # (n_total,)
    cs = mat_props['cs_star']  # (n_total,)

    k_col  = k_arr[:, None]   # (N, 1)
    cp_row = cp[None, :]      # (1, n_total)
    cs_row = cs[None, :]      # (1, n_total)

    with np.errstate(divide='ignore', invalid='ignore'):
        vf    = omega / k_col
        s_all = -1j * np.sqrt(1 - (vf / cp_row) ** 2)
        t_all = -1j * np.sqrt(1 - (vf / cs_row) ** 2)

    return s_all, t_all

# ==============================================================================
# 4. MONTAGEM VETORIZADA DA MATRIZ GLOBAL
# ==============================================================================

def montar_K_global_vetorizado(mat_props, s_all, t_all, k_arr, num_gl):
    """
    Monta as matrizes de rigidez dinâmica global para todos os N valores de
    k_global em uma única operação batch.

    Cobre o **caso geral** (ω > 0, k > 0). Casos especiais devem ser
    tratados via :func:`calcular_matriz_camada_escalar` e
    :func:`calcular_matriz_semiespaco_escalar`.

    Para cada camada i, a sub-matriz 4x4 é calculada conforme Wolf (1985),
    Tabela 5-3, e acumulada na posição ``[2i:2i+4, 2i:2i+4]`` da matriz global.
    O semi-espaço contribui com a sub-matriz 2x2 (Eq. 5.135) na posição final.

    Args:
        mat_props (dict): Saída de :func:`precalcular_propriedades_materiais`.
        s_all (ndarray complexo, shape (N, n_total)): Parâmetro s por camada.
        t_all (ndarray complexo, shape (N, n_total)): Parâmetro t por camada.
        k_arr (ndarray complexo, shape (N,)): Valores de k_global [rad/m].
        num_gl (int): Dimensão da matriz global (= 2 * (n_camadas + 1)).

    Returns:
        ndarray complexo, shape (N, num_gl, num_gl): Matrizes globais K [kPa].
    """
    N     = len(k_arr)
    n_cam = mat_props['n_camadas']
    G     = mat_props['G_complex']
    d     = mat_props['d']

    K_all = np.zeros((N, num_gl, num_gl), dtype=complex)

    for i in range(n_cam):
        s  = s_all[:, i]
        t  = t_all[:, i]
        Gi = G[i]
        di = d[i]
        ki = k_arr

        s2          = s ** 2
        t2          = t ** 2
        one_plus_t2 = 1 + t2
        ksd         = ki * s * di
        ktd         = ki * t * di

        sin_ksd = np.sin(ksd)
        cos_ksd = np.cos(ksd)
        sin_ktd = np.sin(ktd)
        cos_ktd = np.cos(ktd)

        st = s * t
        with np.errstate(divide='ignore', invalid='ignore'):
            term_st = st + 1.0 / st

        D     = 2 * (1 - cos_ksd * cos_ktd) + term_st * sin_ksd * sin_ktd
        fator = (one_plus_t2 * ki * Gi) / D

        t11 = (1 / t) * cos_ksd * sin_ktd + s * sin_ksd * cos_ktd
        t22 = (1 / s) * sin_ksd * cos_ktd + t * cos_ksd * sin_ktd
        t12 = (
            ((3 - t2) / one_plus_t2) * (1 - cos_ksd * cos_ktd)
            + ((1 + 2 * s2 * t2 - t2) / (st * one_plus_t2)) * sin_ksd * sin_ktd
        )
        t13 = -s * sin_ksd - (1 / t) * sin_ktd
        t14 = cos_ksd - cos_ktd
        t23 = -t14
        t24 = -(1 / s) * sin_ksd - t * sin_ktd

        sub = np.empty((N, 4, 4), dtype=complex)

        sub[:, 0, 0] =  t11;  sub[:, 0, 1] =  t12;  sub[:, 0, 2] =  t13;  sub[:, 0, 3] =  t14
        sub[:, 1, 0] =  t12;  sub[:, 1, 1] =  t22;  sub[:, 1, 2] =  t23;  sub[:, 1, 3] =  t24
        sub[:, 2, 0] =  t13;  sub[:, 2, 1] =  t23;  sub[:, 2, 2] =  t11;  sub[:, 2, 3] = -t12
        sub[:, 3, 0] =  t14;  sub[:, 3, 1] =  t24;  sub[:, 3, 2] = -t12;  sub[:, 3, 3] =  t22

        sub *= fator[:, None, None]

        idx = 2 * i
        K_all[:, idx:idx+4, idx:idx+4] += sub

    if n_cam < len(G):
        i_s    = n_cam
        s_semi = s_all[:, i_s]
        t_semi = t_all[:, i_s]
        G_semi = G[i_s]

        t2_s        = t_semi ** 2
        one_plus_t2 = 1 + t2_s
        denom       = 1 + s_semi * t_semi

        with np.errstate(divide='ignore', invalid='ignore'):
            k11 = (s_semi * one_plus_t2) * 1j / denom
            k22 = (t_semi * one_plus_t2) * 1j / denom
            k12 = 2 - one_plus_t2 / denom

        fator = k_arr * G_semi
        idx   = 2 * n_cam
        K_all[:, idx,     idx    ] += k11 * fator
        K_all[:, idx,     idx + 1] += k12 * fator
        K_all[:, idx + 1, idx    ] += k12 * fator
        K_all[:, idx + 1, idx + 1] += k22 * fator

    return K_all

# ==============================================================================
# 5. FORMULAÇÕES ESCALARES — CASOS ESPECIAIS (ω=0 ou k=0)
# ==============================================================================

def calcular_propriedades_dinamicas_locais(props, omega):
    """
    Calcula as propriedades dinâmicas complexas de uma camada (G*, cp*, cs*).

    Usado no fallback escalar para os casos especiais ω = 0 ou k = 0.
    Não calcula s e t — esses são derivados diretamente em cada formulação
    especializada (vertical, estática, etc.) para evitar cálculo duplicado.

    Args:
        props (dict): Propriedades do material com chaves
                      ``G``, ``nu``, ``zeta_p``, ``zeta_s``, ``rho``, ``d``.
        omega (float): Frequência angular [rad/s]. Mantido para consistência
                       de interface, não é usado no cálculo das propriedades
                       elásticas (que são independentes de frequência).

    Returns:
        dict: ``G_complex``, ``cp_star``, ``cs_star``, ``d``.
    """
    G_complex, cp_star, cs_star, d = _calcular_props_material(props)
    return {
        'G_complex': G_complex,
        'cp_star'  : cp_star,
        'cs_star'  : cs_star,
        'd'        : d,
    }


def calcular_matriz_camada_vertical(params, omega):
    """
    Matriz de rigidez 4x4 para ondas verticalmente incidentes (ω > 0, k = 0).

    Wolf (1985), Seção 5.4.3, caso especial 1, Eq. 5.136a.
    Neste limite, P e S ficam desacoplados e os argumentos são
    ωd/cs* (modo S) e ωd/cp* (modo P).

    Args:
        params (dict): Saída de :func:`calcular_propriedades_dinamicas_locais`.
        omega (float): Frequência angular [rad/s].

    Returns:
        ndarray complexo (4, 4): Matriz de rigidez [kPa].

    Raises:
        ValueError: Se sin(ωd/cs*) ≈ 0 ou sin(ωd/cp*) ≈ 0 (ressonância da camada).
    """
    G  = params['G_complex']
    cp = params['cp_star']
    cs = params['cs_star']
    d  = params['d']

    arg_s = omega * d / cs
    arg_p = omega * d / cp
    sin_s = cmath.sin(arg_s)
    sin_p = cmath.sin(arg_p)

    if abs(sin_s) < TOL_DENOM:
        raise ValueError(
            f"Ressonância modo S: sin(ωd/cs*)≈0 [arg={arg_s:.6f}]."
        )
    if abs(sin_p) < TOL_DENOM:
        raise ValueError(
            f"Ressonância modo P: sin(ωd/cp*)≈0 [arg={arg_p:.6f}]."
        )

    cot_s = cmath.cos(arg_s) / sin_s
    csc_s = 1.0 / sin_s
    cot_p = cmath.cos(arg_p) / sin_p
    csc_p = 1.0 / sin_p
    ratio = cp / cs
    fator = G * omega / cs

    K = np.zeros((4, 4), dtype=complex)
    K[0, 0] =  cot_s;          K[0, 2] = -csc_s
    K[1, 1] =  ratio * cot_p;  K[1, 3] = -ratio * csc_p
    K[2, 0] = -csc_s;          K[2, 2] =  cot_s
    K[3, 1] = -ratio * csc_p;  K[3, 3] =  ratio * cot_p

    return K * fator


def calcular_matriz_camada_estatica(params, k):
    """
    Matriz de rigidez 4x4 para o caso estático (ω = 0, k ≠ 0).

    Wolf (1985), Seção 5.4.3, caso especial 2, Eq. 5.137a / Tabela 5-4.

    Args:
        params (dict): Saída de :func:`calcular_propriedades_dinamicas_locais`.
        k (complex): Número de onda horizontal [rad/m].

    Returns:
        ndarray complexo (4, 4): Matriz de rigidez [kPa].

    Raises:
        ValueError: Se o denominador D ≈ 0 (configuração degenerada de k e d).
    """
    G  = params['G_complex']
    cp = params['cp_star']
    cs = params['cs_star']
    d  = params['d']

    kd      = k * d
    r       = (cs / cp) ** 2
    sinh_kd = cmath.sinh(kd)
    cosh_kd = cmath.cosh(kd)
    D       = (1 + r) ** 2 * sinh_kd ** 2 - kd ** 2 * (1 - r) ** 2

    if abs(D) < TOL_DENOM:
        raise ValueError(
            f"Denominador D ≈ 0 [D={D:.2e}, kd={kd:.6f}]. "
            "Ajuste k ou a espessura da camada."
        )

    fator = 2 * k * G / D
    a     = 1 + r
    b     = 1 - r

    S11 =  a * sinh_kd * cosh_kd - b * kd
    S12 = -a * sinh_kd ** 2 + D
    S13 =  b * kd * cosh_kd - a * sinh_kd
    S14 =  kd * b * sinh_kd
    S22 =  a * sinh_kd * cosh_kd + kd * b
    S23 = -kd * b * sinh_kd
    S24 = -b * kd * cosh_kd - a * sinh_kd
    S33 =  a * sinh_kd * cosh_kd - b * kd
    S34 =  a * sinh_kd ** 2 - D
    S44 =  a * sinh_kd * cosh_kd + kd * b

    K = np.array([
        [S11, S12, S13, S14],
        [S12, S22, S23, S24],
        [S13, S23, S33, S34],
        [S14, S24, S34, S44],
    ], dtype=complex)

    return K * fator


def calcular_matriz_camada_estatica_k0(params):
    """
    Matriz de rigidez 4x4 para o caso completamente estático (ω = 0, k = 0).

    Wolf (1985), Seção 5.4.3, caso especial 3, Eq. 5.138a.

    Args:
        params (dict): Saída de :func:`calcular_propriedades_dinamicas_locais`.

    Returns:
        ndarray complexo (4, 4): Matriz de rigidez [kPa].
    """
    G  = params['G_complex']
    cp = params['cp_star']
    cs = params['cs_star']
    d  = params['d']

    r = (cp / cs) ** 2

    K = np.array([
        [ 1,  0, -1,  0],
        [ 0,  r,  0, -r],
        [-1,  0,  1,  0],
        [ 0, -r,  0,  r],
    ], dtype=complex)

    return K * (G / d)


def calcular_matriz_camada_escalar(params, k, omega=None):
    """
    Roteia para a formulação escalar correta da matriz de camada 4x4.

    Utilizado como fallback para os casos especiais não cobertos pelo
    caminho vetorizado (ω = 0 ou k = 0). O caso geral (ω > 0, k > 0)
    é tratado exclusivamente pelo caminho vetorizado.

    ============  ===========  =====================
    ω             k            Formulação
    ============  ===========  =====================
    ω = 0         k = 0        Eq. 5.138a
    ω = 0         k ≠ 0        Eq. 5.137a / Tab. 5-4
    ω > 0         k = 0        Eq. 5.136a
    ============  ===========  =====================

    Args:
        params (dict): Saída de :func:`calcular_propriedades_dinamicas_locais`.
        k (complex): Número de onda horizontal [rad/m].
        omega (float, opcional): Frequência angular [rad/s].

    Returns:
        ndarray complexo (4, 4): Matriz de rigidez [kPa].

    Raises:
        ValueError: Se denominadores forem nulos ou ressonância detectada.
    """
    omega_val = omega if omega is not None else 0.0
    k_zero    = abs(k)         < TOL_K_ZERO
    om_zero   = abs(omega_val) < TOL_OMEGA_ZERO

    if om_zero and k_zero:
        return calcular_matriz_camada_estatica_k0(params)
    if om_zero:
        return calcular_matriz_camada_estatica(params, k)
    # k_zero e omega > 0
    return calcular_matriz_camada_vertical(params, omega_val)


def calcular_matriz_semiespaco_escalar(params, k, omega=None):
    """
    Calcula a matriz de rigidez 2x2 do semi-espaço para um único par (omega, k).

    Roteia conforme o regime:

    ============  ===========  =====================
    ω             k            Formulação
    ============  ===========  =====================
    ω = 0         k = 0        Matriz nula (Eq. 5.138b)
    ω = 0         k ≠ 0        Eq. 5.137b
    ω > 0         k = 0        Eq. 5.136b
    ω > 0         k > 0        Eq. 5.135 (caso geral)
    ============  ===========  =====================

    Args:
        params (dict): Saída de :func:`calcular_propriedades_dinamicas_locais`.
        k (complex): Número de onda horizontal [rad/m].
        omega (float, opcional): Frequência angular [rad/s].

    Returns:
        ndarray complexo (2, 2): Matriz de rigidez [kPa].
    """
    omega_val = omega if omega is not None else 0.0
    k_zero    = abs(k)         < TOL_K_ZERO
    om_zero   = abs(omega_val) < TOL_OMEGA_ZERO

    G  = params['G_complex']
    cp = params['cp_star']
    cs = params['cs_star']

    if om_zero and k_zero:
        return np.zeros((2, 2), dtype=complex)

    if om_zero:
        r = (cs / cp) ** 2
        return 2 * k * G * np.array(
            [[1 / (1 + r),       1 / (1 + 1 / r)],
             [1 / (1 + 1 / r),   1 / (1 + r)    ]],
            dtype=complex
        )

    if k_zero:
        return (1j * G * omega_val / cs) * np.array(
            [[1, 0], [0, cp / cs]],
            dtype=complex
        )

    # Caso geral (ω > 0, k > 0): calcula s e t localmente
    vf = omega_val / k
    s  = -1j * cmath.sqrt(1 - (vf / cp) ** 2)
    t  = -1j * cmath.sqrt(1 - (vf / cs) ** 2)

    t2          = t ** 2
    one_plus_t2 = 1 + t2
    denom       = 1 + s * t

    k11 = (s * one_plus_t2) * 1j / denom
    k22 = (t * one_plus_t2) * 1j / denom
    k12 = 2 - one_plus_t2 / denom

    return k * G * np.array([[k11, k12], [k12, k22]], dtype=complex)

# ==============================================================================
# 6. LOOP PRINCIPAL VETORIZADO
# ==============================================================================

def executar_vetorizado(dados, omega, valores_brutos, modo, num_gl, **kwargs):
    """
    Executa a análise paramétrica completa processando todos os valores em batch.

    Estratégia de execução:

    1. Pré-calcula G*, cp*, cs* de todas as camadas uma única vez.
    2. Converte ângulos em k_global (modo ``'angulos'``) ou usa k direto.
    3. Separa os índices em dois grupos:

       - **Caso geral** (ω > 0, k > 0): processado via
         :func:`calcular_s_t_vetorizado` e :func:`montar_K_global_vetorizado`.
       - **Casos especiais** (k ≈ 0 ou ω ≈ 0): fallback escalar com
         :func:`calcular_matriz_camada_escalar`.

    4. Coleta os resultados na ordem original e formata para escrita.

    Para ângulos próximos a 90°, o caso é sinalizado com aviso ao usuário
    em vez de substituição silenciosa do cosseno.

    Erros numéricos em casos individuais (ressonância) são logados e pulados;
    os demais casos continuam normalmente.

    Args:
        dados (dict): Saída de :func:`ler_arquivo_entrada`.
        omega (float): Frequência angular [rad/s].
        valores_brutos (list): Ângulos [graus] ou k_global [rad/m].
        modo (str): ``'angulos'`` ou ``'k_global'``.
        num_gl (int): Dimensão da matriz global.
        **kwargs: Argumentos extras ignorados (compatibilidade com cfg).

    Returns:
        list[str]: Lista de strings formatadas, uma por caso calculado.
    """
    camadas     = dados['camadas']
    semi_espaco = dados.get('semi_espaco')

    mat_props = precalcular_propriedades_materiais(dados)
    cp_ref    = mat_props['cp_star'][0]
    cs_ref    = mat_props['cs_star'][0]

    if modo == 'angulos':
        ang_arr = np.array(valores_brutos, dtype=float)
        lx_arr  = np.cos(np.radians(ang_arr))

        # Detecta e avisa sobre ângulos próximos a 90° em vez de substituir silenciosamente
        mask_90 = np.abs(lx_arr) < TOL_ANGULO_90
        if np.any(mask_90):
            angulos_problematicos = ang_arr[mask_90]
            logger.warning(
                f"Ângulo(s) muito próximo(s) de 90° detectado(s): "
                f"{angulos_problematicos.tolist()}. "
                f"cos(θ) < {TOL_ANGULO_90:.0e} — esses casos serão tratados "
                f"como incidência vertical (k ≈ 0)."
            )
            lx_arr = np.where(mask_90, 0.0, lx_arr)

        if abs(omega) < TOL_OMEGA_ZERO:
            k_arr = np.zeros(len(ang_arr), dtype=complex)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                k_arr = np.where(
                    mask_90,
                    0.0,
                    omega / (cp_ref / lx_arr)
                ).astype(complex)
    else:
        k_arr = np.array(valores_brutos, dtype=complex)

    N = len(k_arr)

    mask_especial = (np.abs(k_arr) < TOL_K_ZERO) | (abs(omega) < TOL_OMEGA_ZERO)
    mask_geral    = ~mask_especial
    idx_geral     = np.where(mask_geral)[0]
    idx_especial  = np.where(mask_especial)[0]

    K_results  = [None] * N
    debug_list = [None] * N

    if len(idx_geral) > 0:
        k_g          = k_arr[idx_geral]
        s_all, t_all = calcular_s_t_vetorizado(mat_props, omega, k_g)
        K_batch      = montar_K_global_vetorizado(mat_props, s_all, t_all, k_g, num_gl)

        for bi, oi in enumerate(idx_geral):
            kv  = k_g[bi]
            c_f = omega / kv
            lx  = cp_ref / c_f
            dbg = {
                'c_fase': c_f,
                'l'     : lx,
                'm'     : cs_ref / c_f,
                's'     : s_all[bi, 0],
                't'     : t_all[bi, 0],
            }
            if modo == 'k_global':
                lr = lx.real
                dbg['angulo_equiv'] = (
                    float(np.degrees(np.arccos(np.clip(lr, -1.0, 1.0))))
                    if abs(lr) <= 1.0 else float('nan')
                )
            K_results[oi]  = K_batch[bi]
            debug_list[oi] = dbg

    for oi in idx_especial:
        kv    = k_arr[oi]
        K_mat = np.zeros((num_gl, num_gl), dtype=complex)
        ok    = True

        # Propriedades locais sem s/t (calculados internamente em cada formulação)
        pcache = [calcular_propriedades_dinamicas_locais(p, omega) for p in camadas]

        for i, pl in enumerate(pcache):
            try:
                S = calcular_matriz_camada_escalar(pl, kv, omega=omega)
                K_mat[2 * i:2 * i + 4, 2 * i:2 * i + 4] += S
            except ValueError as e:
                logger.warning(f"Caso ignorado (idx={oi}, k={kv:.4e}): {e}")
                ok = False
                break

        if ok and semi_espaco:
            ps = calcular_propriedades_dinamicas_locais(semi_espaco, omega)
            try:
                Sb = calcular_matriz_semiespaco_escalar(ps, kv, omega=omega)
                ix = 2 * len(camadas)
                K_mat[ix:ix + 2, ix:ix + 2] += Sb
            except ValueError as e:
                logger.warning(f"Semi-espaço ignorado (idx={oi}): {e}")

        # Debug: s e t da camada 1 para log (calculados pontualmente aqui)
        cp0 = pcache[0]['cp_star']
        cs0 = pcache[0]['cs_star']
        if abs(kv) < TOL_K_ZERO:
            s_dbg = t_dbg = None
            c_f = complex(float('inf'))
            lx  = complex(1)
            mx  = complex(0)
        else:
            c_f   = omega / kv
            lx    = cp_ref / c_f
            mx    = cs_ref / c_f
            s_dbg = -1j * cmath.sqrt(1 - (c_f / cp0) ** 2)
            t_dbg = -1j * cmath.sqrt(1 - (c_f / cs0) ** 2)

        dbg = {
            'c_fase': c_f,
            'l'     : lx,
            'm'     : mx,
            's'     : s_dbg,
            't'     : t_dbg,
        }
        if modo == 'k_global':
            dbg['angulo_equiv'] = 0.0 if abs(kv) < TOL_K_ZERO else float('nan')

        K_results[oi]  = K_mat if ok else None
        debug_list[oi] = dbg

    resultados = []
    for i in range(N):
        if K_results[i] is None:
            continue
        label = valores_brutos[i] if modo == 'angulos' else k_arr[i]
        resultados.append(
            formatar_resultado(label, K_results[i], k_arr[i], debug_list[i], num_gl, modo)
        )

    return resultados

# ==============================================================================
# 7. FORMATAÇÃO E I/O
# ==============================================================================

def formatar_resultado(label_valor, K_total, k_calc, debug, num_gl, modo):
    """
    Formata um resultado individual em string pronta para escrita em arquivo.

    Args:
        label_valor: Ângulo [graus] (float) ou k_global [rad/m] (complex).
        K_total (ndarray complexo, shape (num_gl, num_gl)): Matriz global [kPa].
        k_calc (complex): Número de onda horizontal calculado [rad/m].
        debug (dict): Parâmetros de controle: ``c_fase``, ``l``, ``m``, ``s``, ``t``
                      e opcionalmente ``angulo_equiv``.
        num_gl (int): Dimensão da matriz.
        modo (str): ``'angulos'`` ou ``'k_global'``.

    Returns:
        str: Bloco formatado com cabeçalho, parâmetros e matriz.
    """
    linhas = []

    if modo == 'angulos':
        linhas.append(f"ANGULO: {label_valor:6.3f} graus\n")
    else:
        linhas.append(
            f"K_GLOBAL: {label_valor.real:+.6e} {label_valor.imag:+.6e}j  [rad/m]"
            f"   |k| = {abs(label_valor):+.6e}\n"
        )

    linhas.append("-" * 80 + "\n")
    linhas.append("PARAMETROS DE CONTROLE (Camada 1):\n")

    c_f = debug['c_fase']
    if abs(c_f) == float('inf'):
        linhas.append("  c (velocidade fase) = inf  [m/s]  (k=0, incidência vertical)\n")
    else:
        linhas.append(f"  c (velocidade fase) = {c_f.real:+.4e} {c_f.imag:+.4e}j  [m/s]\n")

    linhas.append(f"  l (cos psi_P)       = {debug['l'].real:+.8f} {debug['l'].imag:+.8f}j\n")
    linhas.append(f"  m (cos psi_S)       = {debug['m'].real:+.8f} {debug['m'].imag:+.8f}j\n")

    s_v = debug.get('s')
    t_v = debug.get('t')
    if s_v is None:
        linhas.append("  s (param onda P)    = N/A (caso especial)\n")
        linhas.append("  t (param onda S)    = N/A (caso especial)\n")
    else:
        linhas.append(f"  s (param onda P)    = {s_v.real:+.8f} {s_v.imag:+.8f}j\n")
        linhas.append(f"  t (param onda S)    = {t_v.real:+.8f} {t_v.imag:+.8f}j\n")

    linhas.append(f"  k (num onda horiz)  = {k_calc.real:+.6e} {k_calc.imag:+.6e}j  [rad/m]\n")
    linhas.append(f"  |k|                 = {abs(k_calc):+.6e}  [rad/m]\n")

    if modo == 'k_global':
        ae = debug.get('angulo_equiv', float('nan'))
        linhas.append(
            f"  angulo equiv (psi_P) = "
            f"{'n/a (fora da faixa física)' if np.isnan(ae) else f'{ae:.4f} graus'}\n"
        )

    linhas.append("\nMATRIZ K [kPa]:\n")
    for i in range(num_gl):
        for j in range(num_gl):
            v = K_total[i, j]
            linhas.append(f"  {v.real:+.4e} {v.imag:+.4e}j")
        linhas.append("\n")

    linhas.append("\nMATRIZ F = K^{-1} [m/kPa]:\n")
    try:
        F_total = np.linalg.solve(K_total, np.eye(num_gl, dtype=complex))
        for i in range(num_gl):
            for j in range(num_gl):
                v = F_total[i, j]
                linhas.append(f"  {v.real:+.4e} {v.imag:+.4e}j")
            linhas.append("\n")
        cond = np.linalg.cond(K_total)
        linhas.append(f"\n  [cond(K) = {cond:.4e}]\n")
    except np.linalg.LinAlgError:
        linhas.append("  [ERRO: matriz K singular — F nao calculada]\n")

    linhas.append("\n" + "=" * 80 + "\n\n")
    return "".join(linhas)


def _resumir_valores_log(valores, n_limiar=5):
    """
    Gera uma representação compacta de uma lista de valores para uso em logs.

    Até ``n_limiar`` itens exibe a lista completa. Acima, exibe
    quantidade, mínimo, máximo e passo (quando uniforme nos primeiros
    elementos — heurística para listas geradas por np.arange).

    Args:
        valores (list): Lista de floats ou complexos.
        n_limiar (int): Limite abaixo do qual a lista é exibida integralmente.

    Returns:
        str: Representação compacta.
    """
    n = len(valores)
    if n <= n_limiar:
        return str(valores)

    eh_complex = isinstance(valores[0], complex)
    if eh_complex:
        vmin = min(abs(v) for v in valores)
        vmax = max(abs(v) for v in valores)
    else:
        vmin = min(float(v) for v in valores)
        vmax = max(float(v) for v in valores)

    passo_str = ""
    if n >= 2:
        diffs = [valores[i + 1] - valores[i] for i in range(min(5, n - 1))]
        ref   = diffs[0]
        tol   = 1e-9 * (abs(ref) + 1e-30)
        if all(abs(d - ref) < tol for d in diffs):
            passo_str = f", passo≈{ref:.6g}"

    return f"{n} valores | min={vmin:.6g}  max={vmax:.6g}{passo_str}"


def salvar_resultados(resultados, arquivo_saida, omega, num_camadas,
                      tem_semi_espaco, num_gl, modo, valores_entrada):
    """
    Grava todos os resultados formatados em arquivo de texto.

    Args:
        resultados (list[str]): Saída de :func:`executar_vetorizado`.
        arquivo_saida (str): Caminho do arquivo de saída.
        omega (float): Frequência angular [rad/s].
        num_camadas (int): Número de camadas do perfil.
        tem_semi_espaco (bool): Indica presença de semi-espaço.
        num_gl (int): Dimensão da matriz global.
        modo (str): ``'angulos'`` ou ``'k_global'``.
        valores_entrada (list): Lista de ângulos ou k_global usados.
    """
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  MATRIZ DE RIGIDEZ DINAMICA GLOBAL - ONDAS P-SV\n")
        f.write("  Formulacao: Wolf (1985) - Dynamic Soil-Structure Interaction\n")
        f.write("=" * 80 + "\n\n")
        f.write("CONFIGURACAO:\n")
        f.write(f"  Frequencia .......: omega = {omega} rad/s  (f = {omega / (2 * np.pi):.4f} Hz)\n")
        f.write(f"  Camadas ..........: {num_camadas}\n")
        f.write(f"  Semi-espaco ......: {'Sim' if tem_semi_espaco else 'Nao'}\n")
        f.write(f"  Dimensao matriz ..: {num_gl} x {num_gl}\n")
        if modo == 'angulos':
            f.write(f"  Modo .............: Angulos de incidencia\n")
            f.write(f"  Angulos [graus] ..: {_resumir_valores_log(valores_entrada)}\n")
        else:
            f.write(f"  Modo .............: k_global direto\n")
            f.write(f"  k_global [rad/m] .: {_resumir_valores_log(valores_entrada)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.writelines(resultados)

    logger.info(f"Resultados salvos em: {arquivo_saida}")

# ==============================================================================
# 8. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    """Ponto de entrada: lê configuração, executa análise e salva resultados."""
    arquivo_input = 'camada.json'

    try:
        dados = ler_arquivo_entrada(arquivo_input)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    cfg = configurar_analise(dados)

    modo            = cfg['modo']
    valores_entrada = cfg['angulos_teste'] if modo == 'angulos' else cfg['k_globais_teste']

    logger.info("=" * 60)
    logger.info("  ANALISE DE RIGIDEZ DINAMICA - METODO DE WOLF")
    logger.info("=" * 60)
    logger.info(f"  Camadas ........: {len(dados['camadas'])}")
    logger.info(f"  Semi-espaco ....: {'Sim' if dados.get('semi_espaco') else 'Nao'}")
    logger.info(f"  Graus liberdade : {cfg['num_gl']}")
    logger.info(f"  Frequencia .....: omega = {cfg['omega']} rad/s  "
                f"(f = {cfg['omega'] / (2 * np.pi):.4f} Hz)")
    logger.info(f"  Modo ...........: {modo}")
    logger.info(f"  Valores entrada : {_resumir_valores_log(valores_entrada)}")
    logger.info(f"  Arquivo saida ..: {cfg['arquivo_saida']}")
    logger.info("=" * 60)

    resultados = executar_vetorizado(
        dados,
        omega          = cfg['omega'],
        valores_brutos = valores_entrada,
        modo           = modo,
        num_gl         = cfg['num_gl'],
    )

    salvar_resultados(
        resultados      = resultados,
        arquivo_saida   = cfg['arquivo_saida'],
        omega           = cfg['omega'],
        num_camadas     = len(dados['camadas']),
        tem_semi_espaco = bool(dados.get('semi_espaco')),
        num_gl          = cfg['num_gl'],
        modo            = modo,
        valores_entrada = valores_entrada,
    )

    logger.info("Análise concluída.")


if __name__ == "__main__":
    main()