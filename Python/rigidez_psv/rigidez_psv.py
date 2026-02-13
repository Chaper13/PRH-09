import numpy as np
import cmath
import json
import logging
import sys
from tqdm import tqdm

# ==============================================================================
# CONFIGURAÇÃO DE LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTES NUMÉRICAS
# ==============================================================================

TOL_DEGENERADO = 1e-15  # tolerância para s*t próximo de zero
TOL_ANGULO_90  = 1e-10  # proteção para ângulo próximo de 90°
TOL_K_ZERO     = 1e-12  # tolerância para k_global ≈ 0
TOL_OMEGA_ZERO = 1e-12  # tolerância para omega ≈ 0
TOL_DENOM      = 1e-30  # tolerância para denominadores (D, sin) próximos de zero

# ==============================================================================
# 1. FUNÇÕES DE LEITURA E CONFIGURAÇÃO
# ==============================================================================

def ler_arquivo_entrada(nome_arquivo):
    """
    Lê e valida arquivo JSON com configuração de camadas de solo e parâmetros de análise.

    Args:
        nome_arquivo (str): Caminho do arquivo JSON a ser lido.

    Returns:
        dict: Dicionário com dados das camadas, semi-espaço e bloco 'analise'.

    Estrutura esperada do JSON:
        {
            "descricao": "Descrição do perfil",
            "analise": {
                "omega"  : 10.0,
                "modo"   : "angulos",         (ou "k_global")
                "valores": [30, 45, 60]        (lista explícita)
                           ou
                "valores": {                   (range gerado automaticamente)
                    "inicio": 0.001,
                    "fim"   : 8.0,
                    "passo" : 0.001
                }
            },
            "camadas": [
                {"id": 1, "d": 10.0, "G": 20000.0, "nu": 0.33,
                 "zeta_p": 0.02, "zeta_s": 0.02, "rho": 2.0},
                ...
            ],
            "semi_espaco": {
                "G": 100000.0, "nu": 0.25, "zeta_p": 0.01, "zeta_s": 0.01, "rho": 2.4
            }
        }

    Raises:
        SystemExit: Se o arquivo não for encontrado, inválido ou faltar campo obrigatório.
    """
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except FileNotFoundError:
        logger.error(f"Arquivo '{nome_arquivo}' não encontrado.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Arquivo '{nome_arquivo}' não é um JSON válido.\nDetalhe: {e}")
        sys.exit(1)

    # Validar 'camadas'
    if "camadas" not in dados or not isinstance(dados["camadas"], list):
        logger.error("Campo obrigatório ausente ou inválido: 'camadas' deve ser uma lista.")
        sys.exit(1)

    # Validar 'analise' e seus subcampos obrigatórios
    analise = dados.get("analise")
    if analise is None:
        logger.error(
            "Campo obrigatório ausente: 'analise'.\n"
            "Adicione ao JSON:\n"
            '  "analise": { "omega": 10.0, "modo": "angulos", "valores": [30, 45, 60] }'
        )
        sys.exit(1)

    # 'omega' e 'modo' são sempre obrigatórios
    for campo, tipos in {"omega": (int, float), "modo": str}.items():
        if campo not in analise:
            logger.error(
                f"Campo obrigatório ausente em 'analise': '{campo}'.\n"
                f"Campos esperados: omega, modo, valores"
            )
            sys.exit(1)
        if not isinstance(analise[campo], tipos):
            logger.error(
                f"Tipo inválido em 'analise.{campo}': "
                f"esperado {tipos}, recebido {type(analise[campo]).__name__}."
            )
            sys.exit(1)

    # 'valores' pode ser lista explícita ou dict de range {inicio, fim, passo}
    if "valores" not in analise:
        logger.error(
            "Campo obrigatório ausente em 'analise': 'valores'.\n"
            "Use lista explícita: \"valores\": [30, 45, 60]\n"
            "ou range:            \"valores\": {\"inicio\": 0.001, \"fim\": 8.0, \"passo\": 0.001}"
        )
        sys.exit(1)

    valores_raw = analise["valores"]
    if not isinstance(valores_raw, (list, dict)):
        logger.error(
            f"Tipo inválido em 'analise.valores': esperado list ou dict, "
            f"recebido {type(valores_raw).__name__}.\n"
            "Use lista explícita: \"valores\": [30, 45, 60]\n"
            "ou range:            \"valores\": {\"inicio\": 0.001, \"fim\": 8.0, \"passo\": 0.001}"
        )
        sys.exit(1)

    if isinstance(valores_raw, dict):
        for subcampo in ("inicio", "fim", "passo"):
            if subcampo not in valores_raw:
                logger.error(
                    f"Campo obrigatório ausente em 'analise.valores': '{subcampo}'.\n"
                    "Range requer: inicio, fim, passo."
                )
                sys.exit(1)
            if not isinstance(valores_raw[subcampo], (int, float)):
                logger.error(
                    f"Tipo inválido em 'analise.valores.{subcampo}': "
                    f"esperado número, recebido {type(valores_raw[subcampo]).__name__}."
                )
                sys.exit(1)
        if valores_raw["passo"] <= 0:
            logger.error("'analise.valores.passo' deve ser maior que zero.")
            sys.exit(1)
        if valores_raw["inicio"] >= valores_raw["fim"]:
            logger.error("'analise.valores.inicio' deve ser menor que 'fim'.")
            sys.exit(1)
    else:
        if len(valores_raw) == 0:
            logger.error("'analise.valores' não pode ser uma lista vazia.")
            sys.exit(1)

    modo = analise["modo"]
    if modo not in ("angulos", "k_global"):
        logger.error(
            f"Valor inválido em 'analise.modo': '{modo}'.\n"
            "Use 'angulos' ou 'k_global'."
        )
        sys.exit(1)

    return dados


def calcular_propriedades_dinamicas_locais(props, omega, k_global=None):
    """
    Calcula propriedades dinâmicas complexas de um material sob carregamento sísmico.

    Quando k_global=None, calcula apenas as propriedades do material (cp*, cs*, G*),
    sem os parâmetros de propagação vertical (s, t), que dependem de k_global.

    Args:
        props (dict): Propriedades do material.
        omega (float): Frequência angular de excitação [rad/s].
        k_global (complex, opcional): Número de onda horizontal global [rad/m].
            - None        -> s e t retornam None (só propriedades do material)
            - abs < TOL   -> caso especial ω>0, k=0: limite analítico aplicado
            - outro       -> s e t calculados normalmente

    Returns:
        dict: Dicionário com propriedades dinâmicas calculadas.
    """
    d      = props.get('d', 0.0)
    G_real = props.get('G', 20000.0)
    nu     = props.get('nu', 0.33)
    zeta_p = props.get('zeta_p', 0.02)
    zeta_s = props.get('zeta_s', 0.02)
    rho    = props.get('rho', 2.0)

    lam_real = (2 * G_real * nu) / (1 - 2 * nu)  # Constante de Lamé [kPa]
    M_real   = lam_real + 2 * G_real              # Módulo de Onda P [kPa]

    M_complex = M_real * complex(1, 2 * zeta_p)   # M* [kPa]
    G_complex = G_real * complex(1, 2 * zeta_s)   # G* [kPa]

    cp_star = cmath.sqrt(M_complex / rho)          # Velocidade P complexa [m/s]
    cs_star = cmath.sqrt(G_complex / rho)          # Velocidade S complexa [m/s]

    if k_global is None:
        s = None
        t = None
    elif abs(k_global) < TOL_K_ZERO:
        if abs(omega) < TOL_OMEGA_ZERO:
            s = None
            t = None
        else:
            s = complex(0, 1)   # flag: indica caso vertical (ks=ω/cp*, kt=ω/cs*)
            t = complex(0, 1)   # flag: indica caso vertical
    else:
        velocidade_fase = omega / k_global
        s = -1j * cmath.sqrt(1 - (velocidade_fase / cp_star)**2)
        t = -1j * cmath.sqrt(1 - (velocidade_fase / cs_star)**2)

    return {
        'G_complex': G_complex,
        'cp_star'  : cp_star,
        'cs_star'  : cs_star,
        's'        : s,
        't'        : t,
        'd'        : d
    }

# ==============================================================================
# 2. MATRIZES DE RIGIDEZ
# ==============================================================================

def calcular_matriz_camada_vertical(params, omega):
    """
    Matriz de rigidez para ondas verticalmente incidentes (ω > 0, k = 0).

    Caso especial 1 da Seção 5.4.3 de Wolf (1985).
    Implementa a Eq. 5.136a: [S^L_{P-SV}] para k = 0.

    Neste limite, os acoplamentos P-SV desaparecem e as direções horizontal
    e vertical ficam desacopladas. Os argumentos dos senos/cossenos são:
        ωd/cs*  (modo S, horizontal)
        ωd/cp*  (modo P, vertical)

    Args:
        params (dict): Propriedades dinâmicas (G_complex, cp_star, cs_star, d).
        omega (float): Frequência angular [rad/s].

    Returns:
        numpy.ndarray: Matriz 4x4 complexa [kPa] (Eq. 5.136a).

    Raises:
        ValueError: Se sin(ωd/cs*) ou sin(ωd/cp*) ≈ 0 (frequência de ressonância
                    da camada), tornando cot e csc numericamente indefinidos.
    """
    G   = params['G_complex']
    cp  = params['cp_star']
    cs  = params['cs_star']
    d   = params['d']

    # Argumentos angulares (complexos com amortecimento)
    arg_s = omega * d / cs   # ωd/cs*
    arg_p = omega * d / cp   # ωd/cp*

    sin_s = cmath.sin(arg_s)
    sin_p = cmath.sin(arg_p)

    # sin(ωd/cs*) = 0 ocorre em ω = n·π·cs*/d (ressonâncias da camada S)
    # sin(ωd/cp*) = 0 ocorre em ω = n·π·cp*/d (ressonâncias da camada P)
    # Nesses pontos, cot = cos/sin e csc = 1/sin divergem → resultado inválido.
    if abs(sin_s) < TOL_DENOM:
        raise ValueError(
            f"Frequência de ressonância (modo S) detectada: sin(ωd/cs*)≈0  "
            f"[arg_s={arg_s:.6f}, |sin|={abs(sin_s):.2e}]. "
            "Ajuste ω ou a espessura da camada."
        )
    if abs(sin_p) < TOL_DENOM:
        raise ValueError(
            f"Frequência de ressonância (modo P) detectada: sin(ωd/cp*)≈0  "
            f"[arg_p={arg_p:.6f}, |sin|={abs(sin_p):.2e}]. "
            "Ajuste ω ou a espessura da camada."
        )

    cot_s  = cmath.cos(arg_s) / sin_s   # cot(ωd/cs*)
    csc_s  = 1.0 / sin_s                # 1/sin(ωd/cs*)

    cot_p  = cmath.cos(arg_p) / sin_p   # cot(ωd/cp*)
    csc_p  = 1.0 / sin_p                # 1/sin(ωd/cp*)

    ratio  = cp / cs   # cp*/cs*

    # Eq. 5.136a — fator global G* ω/cs*
    fator = G * omega / cs

    K = np.zeros((4, 4), dtype=complex)

    K[0, 0] =  cot_s
    K[0, 2] = -csc_s
    K[1, 1] =  ratio * cot_p
    K[1, 3] = -ratio * csc_p
    K[2, 0] = -csc_s
    K[2, 2] =  cot_s
    K[3, 1] = -ratio * csc_p
    K[3, 3] =  ratio * cot_p

    return K * fator


def calcular_matriz_camada_estatica(params, k):
    """
    Matriz de rigidez para o caso estático (ω = 0, k ≠ 0).

    Caso especial 2 da Seção 5.4.3 de Wolf (1985).
    Implementa a Eq. 5.137a / Tabela 5-4.

    Para ω = 0, as ondas P e S degeneradas em funções hiperbólicas.
    A formulação usa velocidades de fase complexas (apenas razões cs*/cp*).

    Args:
        params (dict): Propriedades dinâmicas (G_complex, cp_star, cs_star, d).
        k (complex): Número de onda horizontal [rad/m].

    Returns:
        numpy.ndarray: Matriz 4x4 complexa [kPa] (Eq. 5.137a / Tabela 5-4).

    Raises:
        ValueError: Se o denominador D ≈ 0 (configuração degenerada de k e d).
    """
    G   = params['G_complex']
    cp  = params['cp_star']
    cs  = params['cs_star']
    d   = params['d']

    kd  = k * d
    r   = (cs / cp) ** 2  # (cs*/cp*)²

    sinh_kd = cmath.sinh(kd)
    cosh_kd = cmath.cosh(kd)

    # Denominador D (Tabela 5-4)
    D = (1 + r)**2 * sinh_kd**2 - kd**2 * (1 - r)**2

    # D = 0 ocorre quando (1+r)·sinh(kd) = ±(1-r)·kd, o que é possível
    # para certos pares (k, d, nu). Nesse caso, fator = 2kG/D → diverge.
    if abs(D) < TOL_DENOM:
        raise ValueError(
            f"Denominador D ≈ 0 em calcular_matriz_camada_estatica  "
            f"[D={D:.2e}, kd={kd:.6f}, r={r:.6f}]. "
            "Configuração degenerada: ajuste k ou a espessura da camada."
        )

    fator = 2 * k * G / D

    a = 1 + r
    b = 1 - r

    S11 =  a * sinh_kd * cosh_kd - b * kd
    S12 = -a * sinh_kd**2 + D
    S13 =  b * kd * cosh_kd - a * sinh_kd
    S14 =  kd * b * sinh_kd

    S22 =  a * sinh_kd * cosh_kd + kd * b
    S23 = -kd * b * sinh_kd
    S24 = -b * kd * cosh_kd - a * sinh_kd

    S33 =  a * sinh_kd * cosh_kd - b * kd
    S34 =  a * sinh_kd**2 - D

    S44 =  a * sinh_kd * cosh_kd + kd * b

    K = np.array([
        [ S11,  S12,  S13,  S14],
        [ S12,  S22,  S23,  S24],
        [ S13,  S23,  S33,  S34],
        [ S14,  S24,  S34,  S44]
    ], dtype=complex)

    return K * fator


def calcular_matriz_camada_estatica_k0(params):
    """
    Matriz de rigidez para o caso completamente estático (ω = 0, k = 0).

    Caso especial 3 da Seção 5.4.3 de Wolf (1985).
    Implementa a Eq. 5.138a.

    Representa o caso de carregamento estático uniforme sem variação espacial.
    A matriz é real (sem amortecimento relevante) e depende apenas da razão cp*/cs*.

    Args:
        params (dict): Propriedades dinâmicas (G_complex, cp_star, cs_star, d).

    Returns:
        numpy.ndarray: Matriz 4x4 complexa [kPa] (Eq. 5.138a).
    """
    G   = params['G_complex']
    cp  = params['cp_star']
    cs  = params['cs_star']
    d   = params['d']

    r   = (cp / cs) ** 2   # (cp*/cs*)²

    fator = G / d

    K = np.array([
        [ 1,  0, -1,  0],
        [ 0,  r,  0, -r],
        [-1,  0,  1,  0],
        [ 0, -r,  0,  r]
    ], dtype=complex)

    return K * fator


def calcular_matriz_camada(params, k, omega=None):
    """
    Calcula matriz de rigidez dinâmica 4x4 de uma camada horizontal.

    Roteia automaticamente para a formulação correta conforme o regime (ω, k):
        - ω > 0, k > 0 : formulação geral (Wolf Eq. 5.134 / Tabela 5-3)
        - ω > 0, k = 0 : incidência vertical (Wolf Eq. 5.136a)
        - ω = 0, k > 0 : caso estático (Wolf Eq. 5.137a / Tabela 5-4)
        - ω = 0, k = 0 : caso completamente estático (Wolf Eq. 5.138a)

    Args:
        params (dict): Propriedades dinâmicas da camada
                       (retorno de calcular_propriedades_dinamicas_locais):
            - G_complex (complex): Módulo de cisalhamento complexo [kPa]
            - d (float): Espessura da camada [m]
            - s (complex ou None): Parâmetro de propagação da onda P [-]
            - t (complex ou None): Parâmetro de propagação da onda S [-]
            - cp_star (complex): Velocidade de onda P [m/s]
            - cs_star (complex): Velocidade de onda S [m/s]
        k (float or complex): Número de onda horizontal [rad/m].
        omega (float, opcional): Frequência angular [rad/s].
                                 Necessário apenas para o roteamento de casos especiais.

    Returns:
        numpy.ndarray: Matriz de rigidez 4x4 complexa [kPa].

    Raises:
        ValueError: Se s·t for degenerado (≈ 0) no caso geral, ou se as funções
                    especializadas detectarem denominadores inválidos.
    """
    omega_val = omega if omega is not None else 0.0

    k_eh_zero     = abs(k)         < TOL_K_ZERO
    omega_eh_zero = abs(omega_val) < TOL_OMEGA_ZERO

    if omega_eh_zero and k_eh_zero:
        return calcular_matriz_camada_estatica_k0(params)

    if omega_eh_zero:
        return calcular_matriz_camada_estatica(params, k)

    if k_eh_zero:
        return calcular_matriz_camada_vertical(params, omega_val)

    # Caso geral: ω > 0, k > 0 → Tabela 5-3 / Eq. 5.134
    G = params['G_complex']
    d = params['d']
    s = params['s']
    t = params['t']

    s2          = s**2
    t2          = t**2
    one_plus_t2 = 1 + t2

    ksd = k * s * d
    ktd = k * t * d

    sin_ksd = cmath.sin(ksd)
    cos_ksd = cmath.cos(ksd)
    sin_ktd = cmath.sin(ktd)
    cos_ktd = cmath.cos(ktd)

    st = s * t
    if abs(st) < TOL_DEGENERADO:
        raise ValueError(
            f"s·t degenerado: s={s}, t={t}. Verifique k_global e as propriedades "
            "da camada. O denominador 1/(s·t) diverge neste regime."
        )

    term_st = st + 1 / st

    D = 2 * (1 - cos_ksd * cos_ktd) + term_st * sin_ksd * sin_ktd

    fator = (one_plus_t2 * k * G) / D

    t11 = (1/t) * cos_ksd * sin_ktd + s * sin_ksd * cos_ktd

    term_complex_12 = (1 + 2*s2*t2 - t2) / (st * one_plus_t2)
    t12 = ((3 - t2) / one_plus_t2) * (1 - cos_ksd*cos_ktd) + term_complex_12 * sin_ksd * sin_ktd

    t13 = -s * sin_ksd - (1/t) * sin_ktd
    t14 = cos_ksd - cos_ktd
    t22 = (1/s) * sin_ksd * cos_ktd + t * cos_ksd * sin_ktd
    t23 = -cos_ksd + cos_ktd
    t24 = -(1/s) * sin_ksd - t * sin_ktd

    S = np.array([
        [t11,  t12,  t13,   t14],
        [t12,  t22,  t23,   t24],
        [t13,  t23,  t11,  -t12],
        [t14,  t24, -t12,   t22]
    ], dtype=complex)

    return S * fator


def calcular_matriz_semiespaco(params, k, omega=None):
    """
    Calcula matriz de rigidez 2x2 do semi-espaço infinito (rocha base).

    Roteia para a formulação correta conforme o regime (ω, k):
        - ω > 0, k > 0 : formulação geral (Wolf Eq. 5.135)
        - ω > 0, k = 0 : ondas verticais (Wolf Eq. 5.136b)
        - ω = 0, k ≠ 0 : caso estático (Wolf Eq. 5.137b)
        - ω = 0, k = 0 : matriz nula (Wolf Eq. 5.138b)

    Args:
        params (dict): Propriedades dinâmicas do semi-espaço.
        k (float or complex): Número de onda horizontal [rad/m].
        omega (float, opcional): Frequência angular [rad/s].

    Returns:
        numpy.ndarray: Matriz de rigidez 2x2 complexa [kPa].
    """
    omega_val = omega if omega is not None else 0.0

    k_eh_zero     = abs(k)         < TOL_K_ZERO
    omega_eh_zero = abs(omega_val) < TOL_OMEGA_ZERO

    G   = params['G_complex']
    cp  = params['cp_star']
    cs  = params['cs_star']

    if omega_eh_zero and k_eh_zero:
        return np.zeros((2, 2), dtype=complex)

    if omega_eh_zero:
        r_sq = (cs / cp) ** 2
        a11  = 1.0 / (1 + r_sq)
        a12  = 1.0 / (1 + 1.0 / r_sq)
        return 2 * k * G * np.array([[a11, a12],
                                     [a12, a11]], dtype=complex)

    if k_eh_zero:
        ratio = cp / cs
        fator = 1j * G * omega_val / cs
        return fator * np.array([[1,      0],
                                 [0,  ratio]], dtype=complex)

    s = params['s']
    t = params['t']

    t2          = t**2
    one_plus_t2 = 1 + t2
    denom       = 1 + s * t

    k11 = (s * one_plus_t2) * 1j / denom
    k22 = (t * one_plus_t2) * 1j / denom
    k12 = 2 - one_plus_t2 / denom

    return k * G * np.array([[k11, k12],
                              [k12, k22]], dtype=complex)

# ==============================================================================
# 3. MONTAGEM DO SISTEMA GLOBAL
# ==============================================================================

def montar_sistema_global(dados_json, omega, angulo_graus, gl_total):
    """
    Monta matriz de rigidez dinâmica global do sistema estratificado completo.

    Args:
        dados_json (dict): Dados do modelo contendo:
            - 'camadas' (list): Lista de dicionários com propriedades de cada camada
            - 'semi_espaco' (dict, opcional): Propriedades do semi-espaço inferior
        omega (float): Frequência angular de excitação [rad/s].
        angulo_graus (float): Ângulo de incidência da onda P na primeira camada [graus].
                              Medido em relação à vertical (0° = incidência vertical).
        gl_total (int): Número total de graus de liberdade do perfil [-].

    Returns:
        tuple: (K_global, k_global, dados_validacao)
    """
    camadas     = dados_json['camadas']
    semi_espaco = dados_json.get('semi_espaco')
    num_camadas = len(camadas)

    K_global = np.zeros((gl_total, gl_total), dtype=complex)

    props_ref  = camadas[0]
    params_ref = calcular_propriedades_dinamicas_locais(props_ref, omega)

    lx_ref = np.cos(np.radians(angulo_graus))

    if abs(lx_ref) < TOL_ANGULO_90:
        lx_ref = TOL_ANGULO_90 + 0j

    if abs(omega) < TOL_OMEGA_ZERO:
        k_global = complex(0.0)
        c_fase   = complex(float('inf'))
    else:
        c_fase   = params_ref['cp_star'] / lx_ref
        k_global = omega / c_fase

    params_cache = [
        calcular_propriedades_dinamicas_locais(props, omega, k_global)
        for props in camadas
    ]

    dados_validacao = {
        'c_fase': c_fase,
        'l'     : complex(lx_ref),
        'm'     : params_cache[0]['cs_star'] / c_fase if abs(c_fase) > 1e-30 else complex(0),
        's'     : params_cache[0]['s'],
        't'     : params_cache[0]['t']
    }

    for i, params_locais in enumerate(params_cache):
        S_elem = calcular_matriz_camada(params_locais, k_global, omega=omega)
        idx    = 2 * i
        K_global[idx:idx+4, idx:idx+4] += S_elem

    if semi_espaco:
        params_semi = calcular_propriedades_dinamicas_locais(semi_espaco, omega, k_global)
        S_bedrock   = calcular_matriz_semiespaco(params_semi, k_global, omega=omega)
        idx         = 2 * num_camadas
        K_global[idx:idx+2, idx:idx+2] += S_bedrock

    return K_global, k_global, dados_validacao


def montar_sistema_global_por_k(dados_json, omega, k_global, gl_total):
    """
    Monta matriz de rigidez dinâmica global usando k_global fornecido diretamente.

    Args:
        dados_json (dict): Dados do modelo.
        omega (float): Frequência angular [rad/s].
        k_global (complex): Número de onda horizontal imposto [rad/m].
        gl_total (int): Número total de graus de liberdade do perfil [-].

    Returns:
        tuple: (K_global, k_global, dados_validacao)
    """
    camadas     = dados_json['camadas']
    semi_espaco = dados_json.get('semi_espaco')
    num_camadas = len(camadas)

    K_global_mat = np.zeros((gl_total, gl_total), dtype=complex)

    params_cache = [
        calcular_propriedades_dinamicas_locais(props, omega, k_global)
        for props in camadas
    ]

    cp_star_ref   = params_cache[0]['cp_star']
    k_eh_zero     = abs(k_global) < TOL_K_ZERO
    omega_eh_zero = abs(omega)    < TOL_OMEGA_ZERO

    if k_eh_zero:
        c_fase       = complex(float('inf'))
        angulo_equiv = 0.0
    elif omega_eh_zero:
        c_fase       = complex(0.0)
        angulo_equiv = float('nan')
    else:
        c_fase  = omega / k_global
        lx_real = (cp_star_ref / c_fase).real
        if abs(lx_real) <= 1.0:
            angulo_equiv = np.degrees(np.arccos(np.clip(lx_real, -1.0, 1.0)))
        else:
            angulo_equiv = float('nan')

    dados_validacao = {
        'c_fase'       : c_fase,
        'l'            : cp_star_ref / c_fase if not k_eh_zero else complex(1),
        'm'            : params_cache[0]['cs_star'] / c_fase if not k_eh_zero else complex(0),
        's'            : params_cache[0]['s'],
        't'            : params_cache[0]['t'],
        'angulo_equiv' : angulo_equiv if not k_eh_zero else 0.0,
    }

    for i, params_locais in enumerate(params_cache):
        S_elem = calcular_matriz_camada(params_locais, k_global, omega=omega)
        idx    = 2 * i
        K_global_mat[idx:idx+4, idx:idx+4] += S_elem

    if semi_espaco:
        params_semi = calcular_propriedades_dinamicas_locais(semi_espaco, omega, k_global)
        S_bedrock   = calcular_matriz_semiespaco(params_semi, k_global, omega=omega)
        idx         = 2 * num_camadas
        K_global_mat[idx:idx+2, idx:idx+2] += S_bedrock

    return K_global_mat, k_global, dados_validacao

# ==============================================================================
# 4. FORMATAÇÃO E I/O
# ==============================================================================

def formatar_resultado(label_valor, K_total, k_calc, debug, num_gl, modo):
    """
    Formata o resultado de um caso (ângulo ou k_global) em string pronta para escrita.
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
    c_fase = debug['c_fase']
    if abs(c_fase) == float('inf'):
        linhas.append(f"  c (velocidade fase) = inf  [m/s]  (k=0, incidência vertical)\n")
    else:
        linhas.append(f"  c (velocidade fase) = {c_fase.real:+.4e} {c_fase.imag:+.4e}j  [m/s]\n")
    linhas.append(f"  l (cos psi_P)       = {debug['l'].real:+.8f} {debug['l'].imag:+.8f}j\n")
    linhas.append(f"  m (cos psi_S)       = {debug['m'].real:+.8f} {debug['m'].imag:+.8f}j\n")
    s_val = debug['s']
    t_val = debug['t']
    if s_val is None:
        linhas.append(f"  s (param onda P)    = N/A (caso especial)\n")
        linhas.append(f"  t (param onda S)    = N/A (caso especial)\n")
    else:
        linhas.append(f"  s (param onda P)    = {s_val.real:+.8f} {s_val.imag:+.8f}j\n")
        linhas.append(f"  t (param onda S)    = {t_val.real:+.8f} {t_val.imag:+.8f}j\n")
    linhas.append(f"  k (num onda horiz)  = {k_calc.real:+.6e} {k_calc.imag:+.6e}j  [rad/m]\n")
    linhas.append(f"  |k|                 = {abs(k_calc):+.6e}  [rad/m]\n")

    if modo == 'k_global':
        angulo_equiv = debug.get('angulo_equiv', float('nan'))
        if np.isnan(angulo_equiv):
            linhas.append(f"  angulo equiv (psi_P) = n/a (k fora da faixa física da camada 1)\n")
        else:
            linhas.append(f"  angulo equiv (psi_P) = {angulo_equiv:.4f} graus\n")

    linhas.append("\n")
    linhas.append("MATRIZ K [kPa]:\n")
    for i in range(num_gl):
        for j in range(num_gl):
            val = K_total[i, j]
            linhas.append(f"  {val.real:+.4e} {val.imag:+.4e}j")
        linhas.append("\n")
    linhas.append("\n" + "=" * 80 + "\n\n")
    return "".join(linhas)


def salvar_resultados(resultados, arquivo_saida, omega, num_camadas,
                      tem_semi_espaco, num_gl, modo, valores_entrada):
    """
    Salva todos os resultados formatados em um arquivo de saída.
    """
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  MATRIZ DE RIGIDEZ DINAMICA GLOBAL - ONDAS P-SV\n")
        f.write("  Formulacao: Wolf (1985) - Dynamic Soil-Structure Interaction\n")
        f.write("=" * 80 + "\n\n")
        f.write("CONFIGURACAO:\n")
        f.write(f"  Frequencia .......: omega = {omega} rad/s  (f = {omega/(2*np.pi):.4f} Hz)\n")
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
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================

def _resumir_valores_log(valores, n_limiar=5):
    """
    Retorna string resumida dos valores para uso no log e no arquivo de saída.

    Até n_limiar itens: exibe a lista completa.
    Acima: exibe qtd + min + max + passo (se uniforme).

    Args:
        valores (list): Lista de floats ou complexos.
        n_limiar (int): Quantidade a partir da qual aplica o resumo.

    Returns:
        str: Representação compacta dos valores.
    """
    n = len(valores)

    if n <= n_limiar:
        return str(valores)

    # Para floats (ângulos): usa min/max direto para preservar sinal.
    # Para complex: usa abs() pois não há ordenação natural.
    eh_complex = isinstance(valores[0], complex) and valores[0].imag != 0
    if eh_complex:
        vmin = min(abs(v) for v in valores)
        vmax = max(abs(v) for v in valores)
    else:
        vmin = min(float(v) for v in valores)
        vmax = max(float(v) for v in valores)

    # Detecta passo uniforme comparando os primeiros 5 intervalos
    passo_str = ""
    if n >= 2:
        diffs = [valores[i+1] - valores[i] for i in range(min(5, n - 1))]
        ref   = diffs[0]
        tol   = 1e-9 * (abs(ref) + 1e-30)
        if all(abs(d - ref) < tol for d in diffs):
            passo_str = f", passo={ref:.6g}"

    return f"{n} valores | min={vmin:.6g}  max={vmax:.6g}{passo_str}"


def _resolver_valores(analise, modo):
    """
    Converte 'analise.valores' (lista ou range) em uma lista Python concreta.

    Não emite logs — responsabilidade de log fica em main().
    """
    valores_raw = analise["valores"]

    if isinstance(valores_raw, dict):
        inicio = float(valores_raw["inicio"])
        fim    = float(valores_raw["fim"])
        passo  = float(valores_raw["passo"])
        valores_expandidos = np.arange(inicio, fim + passo / 2, passo).tolist()
    else:
        valores_expandidos = valores_raw

    if modo == "k_global":
        k_globais_teste = []
        for v in valores_expandidos:
            if isinstance(v, list):
                if len(v) != 2:
                    logger.error(
                        f"Valor inválido em 'analise.valores': {v}. "
                        "Para k complexo use [real, imag], ex: [0.05, 0.0]."
                    )
                    sys.exit(1)
                k_globais_teste.append(complex(v[0], v[1]))
            else:
                k_globais_teste.append(complex(v))
        return [], k_globais_teste
    else:
        return [float(v) for v in valores_expandidos], []


def configurar_analise(dados):
    """
    Extrai e retorna os parâmetros de análise diretamente do bloco 'analise' do JSON.
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
        'arquivo_saida'  : 'SAIDA_Rigidez_Global.txt',
        'num_gl'         : num_gl,
    }


def executar_loop_angulos(dados, omega, angulos_teste, num_gl, **kwargs):
    """
    Executa a análise para cada ângulo de incidência.

    Erros numéricos em casos individuais (ex: ressonância, s·t degenerado) são
    capturados, logados com o ângulo causador e pulados — os demais casos continuam.
    """
    resultados = []

    for angulo in tqdm(
        angulos_teste,
        desc="Processando (ângulos)",
        unit=" caso",
        ncols=80,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ):
        try:
            K_total, k_calc, debug = montar_sistema_global(dados, omega, angulo, num_gl)
        except ValueError as e:
            logger.warning(f"Caso ignorado (angulo={angulo:.4f}°): {e}")
            continue

        resultados.append(
            formatar_resultado(angulo, K_total, k_calc, debug, num_gl, modo='angulos')
        )

    return resultados


def executar_loop_k_global(dados, omega, k_globais_teste, num_gl, **kwargs):
    """
    Executa a análise para cada valor de k_global fornecido diretamente.

    Erros numéricos em casos individuais são capturados, logados com o k causador
    e pulados — os demais casos continuam.
    """
    resultados = []

    for k in tqdm(
        k_globais_teste,
        desc="Processando (k_global)",
        unit=" caso",
        ncols=80,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ):
        k_complex = complex(k)
        try:
            K_total, k_calc, debug = montar_sistema_global_por_k(dados, omega, k_complex, num_gl)
        except ValueError as e:
            logger.warning(f"Caso ignorado (k={k_complex:.4e}): {e}")
            continue

        resultados.append(
            formatar_resultado(k_complex, K_total, k_calc, debug, num_gl, modo='k_global')
        )

    return resultados


def main():
    """
    Função principal: executa análise paramétrica de rigidez dinâmica.
    """
    arquivo_input = 'camada.json'
    dados         = ler_arquivo_entrada(arquivo_input)
    cfg           = configurar_analise(dados)

    modo = cfg['modo']
    if modo not in ('angulos', 'k_global'):
        logger.error(f"Modo inválido: '{modo}'. Use 'angulos' ou 'k_global'.")
        sys.exit(1)

    valores_entrada = cfg['angulos_teste'] if modo == 'angulos' else cfg['k_globais_teste']

    logger.info("=" * 60)
    logger.info("  ANALISE DE RIGIDEZ DINAMICA - METODO DE WOLF")
    logger.info("=" * 60)
    logger.info(f"  Camadas ........: {len(dados['camadas'])}")
    logger.info(f"  Semi-espaco ....: {'Sim' if dados.get('semi_espaco') else 'Nao'}")
    logger.info(f"  Graus liberdade : {cfg['num_gl']}")
    logger.info(f"  Frequencia .....: omega = {cfg['omega']} rad/s  "
                f"(f = {cfg['omega']/(2*np.pi):.4f} Hz)")
    logger.info(f"  Modo ...........: {modo}")
    logger.info(f"  Valores entrada : {_resumir_valores_log(valores_entrada)}")
    logger.info(f"  Arquivo saida ..: {cfg['arquivo_saida']}")
    logger.info("=" * 60)

    if modo == 'angulos':
        resultados = executar_loop_angulos(dados, **cfg)
    else:
        resultados = executar_loop_k_global(dados, **cfg)

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

    logger.info("Análise concluída!")


if __name__ == "__main__":
    main()