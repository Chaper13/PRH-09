"""
plot_fww.py
-----------
Script independente que calcula F[1,1] (componente Fww da interface superior)
em função de k_global e plota Re e Im em subplots separados.

Configuração: edite o bloco PARAMETROS abaixo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# PARAMETROS
# ==============================================================================

# Frequência angular [rad/s]
OMEGA = 10.0

# Vetor de k_global [rad/m]
K_INICIO = 0
K_FIM    = 8.0
K_PASSO  = 0.001

# Limite de visualização no eixo x do gráfico [rad/m]
# (não afeta o cálculo — apenas o zoom do plot)
# Use None para exibir o intervalo completo calculado.
K_XLIM_PLOT = 0.2

# Arquivo JSON do modelo (mesmo usado no código principal)
ARQUIVO_JSON = "camada.json"

# Arquivo de imagem gerado (None = apenas exibir na tela)
ARQUIVO_SAIDA = "plot_fww.png"

# ==============================================================================
# IMPORTS DO CODIGO PRINCIPAL
# (o script deve estar na mesma pasta que rigidez_psv.py)
# ==============================================================================

try:
    from rigidez_psv import (
        ler_arquivo_entrada,
        precalcular_propriedades_materiais,
        calcular_s_t_vetorizado,
        montar_K_global_vetorizado,
        TOL_K_ZERO,
        calcular_propriedades_dinamicas_locais,
        calcular_matriz_camada_escalar,
        calcular_matriz_semiespaco_escalar,
    )
except ImportError as e:
    raise ImportError(
        "Não foi possível importar rigidez_psv.py. "
        "Certifique-se de que este script está na mesma pasta."
    ) from e

# ==============================================================================
# CALCULO
# ==============================================================================

def calcular_F11_vs_k(arquivo_json, omega, k_inicio, k_fim, k_passo):
    """
    Calcula o elemento F[1,1] da matriz de flexibilidade para cada k_global.

    Returns:
        tuple[ndarray, ndarray]: (k_arr, F11_arr)
    """
    dados     = ler_arquivo_entrada(arquivo_json)
    num_gl    = 2 * (len(dados['camadas']) + 1)
    camadas   = dados['camadas']
    semi      = dados.get('semi_espaco')
    mat_props = precalcular_propriedades_materiais(dados)

    k_arr  = np.arange(k_inicio, k_fim + k_passo / 2, k_passo, dtype=float)
    k_cplx = k_arr.astype(complex)
    N      = len(k_arr)

    F11 = np.full(N, np.nan, dtype=complex)

    # --- caso geral (omega > 0, k > 0) em batch ---
    mask_geral   = k_arr >= TOL_K_ZERO
    idx_geral    = np.where(mask_geral)[0]
    idx_especial = np.where(~mask_geral)[0]

    if len(idx_geral) > 0:
        k_g          = k_cplx[idx_geral]
        s_all, t_all = calcular_s_t_vetorizado(mat_props, omega, k_g)
        K_batch      = montar_K_global_vetorizado(
            mat_props, s_all, t_all, k_g, omega, num_gl
        )
        for bi, oi in enumerate(idx_geral):
            try:
                F = np.linalg.solve(K_batch[bi], np.eye(num_gl, dtype=complex))
                F11[oi] = F[1, 1]
            except np.linalg.LinAlgError:
                pass  # deixa NaN

    # --- casos especiais (k ≈ 0) ---
    for oi in idx_especial:
        kv    = k_cplx[oi]
        K_mat = np.zeros((num_gl, num_gl), dtype=complex)
        ok    = True

        for i, props in enumerate(camadas):
            pl = calcular_propriedades_dinamicas_locais(props, omega)
            try:
                S = calcular_matriz_camada_escalar(pl, kv, omega=omega)
                K_mat[2*i:2*i+4, 2*i:2*i+4] += S
            except ValueError:
                ok = False
                break

        if ok and semi:
            ps = calcular_propriedades_dinamicas_locais(semi, omega)
            try:
                Sb = calcular_matriz_semiespaco_escalar(ps, kv, omega=omega)
                ix = 2 * len(camadas)
                K_mat[ix:ix+2, ix:ix+2] += Sb
            except ValueError:
                ok = False

        if ok:
            try:
                F = np.linalg.solve(K_mat, np.eye(num_gl, dtype=complex))
                F11[oi] = F[1, 1]
            except np.linalg.LinAlgError:
                pass

    return k_arr, F11

# ==============================================================================
# PLOT
# ==============================================================================

def plotar(k_arr, F11, omega, arquivo_saida=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        r"Flexibilidade $F_{ww}$ = F[1,1]  vs  $k_{global}$"
        f"\n$\\omega$ = {omega} rad/s",
        fontsize=13
    )

    # --- Re ---
    ax1.plot(k_arr, F11.real, color='steelblue', linewidth=1.2)
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax1.set_ylabel(r"Re$(F_{ww})$  [m/kPa]", fontsize=11)
    ax1.grid(True, alpha=0.35)
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.tick_params(which='both', direction='in')

    # --- Im ---
    ax2.plot(k_arr, F11.imag, color='tomato', linewidth=1.2)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_ylabel(r"Im$(F_{ww})$  [m/kPa]", fontsize=11)
    ax2.set_xlabel(r"$k_{global}$  [rad/m]", fontsize=11)
    ax2.grid(True, alpha=0.35)
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.tick_params(which='both', direction='in')

    plt.tight_layout()

    # --- Intervalo de visualização no eixo x (não afeta o cálculo) ---
    if K_XLIM_PLOT is not None:
        ax2.set_xlim(0, K_XLIM_PLOT)
        ax2.set_xticks(np.arange(0, K_XLIM_PLOT + K_XLIM_PLOT / 10, K_XLIM_PLOT / 10))
    else:
        ax2.set_xlim(k_arr[0], k_arr[-1])

    if arquivo_saida:
        plt.savefig(arquivo_saida, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {arquivo_saida}")

    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print(f"Calculando F[1,1] para k = [{K_INICIO}, {K_FIM}] passo {K_PASSO} ...")
    k_arr, F11 = calcular_F11_vs_k(
        ARQUIVO_JSON, OMEGA, K_INICIO, K_FIM, K_PASSO
    )
    n_validos = np.sum(~np.isnan(F11.real))
    print(f"Pontos calculados: {n_validos} / {len(k_arr)}")
    plotar(k_arr, F11, omega=OMEGA, arquivo_saida=ARQUIVO_SAIDA)