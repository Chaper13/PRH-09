import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time

# =============================================================================
# MÓDULO DE SIMULAÇÃO DE MOLAS
# =============================================================================

def achar_deslocamentos(ks, fe):
    """
    Calcula os deslocamentos de um sistema de molas em série.
    
    O sistema é modelado com o primeiro nó livre (recebendo a força) e o último nó engastado (parede).
    Utiliza matrizes esparsas para otimização de memória e performance.

    Args:
        ks (List[float]): Lista contendo a rigidez (k) de cada mola em N/m.
        fe (float): Força externa aplicada no primeiro nó (N).

    Returns:
        Tuple[u, reacao, duracao]: 
            - u (np.array): Array com os deslocamentos de cada nó (incluindo o zero da parede).
            - reacao (float): Valor da reação de apoio na parede (N).
            - duracao (float): Tempo de execução do cálculo (segundos).
    """
    inicio = time.time()
    
    n = len(ks)             # Número de elementos (molas)
    ks = np.array(ks)       # Converte para numpy array para permitir vetorização
    indices = np.arange(n)  # Índices das molas [0, 1, 2, ..., n-1]

    # --- MONTAGEM DA MATRIZ DE RIGIDEZ GLOBAL (Formato COO) ---
    # Em vez de um loop 'for', é usado vetorização do Numpy para criar os índices.
    # A matriz local de uma mola é: [[k, -k], 
    #                                [-k, k]]
    # mapeando para os nós [i, i+1].
    
    # Define as coordenadas (linha, coluna) para os valores na matriz esparsa
    linhas = np.concatenate([indices, indices, indices + 1, indices + 1])
    colunas = np.concatenate([indices, indices + 1, indices, indices + 1])
    
    # Define os valores correspondentes a essas coordenadas
    # Estrutura: diagonal principal (ks, ks) e diagonais secundárias (-ks, -ks)
    dados = np.concatenate([ks, -ks, -ks, ks])

    # Cria a matriz esparsa no formato Coordinate (COO). 
    # Shape é (n+1, n+1) pois n molas possuem n+1 nós.
    k_global = coo_matrix((dados, (linhas, colunas)), shape=(n + 1, n + 1))

    # Converte para CSC (Compressed Sparse Column) para permitir fatiamento (slicing) eficiente
    k_global_csc = k_global.tocsc()

    # --- APLICAÇÃO DAS CONDIÇÕES DE CONTORNO ---
    # O último nó (índice n) está fixo na parede -> deslocamento u[n] = 0.
    # Removemos a última linha e coluna da matriz para resolver apenas os graus de liberdade livres.
    k_reduzida = k_global_csc[:-1, :-1]

    # --- VETOR DE FORÇAS ---
    f = np.zeros(n)
    f[0] = fe  # Aplica a força externa apenas no primeiro nó (índice 0)

    # --- SOLUÇÃO DO SISTEMA LINEAR ---
    # Resolve [K]{u} = {f} usando solver otimizado para matrizes esparsas
    u_livres = spsolve(k_reduzida, f)
    
    # Reconstrói o vetor total adicionando o deslocamento zero do nó engastado
    u = np.append(u_livres, 0)
    
    # --- CÁLCULO DA REAÇÃO DE APOIO ---
    # A reação é calculada pela lei de Hooke na última mola (conectada à parede)
    # R = -k * (u_final - u_anterior)
    reacao = -ks[-1] * (u[-1] - u[-2])
    
    fim = time.time()
    duracao = fim - inicio
    
    return u, reacao, duracao

def gerar_grafico(ks, deslocamentos, fe):
    """
    Plota a representação física das molas antes e depois da deformação.
    
    Args:
        ks (np.ndarray): Array de rigidezes (usado apenas para metadados/tamanho).
        deslocamentos (np.ndarray): Vetor de deslocamentos nodais.
        fe (float): Valor da força externa para o título.
    """
    n = len(ks)
    
    # Define a posição inicial dos nós (assumindo comprimento de repouso L=1m para visualização)
    # Ex: [0, -1, -2, -3...]
    p_inicial = np.arange(0, -(n + 1), -1) 
    
    # A posição final é a inicial + o deslocamento calculado
    p_final = p_inicial + deslocamentos

    plt.figure(figsize=(10, 5))
    
    # Plota a configuração original (Repouso)
    plt.plot(p_inicial, np.ones(n + 1), 'o-', label='Repouso', color='blue', alpha=0.5)
    
    # Plota a configuração deformada
    plt.plot(p_final, np.zeros(n + 1), 'o-', label='Deformada', color='red', markersize=4)

    # Configurações estéticas do gráfico
    plt.yticks([0, 1], ['Carga Aplicada', 'Posição Inicial'])
    plt.xlabel('Posição X (m)')
    plt.title(f'Simulação de {n} Molas (Esparso) - Força: {fe}N')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    Função principal que gerencia a entrada do usuário e executa a simulação.
    """
    print("=== SIMULADOR DE MOLAS 1D ===")
    print("[1] Modo Manual (Inserir cada mola + Gráfico)")
    print("[2] Modo Automático (Gerar molas aleatórias - Sem Gráfico)")
    
    opcao = input("\nEscolha o modo: ")

    try:
        if opcao == "1":
            # --- MODO MANUAL ---
            n_molas = int(input("Número de molas: "))
            # List comprehension para pegar input de cada mola
            ks = [float(input(f"Rigidez k da mola {i+1} [N/m]: ")) for i in range(n_molas)]
            fe = float(input("Força externa (F) [N]: "))
            mostrar_grafico = True
            
        elif opcao == "2":
            # --- MODO AUTOMÁTICO ---
            n_molas = int(input("Número de molas: "))
            k_min = float(input("Rigidez mínima [N/m]: "))
            k_max = float(input("Rigidez máxima [N/m]: "))
            fe = float(input("Força externa (F) [N]: "))
            
            # Gera rigidezes aleatórias com distribuição uniforme
            ks = np.random.uniform(k_min, k_max, n_molas)
            mostrar_grafico = False # Desativa gráfico para grandes quantidades
        else:
            print("Opção inválida.")
            return

        # Executa o cálculo central
        deslocamentos, reacao, duracao = achar_deslocamentos(ks, fe)

        # --- RELATÓRIO DE SAÍDA ---
        print("\n" + "="*50)
        print(f"{'RELATÓRIO DA SIMULAÇÃO':^50}")
        print("="*50)
        print(f"A execução durou {duracao:.3f} segundos")
        
        # Exibe nós detalhados apenas se houver poucos (para não poluir o terminal)
        if n_molas <= 25:
            for i, d in enumerate(deslocamentos):
                # Identifica visualmente qual é a parede e qual é o nó livre
                tipo = "(Parede)" if i == len(deslocamentos) - 1 else "(Nó livre)" if i == 0 else ""
                print(f"Nó {i:02d}: Deslocamento = {d:8.4f} m {tipo}")
        else:
            # Resumo para grandes sistemas
            print(f"Nó 00 (Nó livre): {deslocamentos[0]:.4f} m")
            print("... (dados omitidos para muitos nós) ...")
            print(f"Nó {n_molas:02d} (Parede): {deslocamentos[-1]:.4f} m")

        print("-" * 50)
        print(f"Reação de Apoio na Parede: {reacao:.4f} N")
        print("=" * 50)

        if mostrar_grafico:
            gerar_grafico(ks, deslocamentos, fe)

    except ValueError:
        print("Erro: Digite apenas valores numéricos válidos.")

if __name__ == "__main__":
    main()
