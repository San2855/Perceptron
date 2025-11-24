# perceptron_treinamento.py
# Implementação simples do perceptron + treino no exemplo dos alunos

def step(u: float) -> int:

    return 1 if u >= 0 else 0


def perceptron_update(x, w, y, alpha):
    
    # produto interno w.x
    u = sum(w_i * x_i for w_i, x_i in zip(w, x))

    # saída do neurônio
    y_hat = step(u)

    # erro
    erro = y - y_hat

    # regra de atualização dos pesos: w(n+1) = w(n) + alpha * erro * x
    w_novo = [
        w_i + alpha * erro * x_i
        for w_i, x_i in zip(w, x)
    ]

    return w_novo, u, y_hat, erro


def main():
    # ---------------------------
    # Base de dados do exercício
    # ---------------------------
    # Codificação:
    # Sim = 1, Não = 0
    #
    # Atributos: [bias, estudou, fez_trabalho]
    X = [
        [1, 0, 0],  # Joãozinho  -> Não passou
        [1, 0, 1],  # Huguinho   -> Não passou
        [1, 1, 0],  # Zezinho    -> Passou
        [1, 1, 1],  # Luizinho   -> Passou
    ]

    # Saídas desejadas (Passou: Não=0, Sim=1)
    y = [0, 0, 1, 1]

    # Hiperparâmetros
    alpha = 0.1            # taxa de aprendizagem
    w = [0.0, 0.0, 0.0]    # pesos iniciais (todos 0)
    num_epocas = 2         # “dois ciclos” sobre a base

    log_lines = []
    log_lines.append("Treinamento do Perceptron - Exemplo da Aula\n")
    log_lines.append(f"Taxa de aprendizagem (alpha): {alpha}\n")
    log_lines.append(f"Pesos iniciais: {w}\n\n")

    for epoca in range(num_epocas):
        log_lines.append(f"========== ÉPOCA {epoca + 1} ==========\n")
        for i, (x_vec, target) in enumerate(zip(X, y), start=1):
            w_antes = w.copy()
            w, u, y_hat, erro = perceptron_update(x_vec, w, target, alpha)

            linha = (
                f"Época {epoca + 1}, exemplo {i}\n"
                f"  x       = {x_vec}\n"
                f"  w antes = {w_antes}\n"
                f"  u       = {u:.2f}\n"
                f"  y_hat   = {y_hat}\n"
                f"  y real  = {target}\n"
                f"  erro    = {erro}\n"
                f"  w depois= {w}\n\n"
            )

            # imprime no console
            print(linha)
            # guarda no log
            log_lines.append(linha)

    # salva o log em arquivo TXT
    with open("treino_perceptron.txt", "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    print("Treinamento concluído. Detalhes salvos em 'treino_perceptron.txt'.")


if __name__ == "__main__":
    main()
