import numpy as np

EPS = 1e-9


def build_tableau(a_matrix, b_vector, c_vector):
    a_matrix = np.array(a_matrix, dtype=float)
    b_vector = np.array(b_vector, dtype=float)
    c_vector = np.array(c_vector, dtype=float)

    m, n = a_matrix.shape

    tableau = np.zeros((m + 1, n + 1), dtype=float)
    tableau[:m, 0] = b_vector
    tableau[:m, 1:] = a_matrix
    tableau[m, 1:] = -c_vector

    return tableau


def choose_pivot(tableau):
    rows = tableau.shape[0] - 1
    cols = tableau.shape[1] - 1

    z_row = tableau[rows, 1:]

    if np.all(z_row >= -EPS):
        return -1, -1

    pivot_col = np.argmin(z_row) + 1

    col = tableau[:rows, pivot_col]
    rhs = tableau[:rows, 0]

    valid = col > EPS
    if not np.any(valid):
        return None, None

    ratios = np.full(rows, np.inf)
    ratios[valid] = rhs[valid] / col[valid]

    pivot_row = np.argmin(ratios)
    return pivot_row, pivot_col


def pivot_transform(tableau, pivot_row, pivot_col):
    pivot_elem = tableau[pivot_row, pivot_col]
    tableau[pivot_row, :] /= pivot_elem

    for i in range(tableau.shape[0]):
        if i != pivot_row:
            factor = tableau[i, pivot_col]
            tableau[i, :] -= factor * tableau[pivot_row, :]


def print_tableau(tableau, basis, iteration):
    rows, cols = tableau.shape
    var_count = cols - 1

    print(f"\nИтерация {iteration}")
    print("-" * (12 * (var_count + 2)))

    header = f"{'Базис':>8}{'b':>10}"
    for j in range(1, var_count + 1):
        header += f"{('x' + str(j)):>10}"
    print(header)

    for i in range(rows):
        if i < rows - 1:
            row_name = f"x{basis[i]}"
        else:
            row_name = "z"

        line = f"{row_name:>8}"
        for j in range(cols):
            line += f"{tableau[i, j]:>10.4f}"
        print(line)


def simplex_solve(a_matrix, b_vector, c_vector, basis, verbose=True):
    tableau = build_tableau(a_matrix, b_vector, c_vector)
    basis = basis[:]

    iteration = 1

    while True:
        if verbose:
            print_tableau(tableau, basis, iteration)

        pivot_row, pivot_col = choose_pivot(tableau)

        if pivot_row == -1:
            solution = np.zeros(tableau.shape[1] - 1)
            for i in range(len(basis)):
                solution[basis[i] - 1] = tableau[i, 0]
            optimum = tableau[-1, 0]
            return "ok", solution, optimum, tableau

        if pivot_row is None:
            return "Целевая функция не ограничена", None, None, tableau

        pivot_transform(tableau, pivot_row, pivot_col)
        basis[pivot_row] = pivot_col

        iteration += 1


if __name__ == "__main__":
    # Вариант 8
    # z = x1 + 3x2 - 5x4 -> max
    # 3x1 + 4x2 + x3 + 2x4 = 27
    # -4x1 + 5x2 - 3x4 + x5 = 32
    # 5x1 - 2x2 + 8x4 + x6 = 24

    A = [
        [3, 4, 1, 2, 0, 0],
        [-4, 5, 0, -3, 1, 0],
        [5, -2, 0, 8, 0, 1]
    ]

    b = [27, 32, 24]
    c = [1, 3, 0, -5, 0, 0]

    # Начальный базис: x3, x5, x6
    basis = [3, 5, 6]

    status, solution, optimum, final_tableau = simplex_solve(A, b, c, basis, verbose=True)

    print("\nСтатус:", status)
    if status == "ok":
        print("Оптимальный план x:", solution)
        print("Максимум z:", optimum)