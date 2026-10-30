"""Punto 1. Distribuciones discretas y decisiones de respuesta."""
import math


def normalize(weights: dict) -> dict:
    """Retorna un diccionario nuevo cuyas probabilidades suman uno.

    Entrada: weights = {resultado: peso}.
    Salida: {resultado: probabilidad}, con las mismas claves.

    Tips:
    - Conserve las claves y su orden; no modifique weights.
    - Rechace pesos negativos, no finitos, diccionarios vacíos y suma cero.
    """
    # Versión inicial del equipo. Se reemplazó por la versión mejorada con ayuda de IA (abajo).
    # total = sum(weights.values())
    # return {outcome: weight / total for outcome, weight in weights.items()}
    # Versión mejorada con ayuda de IA (activa).
    if not weights:
        raise ValueError("La distribución no puede estar vacía")
    for weight in weights.values():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("Los pesos deben ser numéricos")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Los pesos deben ser finitos y no negativos")
    total = math.fsum(weights.values())
    if total == 0:
        raise ValueError("La suma de los pesos es cero")
    return {outcome: weight / total for outcome, weight in weights.items()}
    # FIN DEL BLOQUE DEL ESTUDIANTE


def expected_utility(distribution: dict, utilities: dict) -> float:
    """Calcula sum_s P(s) U(s) para una acción.

    Entradas: distribution = {resultado: probabilidad} y
    utilities = {resultado: utilidad} para una sola acción.
    Salida: un número con la utilidad esperada.

    Tips:
    - distribution ya debe estar normalizada; no normalice utilidades.
    - Ambas tablas deben contener los mismos resultados posibles.
    """
    # Validación de las entradas; no modificar.
    if set(distribution) != set(utilities) or not distribution:
        raise ValueError("Probabilidades y utilidades deben tener las mismas claves")
    if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
        raise ValueError("Distribución inválida")
    if not math.isclose(sum(distribution.values()), 1, abs_tol=1e-9):
        raise ValueError("La distribución debe sumar uno")
    if any(not math.isfinite(u) for u in utilities.values()):
        raise ValueError("Utilidades no finitas")
    # Versión inicial del equipo. Se reemplazó por la versión mejorada con ayuda de IA (abajo).
    # total = 0
    # for s in distribution:
    #     total += distribution[s] * utilities[s]
    # return total
    # Versión mejorada con ayuda de IA (activa).
    return math.fsum(distribution[s] * utilities[s] for s in distribution)
    # FIN DEL BLOQUE DEL ESTUDIANTE


def best_action(distribution: dict, actions: dict) -> tuple[str, dict]:
    """Retorna (nombre de acción, utilidades esperadas por acción).

    Entradas: distribution = {resultado: probabilidad} y
    actions = {acción: {resultado: utilidad}}.
    Salida: (acción elegida, {acción: utilidad esperada}).

    Tips:
    - actions asocia cada acción con su tabla de utilidades.
    - Reutilice expected_utility y conserve la primera acción en los empates.
    """
    # Validación de las acciones; no modificar.
    if not actions:
        raise ValueError("Se requiere al menos una acción")
    # Versión inicial del equipo. Se reemplazó por la versión mejorada con ayuda de IA (abajo).
    # values = {a: expected_utility(distribution, u) for a, u in actions.items()}
    # return max(values, key=values.get), values
    # Versión mejorada con ayuda de IA (activa).
    values = {}
    chosen = None
    for action, utilities in actions.items():
        values[action] = expected_utility(distribution, utilities)
        # Solo una mejora estricta reemplaza la acción elegida: en un empate
        # (con margen para el redondeo) se conserva la primera de la tabla.
        if chosen is None or values[action] > values[chosen] + 1e-12:
            chosen = action
    return chosen, values
    # FIN DEL BLOQUE DEL ESTUDIANTE
