"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize


def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)

    # Versión inicial del equipo. Se reemplazó por la versión mejorada con ayuda de IA (abajo).
    # def enumerate_all(variables, event):
    #     if not variables:
    #         return 1.0
    #     first, rest = variables[0], variables[1:]
    #     if first in event:
    #         return network.probability(first, event[first], event) * enumerate_all(rest, event)
    #     total = 0.0
    #     for value in (0, 1):
    #         event[first] = value
    #         total += network.probability(first, value, event) * enumerate_all(rest, event)
    #     del event[first]
    #     return total
    #
    # weights = {}
    # for value in (0, 1):
    #     evidence[query] = value
    #     weights[value] = enumerate_all(network.variables, evidence)
    # del evidence[query]
    # return normalize(weights)
    # Versión mejorada con ayuda de IA (activa).
    variables = network.variables

    def enumerate_all(index, event):
        """Suma de productos desde la variable index en adelante."""
        if index == len(variables):
            return 1.0
        variable = variables[index]
        if variable in event:
            return (network.probability(variable, event[variable], event)
                    * enumerate_all(index + 1, event))
        total = 0.0
        for value in (0, 1):
            extended = {**event, variable: value}
            total += (network.probability(variable, value, extended)
                      * enumerate_all(index + 1, extended))
        return total

    weights = {}
    for value in (0, 1):
        if query in evidence and evidence[query] != value:
            weights[value] = 0.0
            continue
        weights[value] = enumerate_all(0, {**evidence, query: value})
    if weights[0] + weights[1] == 0:
        raise ValueError("La evidencia es imposible en esta red")
    return normalize(weights)
    # FIN DEL BLOQUE DEL ESTUDIANTE
