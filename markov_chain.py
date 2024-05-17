# markov_chain.py

import random
from typing import Any, Iterable, Sequence, Self

import numpy as np

class MarkovChain[S, O]:

    def __init__(
            self,
            transitions: np.ndarray = None,
            states: dict[S, int] = None,
            index: dict[int, S] = None
    ) -> None:

        self.transitions = transitions or np.array([])
        self._states = states or dict()
        self._index = index or dict()

    def __bool__(self) -> bool:

        return self.built

    def prepare(
            self,
            observations: Sequence[O],
            states: Iterable[S] = None,
            add: bool = False
    ) -> None:

        if not add:
            self.clear()

            unique = list(set(observations))

        else:
            unique = set(self._states.keys())
            unique = [*unique, *set(observations).difference(unique)]

        self._states.update(
            {state: i for i, state in enumerate(states or unique)}
        )
        self._index.update(
            {index: state for state, index in self._states.items()}
        )

        self.transitions = np.zeros((len(self._states), len(self._states)))

    def load(
            self,
            transitions: dict[S, dict[S, float]],
            states: Iterable[S] = None,
            add: bool = False,
            weighted: bool = True
    ) -> None:

        self.prepare(transitions, states=states, add=add)

        for key, states in transitions.items():
            for state, weight in states.items():
                self.transitions[self._states[key]][self._states[state]] = (
                    weight * (len(state) if weighted else 1)
                )

    @property
    def built(self) -> bool:

        return self.transitions.size > 0

    @property
    def states(self) -> list[O]:

        return list(self._states.keys())

    def fit(
            self,
            observations: Sequence[O],
            states: Iterable[S] = None,
            add: bool = False
    ) -> None:

        if not observations:
            return

        self.prepare(observations=observations, states=states, add=add)

        for i in range(1, len(observations)):
            self.transitions[
                self._states[observations[i - 1]]
            ][self._states[observations[i]]] += 1

    def forward(self, start: S, length: int, adjust: bool) -> list[S]:

        previous = self._states[start]

        states: list[Any] = list()

        choices = list(range(len(self._states)))

        for i in range(length):
            probabilities = self.transitions[previous]

            try:
                current = random.choices(choices, weights=probabilities, k=1)[0]

            except ValueError as e:
                if adjust:
                    return states

                else:
                    raise e

            states.append(self._index[current])

            previous = current

        return states

    def clear(self) -> None:

        self.transitions = np.array([])
        self._states.clear()
        self._index.clear()

    def copy(self) -> Self:

        chain = MarkovChain()

        chain.transitions = self.transitions.copy()
        chain._states = self._states.copy()
        chain._index = self._index.copy()

        return chain
