# This code is part of cqlib.
#
# Copyright (C) 2025 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Sampler module for executing quantum circuits on the TianYan backend.

This module provides a class to sample quantum circuits using the TianYan backend,
leveraging Qiskit's primitives for efficient execution.
"""
from collections import defaultdict
from typing import Iterable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BackendSamplerV2
from qiskit.primitives.backend_sampler_v2 import _prepare_memory, _analyze_circuit
from qiskit.primitives.containers import (
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.result import Result

from .tianyan_backend import TianYanBackend


class TianYanSampler(BackendSamplerV2):
    """A sampler class for executing quantum circuits on the TianYan backend.

    This class extends Qiskit's `BackendSamplerV2` to provide sampling functionality
    specifically for the TianYan backend.
    """

    def __init__(
            self,
            backend: TianYanBackend,
            default_shots: int = 1024,
            options: dict | None = None
    ) -> None:
        """Initializes the TianYanSampler instance.

        Args:
            backend (TianYanBackend): The TianYan backend to use for sampling.
            options (dict, optional): Additional options for the sampler. Defaults to None.
        """
        super().__init__(backend=backend, options=options)
        self._backend = backend
        self._default_shots = default_shots

    def run(
            self,
            pubs: Iterable[SamplerPubLike],
            *,
            shots: int | None = None,
            readout_calibration: bool = True,
            auto_transpile: bool = True,
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self._options.default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(
            self._run,
            coerced_pubs,
            readout_calibration=readout_calibration,
            auto_transpile=auto_transpile
        )
        # pylint: disable=protected-access
        job._submit()
        return job

    def _run(
            self,
            pubs: list[SamplerPub],
            readout_calibration: bool = True,
            auto_transpile: bool = True,
    ) -> PrimitiveResult[SamplerPubResult]:
        pub_dict = defaultdict(list)
        # consolidate pubs with the same number of shots
        for i, pub in enumerate(pubs):
            pub_dict[pub.shots].append(i)

        results = [None] * len(pubs)
        for shots, lst in pub_dict.items():
            # run pubs with the same number of shots at once
            pub_results = self._run_pubs(
                pubs=[pubs[i] for i in lst],
                shots=shots,
                readout_calibration=readout_calibration,
                auto_transpile=auto_transpile
            )
            # reconstruct the result of pubs
            for i, pub_result in zip(lst, pub_results):
                results[i] = pub_result
        return PrimitiveResult(results, metadata={"version": 2})

    # pylint: disable=too-many-locals
    def _run_pubs(
            self,
            pubs: list[SamplerPub],
            shots: int,
            readout_calibration: bool = True,
            auto_transpile: bool = True,
    ) -> list[SamplerPubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        # prepare circuits
        bound_circuits = [pub.parameter_values.bind_all(pub.circuit) for pub in pubs]
        flatten_circuits = []
        for circuits in bound_circuits:
            flatten_circuits.extend(np.ravel(circuits).tolist())

        run_opts = dict(self._options.run_options or {}) | {
            'readout_calibration': readout_calibration,
            'auto_transpile': auto_transpile
        }
        # run circuits
        results, _ = _run_circuits(
            flatten_circuits,
            self._backend,
            clear_metadata=False,
            memory=True,
            shots=shots,
            seed_simulator=self._options.seed_simulator,
            **run_opts,
        )
        result_memory = _prepare_memory(results)

        # pack memory to an ndarray of uint8
        results = []
        start = 0
        meas_level = (
            None
            if self._options.run_options is None
            else self._options.run_options.get("meas_level")
        )
        for pub, bound in zip(pubs, bound_circuits):
            meas_info, max_num_bytes = _analyze_circuit(pub.circuit)
            end = start + bound.size
            results.append(
                self._postprocess_pub(
                    result_memory[start:end],
                    shots,
                    bound.shape,
                    meas_info,
                    max_num_bytes,
                    pub.circuit.metadata,
                    meas_level,
                )
            )
            start = end

        return results


def _run_circuits(
        circuits: QuantumCircuit | list[QuantumCircuit],
        backend: TianYanBackend,
        clear_metadata: bool = True,
        **run_options,
) -> tuple[list[Result], list[dict]]:
    """Remove metadata of circuits and run the circuits on a backend.
    Args:
        circuits: The circuits
        backend: The backend
        clear_metadata: Clear circuit metadata before passing to backend.run if
            True.
        **run_options: run_options
    Returns:
        The result and the metadata of the circuits
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    metadata = []
    for circ in circuits:
        metadata.append(circ.metadata)
        if clear_metadata:
            circ.metadata = {}
    if isinstance(backend, TianYanBackend):
        max_circuits = backend.max_circuits
    else:
        raise RuntimeError("Backend version not supported")
    if max_circuits:
        jobs = [
            backend.run(circuits[pos: pos + max_circuits], **run_options)
            for pos in range(0, len(circuits), max_circuits)
        ]
        result = [x.result() for x in jobs]
    else:
        result = [backend.run(circuits, **run_options).result()]
    return result, metadata
