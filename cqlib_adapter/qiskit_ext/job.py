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

import time
from collections import defaultdict

import numpy as np
from qiskit.providers import JobV1, JobStatus
from qiskit.result import Result

from qiskit.providers.backend import Backend

from cqlib.utils.laboratory_utils import LaboratoryUtils
from .api_client import ApiClient


class TianYanJob(JobV1):

    def __init__(
            self,
            job_id: str,
            backend: Backend | None = None,
            api_client: ApiClient | None = None,
            **kwargs
    ) -> None:
        super().__init__(backend=backend, job_id=job_id, **kwargs)
        if api_client is None:
            api_client = ApiClient()
        self._api_client = api_client

    def submit(self):
        """Submit the job to the backend for execution."""
        pass

    def result(self) -> Result:
        """Return the results of the job."""
        task_ids = self._job_id.split(',')
        while 1:
            status = self.status()
            if status == JobStatus.DONE:
                break
            time.sleep(1)

        lu = LaboratoryUtils()
        results = []
        for item in self._api_client.query_job(task_ids):
            readout_calibration = self.metadata.get('readout_calibration', True)
            if 'shots' in  self.metadata:
                shots = self.metadata['shots']
            else:
                shots = len(item['resultStatus']) - 1
            if self._backend.simulator or not readout_calibration:
                basis_list = lu.readout_data_to_state_probabilities(item)
                counts, memory_list = self.to_counts(basis_list)
            else:
                calibration_result = lu.probability_calibration(
                    result=item, laboratory=None, config_json=self._backend.machine_config
                )
                counts, memory_list = self.probability_correction(calibration_result, shots)

            results.append({
                'shots': shots,
                'success': True,
                'data': {
                    'counts': counts,
                    'memory': memory_list
                }
            })

        return Result.from_dict({
            "backend_name": self._backend.configuration.backend_name,
            "backend_version": self._backend.version,
            "qobj_id": id(self._job_id),
            "job_id": self._job_id,
            "success": True,
            'status': JobStatus.DONE,
            "results": results,
        })

    def cancel(self):
        """Attempt to cancel the job."""
        raise NotImplementedError

    def status(self) -> JobStatus:
        """Return the status of the job, among the values of ``JobStatus``."""
        task_ids = self._job_id.split(',')
        # todo: use new light api
        data = self._api_client.query_job(task_ids)
        ld = len(data)

        if ld == len(task_ids):
            return JobStatus.DONE
        if ld > 0:
            return JobStatus.RUNNING
        return JobStatus.QUEUED

    @staticmethod
    def to_counts(state01):
        """Calculates the probability distribution of measurement outcomes for a given quantum state.

        Args:
            state01 (list): A one-dimensional or two-dimensional list representing the quantum state, with complex numbers as elements.

        Returns:
            dict: A dictionary where keys are binary strings representing measurement outcomes and values are their corresponding probabilities.
        """
        if isinstance(state01[0], bool):
            state01 = [state01]
        counts = defaultdict(int)
        state_01_t = np.transpose(state01)
        memory_list = []
        for num in range(len(state_01_t)):
            k = 0
            for i in range(len(state01)):
                k += state_01_t[num][i] * (2 ** i)
            prob_state = hex(k)
            counts[prob_state] += 1
            memory_list.append(prob_state)

        return counts, memory_list

    @staticmethod
    def probability_correction(probabilities, shots):
        """correction of the measured probability of 01 quantum state.
           If there is a probability greater than 1, change this item to 1.
           If there is anything less than 0, change the item to 0.

        Args:
            probabilities:
                corrected probability.

        Returns:
            Dict: corrected probability.
        """
        hex_keys = []
        norm_probs = []
        for k, v in probabilities.items():
            hex_keys.append(hex(int(k, 2)))
            norm_probs.append(v)

        probs = np.clip(norm_probs, 0, 1)
        probs /= probs.sum()

        exact_counts = (shots * probs).astype(np.float64)
        integer_part = exact_counts.astype(int)
        fractional_part = exact_counts - integer_part
        remaining = shots - integer_part.sum()
        if remaining > 0:
            # 按小数部分大小分配剩余计数
            indices = np.argpartition(-fractional_part, remaining)[:remaining]
            integer_part[indices] += 1
        counts_dict = {k: c for k, c in zip(hex_keys, integer_part) if c > 0}
        memory_list = np.repeat(hex_keys, integer_part)
        np.random.shuffle(memory_list)
        return counts_dict, memory_list.tolist()
