from dataclasses import dataclass
import time
from typing import Dict, List
import requests
from nucleus.constants import (
    JOB_CREATION_TIME_KEY,
    JOB_ID_KEY,
    JOB_LAST_KNOWN_STATUS_KEY,
    JOB_TYPE_KEY,
    STATUS_KEY,
)

JOB_POLLING_INTERVAL = 5


@dataclass
class AsyncJob:
    job_id: str
    job_last_known_status: str
    job_type: str
    job_creation_time: str
    client: "NucleusClient"  # type: ignore # noqa: F821

    def status(self) -> Dict[str, str]:
        response = self.client.make_request(
            payload={},
            route=f"job/{self.job_id}",
            requests_command=requests.get,
        )
        self.job_last_known_status = response[STATUS_KEY]
        return response

    def errors(self) -> List[str]:
        return self.client.make_request(
            payload={},
            route=f"job/{self.job_id}/errors",
            requests_command=requests.get,
        )

    def sleep_until_complete(self, verbose_std_out=True):
        while 1:
            status = self.status()
            time.sleep(JOB_POLLING_INTERVAL)

            if verbose_std_out:
                print(f"Status at {time.ctime()}: {status}")
            if status["status"] == "Running":
                continue

            break

        final_status = status
        if final_status["status"] == "Errored":
            raise JobError(final_status, self)

    @classmethod
    def from_json(cls, payload: dict, client):
        return cls(
            job_id=payload[JOB_ID_KEY],
            job_last_known_status=payload[JOB_LAST_KNOWN_STATUS_KEY],
            job_type=payload[JOB_TYPE_KEY],
            job_creation_time=payload[JOB_CREATION_TIME_KEY],
            client=client,
        )


class JobError(Exception):
    def __init__(self, job_status: Dict[str, str], job: AsyncJob):
        final_status_message = job_status["message"]
        final_status = job_status["status"]
        message = (
            f"The job reported a final status of {final_status} "
            "This could, however, mean a partial success with some successes and some failures. "
            f"The final status message was: {final_status_message} \n"
            f"For more detailed error messages you can call {str(job)}.errors()"
        )
        super().__init__(message)
