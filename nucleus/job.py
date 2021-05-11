from dataclasses import dataclass
import time
from typing import Dict, List

import requests


@dataclass
class AsyncJob:
    id: str
    client: "NucleusClient"  # type: ignore # noqa: F821

    def status(self) -> Dict[str, str]:
        return self.client.make_request(
            payload={},
            route=f"job/{self.id}",
            requests_command=requests.get,
        )

    def errors(self) -> List[str]:
        return self.client.make_request(
            payload={},
            route=f"job/{self.id}/errors",
            requests_command=requests.get,
        )

    def sleep_until_complete(self, verbose_std_out=True):
        while 1:
            time.sleep(1)
            status = self.status()
            if verbose_std_out:
                print(status)
            if status["status"] == "Running":
                continue
            break

        final_status = status
        if final_status["status"] == "Errored":
            raise JobError(final_status, self)


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
