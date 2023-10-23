import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set

import requests

from nucleus.constants import (
    JOB_CREATION_TIME_KEY,
    JOB_ID_KEY,
    JOB_LAST_KNOWN_STATUS_KEY,
    JOB_TYPE_KEY,
    STATUS_KEY,
)
from nucleus.utils import replace_double_slashes

JOB_POLLING_INTERVAL = 5


class JobStatus(str, Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    ERRORED_DEPRECATED = "Errored"
    ERRORED_SERVER = "Errored_Server"  # Server Error
    ERRORED_USER = "Errored_User"  # User Error
    ERRORED_PARTIAL = "Errored_Partial"  # Partially Completed
    ERRORED_HANGING = "Errored_Hanging"  # Hanging
    CANCELLED = "Cancelled"
    RETRIED = "Retried"


JOB_ERROR_PREFIX = JobStatus.ERRORED_DEPRECATED
JOB_ERROR_STATES: Set[JobStatus] = {
    JobStatus.ERRORED_DEPRECATED,
    JobStatus.ERRORED_SERVER,
    JobStatus.ERRORED_USER,
    JobStatus.ERRORED_PARTIAL,
    JobStatus.ERRORED_HANGING,
}


@dataclass
class AsyncJob:
    """Object used to check the status or errors of a long running asynchronous operation.

    ::

        import nucleus

        client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
        dataset = client.get_dataset("ds_bwkezj6g5c4g05gqp1eg")

        # When kicking off an asynchronous job, store the return value as a variable
        job = dataset.append(items=YOUR_DATASET_ITEMS, asynchronous=True)

        # Poll for status or errors
        print(job.status())
        print(job.errors())

        # Block until job finishes
        job.sleep_until_complete()
    """

    job_id: str
    job_last_known_status: str
    job_type: str
    job_creation_time: str
    client: "NucleusClient"  # type: ignore # noqa: F821

    def status(self) -> Dict[str, str]:
        """Fetches status of the job and an informative message on job progress.

        Returns:
            A dict of the job ID, status (one of Running, Completed, or Errored),
            an informative message on the job progress, and number of both completed
            and total steps.
            ::

                {
                    "job_id": "job_c19xcf9mkws46gah0000",
                    "status": "Completed",
                    "message": "Job completed successfully.",
                    "job_progress": "0.33",
                    "completed_steps": "1",
                    "total_steps:": "3",
                }
        """
        response = self.client.make_request(
            payload={},
            route=f"job/{self.job_id}",
            requests_command=requests.get,
        )
        self.job_last_known_status = response[STATUS_KEY]
        return response

    def errors(self) -> List[str]:
        """Fetches a list of the latest errors generated by the asynchronous job.

        Useful for debugging failed or partially successful jobs.

        Returns:
            A list of strings containing the 10,000 most recently generated errors.
            ::

                [
                    '{"annotation":{"label":"car","type":"box","geometry":{"x":50,"y":60,"width":70,"height":80},"referenceId":"bad_ref_id","annotationId":"attempted_annot_upload","metadata":{}},"error":"Item with id bad_ref_id does not exist."}'
                ]
        """
        errors = self.client.make_request(
            payload={},
            route=f"job/{self.job_id}/errors",
            requests_command=requests.get,
        )
        return [replace_double_slashes(error) for error in errors]

    def sleep_until_complete(self, verbose_std_out=True):
        """Blocks until the job completes or errors.

        Parameters:
            verbose_std_out (Optional[bool]): Whether or not to verbosely log while
              sleeping. Defaults to True.
        """
        start_time = time.perf_counter()
        while 1:
            status = self.status()
            time.sleep(JOB_POLLING_INTERVAL)

            if verbose_std_out:
                print(
                    f"Status at {time.perf_counter() - start_time} s: {status}"
                )
            if status["status"] == "Running":
                continue

            break

        if verbose_std_out:
            print(
                f"Finished at {time.perf_counter() - start_time} s: {status}"
            )
        final_status = status
        if final_status["status"] in JOB_ERROR_STATES or final_status[
            "status"
        ].startswith(JOB_ERROR_PREFIX):
            raise JobError(final_status, self)

    @classmethod
    def from_id(cls, job_id: str, client: "NucleusClient"):  # type: ignore # noqa: F821
        """Creates a job instance from a specific job Id.

        Parameters:
            job_id: Defines the job Id
            client: The client to use for the request.

        Returns:
            The specific AsyncMethod (or inherited) instance.
        """
        job = client.get_job(job_id)
        return cls.from_json(job.__dict__, client)

    @classmethod
    def from_json(cls, payload: dict, client):
        # TODO: make private
        return cls(
            job_id=payload[JOB_ID_KEY],
            job_last_known_status=payload[JOB_LAST_KNOWN_STATUS_KEY],
            job_type=payload[JOB_TYPE_KEY],
            job_creation_time=payload[JOB_CREATION_TIME_KEY],
            client=client,
        )


class EmbeddingsExportJob(AsyncJob):
    def result_urls(self, wait_for_completion=True) -> List[str]:
        """Gets a list of signed Scale URLs for each embedding batch.

        Parameters:
            wait_for_completion: Defines whether the call shall wait for
                the job to complete. Defaults to True

        Returns:
            A list of signed Scale URLs which contain batches of embeddings.

            The files contain a JSON array of embedding records with the following schema:
                [{
                    "reference_id": str,
                    "embedding_vector": List[float]
                }]
        """
        if wait_for_completion:
            self.sleep_until_complete(verbose_std_out=False)

        status = self.status()

        if status["status"] != "Completed":
            raise JobError(status, self)

        return status["message"]["result"]  # type: ignore


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
        message = replace_double_slashes(message)
        super().__init__(message)
