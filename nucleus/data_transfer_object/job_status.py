from typing import List, Optional

from dateutil.parser import parse
from pydantic import validator

from nucleus.job import CustomerJobTypes
from nucleus.pydantic_base import DictCompatibleModel

JOB_REQ_LIMIT = 50_000


class JobInfoRequestPayload(DictCompatibleModel):
    dataset_id: Optional[str]
    job_types: Optional[List[CustomerJobTypes]]
    from_date: Optional[str]
    to_date: Optional[str]
    limit: Optional[int]
    show_completed: bool

    @validator("from_date", "to_date")
    def ensure_date_format(cls, date):
        if date is None:
            return
        try:
            parse(date)
        except:
            raise ValueError(
                f"Field: {date} not a valid date. Try using YYYY-MM-DD format."
            )
        return date

    @validator("limit")
    def ensure_limit(cls, limit):
        if limit is None:
            return JOB_REQ_LIMIT
        if limit > JOB_REQ_LIMIT:
            raise ValueError(f"Max request limit is 50,000, but got: {limit}.")
        return limit

    @validator("job_types")
    def ensure_job_type(cls, job_types):
        if job_types is None:
            return []
        try:
            assert all(
                [job_type in CustomerJobTypes for job_type in job_types]
            )
        except:
            raise ValueError(
                f"Job types must be one of: {CustomerJobTypes.options()}"
            )

        return job_types
