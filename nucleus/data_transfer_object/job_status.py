# pylint: disable=E0213

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

from dateutil.parser import ParserError, parse

if TYPE_CHECKING:
    # Backwards compatibility is even uglier with mypy
    from pydantic.v1 import validator
else:
    try:
        # NOTE: we always use pydantic v1 but have to do these shenanigans to support both v1 and v2
        from pydantic.v1 import validator
    except ImportError:
        from pydantic import validator

from nucleus.constants import JOB_REQ_LIMIT
from nucleus.job import CustomerJobTypes
from nucleus.pydantic_base import ImmutableModel


class JobInfoRequestPayload(ImmutableModel):
    dataset_id: Optional[str]
    job_types: Optional[List[CustomerJobTypes]]
    from_date: Optional[Union[str, datetime]]
    to_date: Optional[Union[str, datetime]]
    limit: Optional[int]
    show_completed: bool

    @validator(  # pylint: disable=used-before-assignment
        "from_date", "to_date"
    )
    def ensure_date_format(cls, date):
        if date is None:
            return None
        if isinstance(date, datetime):
            return str(date)
        try:
            parse(date)
        except ParserError as err:
            raise ValueError(
                f"Date {date} not a valid date. Try using YYYY-MM-DD format."
            ) from err
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
            assert all(t in CustomerJobTypes for t in job_types)
        except AssertionError as badType:
            raise ValueError(
                f"Job types must be one of: {CustomerJobTypes.options()}"
            ) from badType

        return [t.value for t in job_types]
