"""The current job."""

import logging
import os
import os.path
from typing import Optional

from . import _utils, _common
from .requests import RequestID, Result


logger = logging.getLogger(__name__)


class StagedRequestError(Exception):

    def __init__(self, reqid: Optional[RequestID], msg: str):
        super().__init__(msg)
        self.reqid = reqid


class StagedRequestDirError(StagedRequestError, _common.RequestDirError):

    def __init__(
            self,
            reqid: Optional[RequestID],
            reqdir: Optional[str],
            reason: str,
            msg: str
    ):
        _common.RequestDirError.__init__(self, reqid, reqdir, reason, msg)


class StagedRequestStatusError(StagedRequestError):

    reason: Optional[str] = None

    def __init__(self, reqid: RequestID, status: Optional[str]):
        assert self.reason
        super().__init__(reqid, self.reason)
        self.status = status


class OutdatedStagedRequestError(StagedRequestStatusError):
    pass


class StagedRequestNotRunningError(OutdatedStagedRequestError):
    reason = 'is no longer running'


class StagedRequestAlreadyFinishedError(OutdatedStagedRequestError):
    reason = 'is still "current" even though it finished'


class StagedRequestUnexpectedStatusError(StagedRequestStatusError):
    reason = 'was set as the current job incorrectly'


class StagedRequestInvalidMetadataError(StagedRequestStatusError):
    reason = 'has invalid metadata'


class StagedRequestMissingStatusError(StagedRequestInvalidMetadataError):
    reason = 'is missing status metadata'


class StagingRequestError(Exception):

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class RequestNotPendingError(StagingRequestError):

    def __init__(self, reqid, status=None):
        super().__init__(reqid, f'could not stage {reqid} (expected pending, got {status or "???"} status)')
        self.status = status


class RequestAlreadyStagedError(StagingRequestError):

    def __init__(self, reqid, curid):
        super().__init__(reqid, f'could not stage {reqid} ({curid} already staged)')
        self.curid = curid


class UnstagingRequestError(Exception):

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class RequestNotStagedError(UnstagingRequestError):

    def __init__(self, reqid, curid=None):
        msg = f'{reqid} is not currently staged'
        if curid:
            msg = f'{msg} ({curid} is)'
        super().__init__(reqid, msg)
        self.curid = curid


def symlink_from_jobsfs(jobsfs: _common.JobsFS) -> str:
    jobsfs = _common.JobsFS.from_raw(jobsfs)
    try:
        return jobsfs.requests.current
    except AttributeError:
        return f'{jobsfs.requests}/CURRENT'


def get_staged_request(
        jobsfs: _common.JobsFS,
        symlink: Optional[str] = None
) -> Optional[RequestID]:
    if not symlink:
        symlink = symlink_from_jobsfs(jobsfs)
    try:
        curid = _read_staged(symlink, jobsfs)
    except StagedRequestError as exc:
        _clear_staged(symlink, exc)
        return None
    if curid:
        jobfs = jobsfs.resolve_request(curid)
        try:
            _check_staged_request(curid, jobfs)
        except StagedRequestError as exc:
            _clear_staged(symlink, exc)
            curid = None
    return curid


def stage_request(
        reqid: RequestID,
        jobsfs: _common.JobsFS,
        symlink: Optional[str] = None
) -> None:
    if not symlink:
        symlink = symlink_from_jobsfs(jobsfs)
    jobfs = jobsfs.resolve_request(reqid)
    status = Result.read_status(jobfs.result.metadata, fail=False)
    if status is not Result.STATUS.PENDING:
        raise RequestNotPendingError(reqid, status)
    _set_staged(reqid, jobfs.request, symlink, jobsfs)


def unstage_request(
        reqid: RequestID,
        jobsfs: _common.JobsFS,
        symlink: Optional[str] = None
) -> None:
    if not symlink:
        symlink = symlink_from_jobsfs(jobsfs)
    reqid = RequestID.from_raw(reqid)
    try:
        curid = _read_staged(symlink, jobsfs)
    except StagedRequestError as exc:
        # One was already set but the link is invalid.
        _clear_staged(symlink, exc)
        raise RequestNotStagedError(reqid)
    else:
        if curid == reqid:
            # It's a match!
            _clear_staged(symlink)
        else:
            if curid:
                # Clear it if it is no longer valid.
                jobfs = jobsfs.resolve_request(curid)
                try:
                    _check_staged_request(curid, jobfs)
                except StagedRequestError as exc:
                    _clear_staged(symlink, exc)
            raise RequestNotStagedError(reqid)


def _read_staged(
        symlink: str,
        jobsfs: _common.JobsFS
) -> Optional[RequestID]:
    try:
        reqdir = os.readlink(symlink)
    except FileNotFoundError:
        return None
    except OSError:
        if os.path.islink(symlink):
            raise  # re-raise
        exc = _common.RequestDirError(None, symlink, 'malformed', 'target is not a link')
        try:
            exc.reqid = _common.check_reqdir(symlink, jobsfs, StagedRequestDirError)
        except StagedRequestDirError:
            raise exc
        raise exc
    else:
        return _common.check_reqdir(reqdir, jobsfs, StagedRequestDirError)


def _check_staged_request(
        reqid: RequestID,
        reqfs: _common.JobFS
) -> None:
    # Check the request status.
    try:
        status = Result.read_status(str(reqfs.result.metadata))
    except _utils.MissingMetadataError:
        raise StagedRequestMissingStatusError(reqid, None)
    except _utils.InvalidMetadataError:
        raise StagedRequestInvalidMetadataError(reqid, None)
    else:
        if status in Result.ACTIVE and status != 'pending':
            if not _utils.PIDFile(str(reqfs.pidfile)).read(orphaned='ignore'):
                raise StagedRequestNotRunningError(reqid, status)
        elif status in Result.FINISHED:
            raise StagedRequestAlreadyFinishedError(reqid, status)
        else:  # created, pending
            raise StagedRequestUnexpectedStatusError(reqid, status)
    # XXX Do other checks?


def _set_staged(
        reqid: RequestID,
        reqdir: str,
        symlink: str,
        jobsfs: _common.JobsFS
) -> None:
    try:
        os.symlink(reqdir, symlink)
    except FileExistsError:
        try:
            curid = _read_staged(symlink, jobsfs)
        except StagedRequestError as exc:
            # One was already set but the link is invalid.
            _clear_staged(symlink, exc)
        else:
            if curid == reqid:
                # XXX Fail?
                logger.warning(f'{reqid} is already set as the current job')
                return
            elif curid:
                # Clear it if it is no longer valid.
                jobfs = jobsfs.resolve_request(curid)
                try:
                    _check_staged_request(curid, jobfs)
                except StagedRequestError as exc:
                    _clear_staged(symlink, exc)
                else:
                    raise RequestAlreadyStagedError(reqid, curid)
        logger.info('trying again')
        # XXX Guard against infinite recursion?
        return _set_staged(reqid, reqdir, symlink, jobsfs)


def _clear_staged(symlink: str, exc: Optional[Exception] = None) -> None:
    if exc is not None:
        if isinstance(exc, StagedRequestInvalidMetadataError):
            log = logger.error
        else:
            log = logger.warn
        reqid = getattr(exc, 'reqid', None)
        reason = getattr(exc, 'reason', None) or 'broken'
        if not reason.startswith(('is', 'was', 'has')):
            reason = f'is {reason}'
        log(f'request {reqid or "???"} {reason} ({exc}); unsetting as the current job...')
    os.unlink(symlink)
