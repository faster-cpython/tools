"""The current job."""

import logging
import os
import os.path

from . import _utils, _common
from .requests import RequestID


logger = logging.getLogger(__name__)


class StagedRequestError(Exception):

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class StagedRequestDirError(StagedRequestError, _common.RequestDirError):

    def __init__(self, reqid, reqdir, reason, msg):
        _common.RequestDirError.__init__(self, reqid, reqdir, reason, msg)


class StagedRequestStatusError(StagedRequestError):

    reason = None

    def __init__(self, reqid, status):
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


def get_staged_request(pfiles):
    try:
        curid = _read_staged(pfiles)
    except StagedRequestError as exc:
        _clear_staged(pfiles, exc)
        return None
    if curid:
        reqfs = pfiles.resolve_request(curid)
        try:
            _check_staged_request(curid, reqfs)
        except StagedRequestError as exc:
            _clear_staged(pfiles, exc)
            curid = None
    return curid


def stage_request(reqid, pfiles):
    jobfs = pfiles.resolve_request(reqid)
    status = Result.read_status(jobfs.result.metadata, fail=False)
    if status is not Result.STATUS.PENDING:
        raise RequestNotPendingError(reqid, status)
    _set_staged(reqid, jobfs.request, pfiles)


def unstage_request(reqid, pfiles):
    reqid = RequestID.from_raw(reqid)
    try:
        curid = _read_staged(pfiles)
    except StagedRequestError as exc:
        # One was already set but the link is invalid.
        _clear_staged(pfiles, exc)
        raise RequestNotStagedError(reqid)
    else:
        if curid == reqid:
            # It's a match!
            _clear_staged(pfiles)
        else:
            if curid:
                # Clear it if it is no longer valid.
                reqfs = pfiles.resolve_request(curid)
                try:
                    _check_staged_request(curid, reqfs)
                except StagedRequestError as exc:
                    _clear_staged(pfiles, exc)
            raise RequestNotStagedError(reqid)


def _read_staged(pfiles):
    link = pfiles.requests.current
    try:
        reqdir = os.readlink(link)
    except FileNotFoundError:
        return None
    except OSError:
        if os.path.islink(link):
            raise  # re-raise
        exc = _common.RequestDirError(None, link, 'malformed', 'target is not a link')
        try:
            exc.reqid = _common.check_reqdir(link, pfiles, StagedRequestDirError)
        except StagedRequestDirError:
            raise exc
        raise exc
    else:
        return _common.check_reqdir(reqdir, pfiles, StagedRequestDirError)


def _check_staged_request(reqid, reqfs):
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


def _set_staged(reqid, reqdir, pfiles):
    try:
        os.symlink(reqdir, pfiles.requests.current)
    except FileExistsError:
        try:
            curid = _read_staged(pfiles)
        except StagedRequestError as exc:
            # One was already set but the link is invalid.
            _clear_staged(pfiles, exc)
        else:
            if curid == reqid:
                # XXX Fail?
                logger.warn(f'{reqid} is already set as the current job')
                return
            elif curid:
                # Clear it if it is no longer valid.
                reqfs = pfiles.resolve_request(curid)
                try:
                    _check_staged_request(curid, reqfs)
                except StagedRequestError as exc:
                    _clear_staged(pfiles, exc)
                else:
                    raise RequestAlreadyStagedError(reqid, curid)
        logger.info('trying again')
        # XXX Guard against infinite recursion?
        return _set_staged(reqid, reqdir, pfiles)


def _clear_staged(pfiles, exc=None):
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
    os.unlink(pfiles.requests.current)
