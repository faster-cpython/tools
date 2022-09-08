from collections import namedtuple
import datetime
import json
import re
import types
from typing import Any, Callable, Optional, Tuple, Union, TYPE_CHECKING

from . import _utils


if TYPE_CHECKING:
    from . import _common


ToRequestIDType = Union[str, "RequestID"]


class RequestID(namedtuple('RequestID', 'kind timestamp user')):

    KIND = types.SimpleNamespace(
        BENCHMARKS='compile-bench',
    )
    _KIND_BY_VALUE = {v: v for _, v in vars(KIND).items()}

    @classmethod
    def from_raw(cls, raw: Any):
        if isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls.parse(raw)
        elif not raw:
            raise NotImplementedError(raw)
        else:
            try:
                args = tuple(raw)
            except TypeError:
                raise NotImplementedError(repr(raw))
            return cls(*args)

    @classmethod
    def parse(cls, idstr: str):
        kinds = '|'.join(cls._KIND_BY_VALUE)
        m = re.match(rf'^req-(?:({kinds})-)?(\d{{10}})-(\w+)$', idstr)
        if not m:
            return None
        kind, timestamp, user = m.groups()
        return cls(kind, int(timestamp), user)

    @classmethod
    def generate(
            cls,
            cfg: int,
            user: Optional[str] = None,
            kind: str = KIND.BENCHMARKS
    ) -> "RequestID":
        user = _utils.resolve_user(cfg, user)
        timestamp = int(_utils.utcnow())
        return cls(kind, timestamp, user)

    def __new__(cls, kind, timestamp, user):
        if not kind:
            kind = cls.KIND.BENCHMARKS
        else:
            try:
                kind = cls._KIND_BY_VALUE[kind]
            except KeyError:
                raise ValueError(f'unsupported kind {kind!r}')

        if not timestamp:
            raise ValueError('missing timestamp')
        elif isinstance(timestamp, str):
            timestamp, _, _ = timestamp.partition('.')
            timestamp = int(timestamp)
        elif not isinstance(timestamp, int):
            try:
                timestamp = int(timestamp)
            except TypeError:
                raise TypeError(f'expected int timestamp, got {timestamp!r}')

        if not user:
            raise ValueError('missing user')
        elif not isinstance(user, str):
            raise TypeError(f'expected str for user, got {user!r}')
        else:
            _utils.check_name(user)

        self = super().__new__(
            cls,
            kind=kind,
            timestamp=timestamp,
            user=user,
        )
        return self

    def __str__(self):
        return f'req-{self.kind}-{self.timestamp}-{self.user}'

    @property
    def date(self) -> Optional[datetime.datetime]:
        dt, _ = _utils.get_utc_datetime(self.timestamp)
        return dt


##################################
# local metadata

class Request(_utils.Metadata):

    FIELDS = [
        'kind',
        'id',
        'datadir',
        'user',
        'date',
    ]

    @classmethod
    def read_kind(cls, metafile: str) -> Optional[str]:
        text = _utils.read_file(metafile, fail=False)
        if not text:
            return None
        data = json.loads(text)
        if not data:
            return None
        kind = data.get('kind')
        if not kind:
            return None
        try:
            return RequestID._KIND_BY_VALUE[kind]
        except KeyError:
            raise ValueError(f'unsupported kind {kind!r}')

    @classmethod
    def load(  # type: ignore[override]
            cls,
            reqfile: str,
            *,
            fs: Optional["_common.JobFS"] = None,
            **kwargs
    ) -> Optional["Request"]:
        self = super().load(reqfile, **kwargs)
        if self is None:
            return None
        if fs:
            self._fs = fs
        return self

    def __init__(
            self,
            id: ToRequestIDType,
            datadir: str,
            *,
            # These are ignored (duplicated by id):
            kind=None,
            user=None,
            date=None,
    ):
        if not id:
            raise ValueError('missing id')
        reqid = RequestID.from_raw(id)

        if not datadir:
            raise ValueError('missing datadir')
        if not isinstance(datadir, str):
            raise TypeError(f'expected dirname for datadir, got {datadir!r}')

        super().__init__(
            id=reqid,
            datadir=datadir,
        )

    def __str__(self):
        return str(self.id)

    @property
    def reqid(self) -> RequestID:
        return self.id

    @property
    def reqdir(self) -> str:
        return self.datadir

    @property
    def kind(self) -> str:
        return self.id.kind

    @property
    def user(self) -> str:
        return self.id.user

    @property
    def date(self) -> datetime.datetime:
        return self.id.date

    @property
    def fs(self) -> "_common.JobFS":
        try:
            return self._fs
        except AttributeError:
            raise NotImplementedError

    def as_jsonable(self, *, withextra: bool = False) -> dict:
        data = super().as_jsonable(withextra=withextra)
        data['id'] = str(data['id'])
        data['date'] = self.date.isoformat()
        return data


class Result(_utils.Metadata):

    FIELDS = [
        'reqid',
        'reqdir',
        'status',
        'history',
    ]

    STATUS = types.SimpleNamespace(
        CREATED='created',
        PENDING='pending',
        ACTIVATED='activated',  # AKA staged
        RUNNING='running',
        SUCCESS='success',
        FAILED='failed',
        CANCELED='canceled',
    )
    _STATUS_BY_VALUE = {v: v for _, v in vars(STATUS).items()}
    _STATUS_BY_VALUE['cancelled'] = STATUS.CANCELED
    # For backward compatibility:
    _STATUS_BY_VALUE['active'] = STATUS.ACTIVATED
    ACTIVE = frozenset([
        STATUS.PENDING,
        STATUS.ACTIVATED,
        STATUS.RUNNING,
    ])
    FINISHED = frozenset([
        STATUS.SUCCESS,
        STATUS.FAILED,
        STATUS.CANCELED,
    ])
    CLOSED = 'closed'

    _request: Request
    _fs: "_common.JobFS"
    _get_request: Callable[[str, str], Request]

    @classmethod
    def resolve_status(cls, status: str) -> str:
        try:
            return cls._STATUS_BY_VALUE[status]
        except KeyError:
            raise ValueError(f'unsupported status {status!r}')

    @classmethod
    def read_status(
            cls,
            metafile: str,
            *,
            fail: bool = True
    ) -> Optional[str]:
        missing = None
        text = _utils.read_file(metafile, fail=fail)
        if text is None:
            return None
        elif not text:
            missing = True
        elif text:
            try:
                data = json.loads(text)
            except json.decoder.JSONDecodeError:
                missing = False
            else:
                if 'status' not in data:
                    missing = True
                else:
                    status = data['status']
                    try:
                        return cls._STATUS_BY_VALUE[status]
                    except KeyError:
                        missing = False
        if not fail:
            return None
        elif missing:
            raise _utils.MissingMetadataError(source=metafile)
        else:
            raise _utils.InvalidMetadataError(source=metafile)

    @classmethod
    def load(cls, resfile, *, fs=None, request=None, **kwargs):
        self = super().load(resfile, **kwargs)
        if self is None:
            return None
        if fs:
            self._fs = fs
        if callable(request):
            self._get_request = request
        elif request:
            self._request = request
        return self

    def __init__(self, reqid, reqdir, status=STATUS.CREATED, history=None):
        if not reqid:
            raise ValueError('missing reqid')
        reqid = RequestID.from_raw(reqid)

        if not reqdir:
            raise ValueError('missing reqdir')
        if not isinstance(reqdir, str):
            raise TypeError(f'expected dirname for reqdir, got {reqdir!r}')

        if status == self.STATUS.CREATED:
            status = None
        elif status:
            status = self.resolve_status(status)
        else:
            status = None

        if history:
            h = []
            for st, date in history:
                try:
                    st = self._STATUS_BY_VALUE[st]
                except KeyError:
                    if st == self.CLOSED:
                        st = self.CLOSED
                    else:
                        raise ValueError(f'unsupported history status {st!r}')
                if not date:
                    date = None
                elif isinstance(date, str):
                    date, _ = _utils.get_utc_datetime(date)
                elif isinstance(date, int):
                    date, _ = _utils.get_utc_datetime(date)
                elif not isinstance(date, datetime.datetime):
                    raise TypeError(f'unsupported history date {date!r}')
                h.append((st, date))
            history = h
        else:
            history = [
                (self.STATUS.CREATED, reqid.date),
            ]
            if status is not None:
                for st in self._STATUS_BY_VALUE:
                    history.append((st, None))
                    if status == st:
                        break
                    if status == self.STATUS.RUNNING:
                        history.append((status, None))

        super().__init__(
            reqid=reqid,
            reqdir=reqdir,
            status=status,
            history=history,
        )

    def __str__(self):
        return str(self.reqid)

    @property
    def short(self) -> str:
        if not self.status:
            return f'<{self.reqid}: (created)>'
        return f'<{self.reqid}: {self.status}>'

    @property
    def request(self) -> Request:
        try:
            return self._request
        except AttributeError:
            get_request = getattr(self, '_get_request', Request)
            self._request = get_request(self.reqid, self.reqdir)
            return self._request

    @property
    def fs(self) -> "_common.JobFS":
        try:
            return self._fs
        except AttributeError:
            raise NotImplementedError

    @property
    def host(self) -> str:
        # XXX This will need to support other hosts.
        return 'fc_linux'

    @property
    def started(
            self
    ) -> Union[Tuple[datetime.datetime, str], Tuple[None, None]]:
        history = list(self.history)
        if not history:
            return None, None
        last_st, last_date = history[-1]
        if last_st == Result.STATUS.ACTIVATED:
            return last_date, last_st
        for st, date in reversed(history):
            if st == Result.STATUS.RUNNING:
                return date, st
        else:
            return None, None

    @property
    def finished(
            self
    ) -> Union[Tuple[datetime.datetime, str], Tuple[None, None]]:
        history = list(self.history)
        if not history:
            return None, None
        for st, date in reversed(history):
            if st in Result.FINISHED:
                return date, st
        else:
            return None, None

    def set_status(self, status: Optional[str]) -> None:
        if not status:
            raise ValueError('missing status')
        status = self.resolve_status(status)
        if self.history[-1][0] is self.CLOSED:
            raise Exception(f'req {self.reqid} is already closed')
        # XXX Make sure it is the next possible status?
        self.history.append(
            (status, datetime.datetime.now(datetime.timezone.utc)),
        )
        self.status = None if status is self.STATUS.CREATED else status
        if status in self.FINISHED:
            self.close()

    def close(self) -> None:
        if self.history[-1][0] is self.CLOSED:
            # XXX Fail?
            return
        self.history.append(
            (self.CLOSED, datetime.datetime.now(datetime.timezone.utc)),
        )

    def as_jsonable(self, *, withextra: bool = False) -> dict:
        data = super().as_jsonable(withextra=withextra)
        if self.status is None:
            data['status'] = self.STATUS.CREATED
        data['reqid'] = str(data['reqid'])
        data['history'] = [(st, d.isoformat() if d else None)
                           for st, d in data['history']]
        return data
