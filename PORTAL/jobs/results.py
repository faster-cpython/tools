


class Result(Metadata):

    FIELDS = [
        'reqid',
        'reqdir',
        'status',
        'history',
    ]

    STATUS = types.SimpleNamespace(
        CREATED='created',
        PENDING='pending',
        ACTIVE='active',
        RUNNING='running',
        SUCCESS='success',
        FAILED='failed',
        CANCELED='canceled',
    )
    _STATUS_BY_VALUE = {v: v for _, v in vars(STATUS).items()}
    _STATUS_BY_VALUE['cancelled'] = STATUS.CANCELED
    FINISHED = frozenset([
        STATUS.SUCCESS,
        STATUS.FAILED,
        STATUS.CANCELED,
    ])
    CLOSED = 'closed'

    @classmethod
    def resolve_status(cls, status):
        try:
            return cls._STATUS_BY_VALUE[status]
        except KeyError:
            raise ValueError(f'unsupported status {status!r}')

    @classmethod
    def read_status(cls, metafile, *, fail=True):
        missing = None
        text = _read_file(metafile, fail=fail)
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
            raise MissingMetadataError(source=metafile)
        else:
            raise InvalidMetadataError(source=metafile)

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
                    date, _ = get_utc_datetime(date)
                elif isinstance(date, int):
                    date, _ = get_utc_datetime(date)
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
    def short(self):
        if not self.status:
            return f'<{self.reqid}: (created)>'
        return f'<{self.reqid}: {self.status}>'

    @property
    def request(self):
        try:
            return self._request
        except AttributeError:
            self._request = Request(self.reqid, self.reqdir)
            return self._request

    def set_status(self, status):
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

    def close(self):
        if self.history[-1][0] is self.CLOSED:
            # XXX Fail?
            return
        self.history.append(
            (self.CLOSED, datetime.datetime.now(datetime.timezone.utc)),
        )

    def as_jsonable(self, *, withextra=False):
        data = super().as_jsonable(withextra=withextra)
        if self.status is None:
            data['status'] = self.STATUS.CREATED
        data['reqid'] = str(data['reqid'])
        data['history'] = [(st, d.isoformat() if d else None)
                           for st, d in data['history']]
        return data
