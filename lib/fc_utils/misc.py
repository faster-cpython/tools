

#######################################

ON_EXISTS_ACTIONS = [
    'warn',
    'skip',
    'replace',
    'fail',
]
ON_EXISTS_DEFAULT_ACTION = ON_EXISTS_ACTIONS[0]

RESOURCE_STATUS = [
    'valid',
    'invalid',
    'missing',
]


def resolve_on_exists(action, label=None, *,
                      customstatus=None,
                      strictstatus=False,
                      ):
    """Return a function that decides how to handle a resource collision.

    The function signature is:

      on_exists(resource, status, kind=None) -> (action, resource)

    Note that it may return a different resource than it was passed.
    """
    if action == 'skip':
        def on_exists(resource, status, kind=None):
            if strictstatus and status == 'missing':
                raise ValueError(f'got unexpected "missing" status for "skip" action on {resource!r}')
            return None, resource
    elif action == 'replace':
        def on_exists(resource, status, kind=None):
            if strictstatus and status == 'missing':
                raise ValueError(f'got unexpected "missing" status for "replace" action on {resource!r}')
            return 'replace', resource
    elif action == 'fail':
        def on_exists(resource, status, kind=None):
            if strictstatus and status == 'missing':
                raise ValueError(f'got unexpected "missing" status for "fail" action on {resource!r}')
            if status == 'valid':
                label = f'{label} ' if label else ''
                kind = f' ({kind})' if kind else ''
                raise Exception(f'{label}{resource!r} already exists{kind}')
            elif status == 'invalid':
                if label:
                    raise Exception(f"{resource!r} already exists but isn't a {label}")
                else:
                    raise Exception(f"{resource!r} already exists but isn't valid")
            elif status in RESOURCE_STATUS:
                raise NotImplementedError(status)
            elif customstatus and status in customstatus:
                reason = customstatus[status]
                raise Exception(f'{resource!r} exists but {reason}')
            else:
                raise ValueError(f'unsupported status {status!r}')
    elif action in ON_EXISTS_ACTIONS:
        raise NotImplementedError(action)
    else:
        raise ValueError(f'unsupported "on_exists" action {action!r}')
    return on_exists


def add_on_exists_cli(parser)
    parser.add_argument('--action-if-exists', dest='actionifexists',
                        choices=ON_EXISTS_ACTIONS,
                        default=ON_EXISTS_DEFAULT_ACTION)
