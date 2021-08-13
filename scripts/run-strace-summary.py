import _common
from fc_tools.strace import parse_args, configure_logger, main


kwargs, verbosity = parse_args()
configure_logger(verbosity)
main(**kwargs)
