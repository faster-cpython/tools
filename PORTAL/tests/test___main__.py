import json
import re
import shutil
import textwrap


import pytest


from jobs import __main__


from . import helpers


def _compare_lines(s):
    return [x.strip() for x in s.split("\n") if x.strip()]


def test_compare(tmp_path, capsys):
    # This is just a simple smoke test

    __main__._parse_and_main(
        [
            *helpers.setup_temp_env(tmp_path),
            "compare",
            "cpython-3.12.0a0-c20186c397-fc_linux-b2cf916db80e-pyperformance",
            "cpython-3.10.4-9d38120e33-fc_linux-b2cf916db80e-pyperformance",
        ],
        __file__,
    )

    expected_start = """
| Benchmark               | cpython-3.12.0a0-c20186c397-fc_linux-b2cf916db80e-pyperformance | cpython-3.10.4-9d38120e33-fc_linux-b2cf916db80e-pyperformance |
    """
    expected_end = """
| Geometric mean          | (ref)                                                           | 1.31x slower                                                  |
+-------------------------+-----------------------------------------------------------------+---------------------------------------------------------------+

Benchmark hidden because not significant (1): pickle
Ignored benchmarks (2) of cpython-3.10.4-9d38120e33-fc_linux-b2cf916db80e-pyperformance.json: genshi_text, genshi_xml
    """

    captured = capsys.readouterr()

    print(captured.out)

    assert expected_start.strip() in captured.out
    assert captured.out.strip().endswith(expected_end.strip())


def test_show(tmp_path):
    # This is just a simple smoke test

    with pytest.raises(SystemExit) as exc:
        __main__._parse_and_main(
            [
                *helpers.setup_temp_env(tmp_path),
                "show",
            ],
            __file__,
        )

    assert exc.value.code == 1


def test_show_with_content(tmp_path, capsys):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid

    reqdir.mkdir()

    shutil.copy(helpers.DATA_ROOT / "request.json", reqdir / "request.json")
    shutil.copy(helpers.DATA_ROOT / "results.json", reqdir / "results.json")

    (reqdir / "job.log").write_text(
        textwrap.dedent(
            """
            LINE 1 OF LOG
            LINE 2 OF LOG
            LINE 3 OF LOG
            """
        )
    )

    __main__._parse_and_main(
        [*args, "show", reqid, "--lines", "2"],
        __file__,
    )

    captured = capsys.readouterr()

    assert re.fullmatch(
        textwrap.dedent(
            r"""
            Request req-compile-bench-1664291728-nobody-mac:
              kind:                  compile-bench
              user:                  nobody
              status:                pending
              is staged:             False

            Details:
              ref:                   main
              pyperformance_ref:     dd53b79de0ea98af6a11481217a961daef4e9774
              remote:                origin
              revision:              main
              branch:                main
              benchmarks:            \['deepcopy'\]
              optimize:              True
              debug:                 False
              ssh okay:              \?\?\?

            History:
              created:               2022-09-27 15:15:28
              pending:               2022-09-27 15:15:29

            Request files:
              data root:             \(/home/benchmarking/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac\)
              metadata:              .*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/request.json
              job_script:            \(.*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/run.sh\)
              portal_script:         \(.*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/send.sh\)
              ssh_okay:              \(.*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/ssh.ok\)
              pyperformance_manifest: \(.*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/benchmarks.manifest\)
              pyperformance_config:  \(.*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/pyperformance.ini\)

            Result files:
              data root:             .*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac
              metadata:              .*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/results.json
              pidfile:               \(.*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/send.pid\)
              logfile:               .*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/job.log
              pyperformance_log:     \(.*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/pyperformance.log\)
              pyperformance_results: \(.*?/BENCH/REQUESTS/req-compile-bench-1664291728-nobody-mac/pyperformance-results.json.gz\)
            LINE 2 OF LOG
            LINE 3 OF LOG
        """
        ).strip(),
        captured.out.strip(),
    )


def test_list(tmp_path, capsys):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
    reqdir.mkdir()
    shutil.copy(helpers.DATA_ROOT / "request.json", reqdir / "request.json")
    shutil.copy(helpers.DATA_ROOT / "results.json", reqdir / "results.json")

    (tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json").write_text(
        json.dumps({"jobs": [reqid], "paused": False})
    )

    __main__._parse_and_main(
        [
            *args,
            "list",
        ],
        __file__,
    )

    captured = capsys.readouterr()

    # TODO: cmd_list uses a combination of print and logger.info -- it's not
    # clear what we should test for here.

    assert _compare_lines(captured.out) == _compare_lines(
        textwrap.dedent(
            """
            request ID                       status      elapsed
            ----------------------------------------------- ------------ ------------
             req-compile-bench-1664291728-nobody-mac         pending             ---
            """
        )
    )


def test_queue_info(tmp_path, capsys):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [reqid], "paused": False}))

    __main__._parse_and_main(
        [
            *args,
            "queue",
            "info",
        ],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == [reqid]

    captured = capsys.readouterr()

    assert re.match(
        textwrap.dedent(
            r"""
            Job Queue \(linux\):
              size:     0
              paused:   False
              lock:     \(not locked\)

            Files:
              data:      .*?/BENCH/QUEUES/linux/queue.json
              lock:      \(.*?/BENCH/QUEUES/linux/queue.lock\)
              log:       \(.*?/BENCH/QUEUES/linux/queue.log\)

            Top 5:
              \(queue is empty\)

            Log size:    0
            Last log entry:
              \(log is empty\)

            Job Queue \(mac\):
              size:     1
              paused:   False
              lock:     \(not locked\)

            Files:
              data:      .*?/BENCH/QUEUES/mac/queue.json
              lock:      \(.*?/BENCH/QUEUES/mac/queue.lock\)
              log:       \(.*?/BENCH/QUEUES/mac/queue.log\)

            Top 5:
              1 req-compile-bench-1664291728-nobody-mac

            Log size:    0
            Last log entry:
              \(log is empty\)
            """
        ).strip(),
        captured.out.strip(),
    )


def test_queue_info_single(tmp_path, capsys):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [reqid], "paused": False}))

    __main__._parse_and_main(
        [*args, "queue", "info", "mac"],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == [reqid]

    captured = capsys.readouterr()

    assert re.match(
        textwrap.dedent(
            r"""
            Job Queue \(mac\):
              size:     1
              paused:   False
              lock:     \(not locked\)

            Files:
              data:      .*?/BENCH/QUEUES/mac/queue.json
              lock:      \(.*?/BENCH/QUEUES/mac/queue.lock\)
              log:       \(.*?/BENCH/QUEUES/mac/queue.log\)

            Top 5:
              1 req-compile-bench-1664291728-nobody-mac

            Log size:    0
            Last log entry:
              \(log is empty\)
            """
        ).strip(),
        captured.out.strip(),
    )


def test_queue_list(tmp_path, capsys):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [reqid], "paused": False}))

    __main__._parse_and_main(
        [
            *args,
            "queue",
            "list",
        ],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == [reqid]

    captured = capsys.readouterr()

    assert (
        captured.out.strip()
        == textwrap.dedent(
            """
            Queue (linux)
            no jobs queued
            Queue (mac)
              1 req-compile-bench-1664291728-nobody-mac

            (total: 1)
            """
        ).strip()
    )


def test_queue_list_single(tmp_path, capsys):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    (tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json").write_text(
        json.dumps({"jobs": [reqid], "paused": False})
    )

    __main__._parse_and_main(
        [*args, "queue", "list", "mac"],
        __file__,
    )

    captured = capsys.readouterr()

    assert (
        captured.out.strip()
        == textwrap.dedent(
            """
            Queue (mac)
              1 req-compile-bench-1664291728-nobody-mac

            (total: 1)
            """
        ).strip()
    )


def test_queue_pause(tmp_path):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [reqid], "paused": False}))

    __main__._parse_and_main(
        [*args, "queue", "pause", "mac"],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["paused"] is True


def test_queue_unpause(tmp_path):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [reqid], "paused": True}))

    __main__._parse_and_main(
        [*args, "queue", "unpause", "mac"],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["paused"] is False


def test_queue_pop(tmp_path):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [reqid], "paused": False}))
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
    reqdir.mkdir()
    shutil.copy(helpers.DATA_ROOT / "results.json", reqdir / "results.json")

    __main__._parse_and_main(
        [*args, "queue", "pop", "mac"],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == []


def test_queue_push(tmp_path, caplog):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [], "paused": False}))
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
    reqdir.mkdir()
    shutil.copy(helpers.DATA_ROOT / "results-created.json", reqdir / "results.json")

    __main__._parse_and_main(
        [*args, "queue", "push", reqid],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == [reqid]

    assert (
        "Adding job req-compile-bench-1664291728-nobody-mac to the queue" in caplog.text
    )


def test_queue_push_already_created(tmp_path, caplog):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [], "paused": False}))
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
    reqdir.mkdir()
    shutil.copy(helpers.DATA_ROOT / "results.json", reqdir / "results.json")

    with pytest.raises(SystemExit) as exc:
        __main__._parse_and_main(
            [*args, "queue", "push", reqid],
            __file__,
        )

    assert exc.value.code == 1

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == []

    assert (
        "request req-compile-bench-1664291728-nobody-mac has already been used"
        in caplog.text
    )


def test_queue_push_default_worker(tmp_path, caplog):
    args = helpers.setup_temp_env(tmp_path)

    # If the workerid isn't provided in the reqid, we want to assert that it is
    # managed on the default "linux" queue.
    reqid = "req-compile-bench-1664291728-nobody"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "linux" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [], "paused": False}))
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
    reqdir.mkdir()
    shutil.copy(helpers.DATA_ROOT / "results-created.json", reqdir / "results.json")

    __main__._parse_and_main(
        [*args, "queue", "push", reqid],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == [reqid]

    assert "Adding job req-compile-bench-1664291728-nobody to the queue" in caplog.text


def test_queue_move(tmp_path, caplog):
    args = helpers.setup_temp_env(tmp_path)

    reqid1 = "req-compile-bench-1664291728-nobody-mac"
    reqid2 = "req-compile-bench-1664291729-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [reqid1, reqid2], "paused": False}))

    for reqid in (reqid1, reqid2):
        reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
        reqdir.mkdir()
        shutil.copy(helpers.DATA_ROOT / "results.json", reqdir / "results.json")

    # TODO: There may be an off-by-one error in the code -- moving to position
    # "2" doesn't change anything

    __main__._parse_and_main(
        [*args, "queue", "move", reqid1, "3"],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == [reqid2, reqid1]

    assert (
        "Moving job req-compile-bench-1664291728-nobody-mac to position 3 in the queue..."
        in caplog.text
    )


def test_queue_remove(tmp_path, caplog):
    args = helpers.setup_temp_env(tmp_path)

    reqid1 = "req-compile-bench-1664291728-nobody-mac"
    reqid2 = "req-compile-bench-1664291729-nobody-mac"
    queue_file = tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json"
    queue_file.write_text(json.dumps({"jobs": [reqid1, reqid2], "paused": False}))

    for reqid in (reqid1, reqid2):
        reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
        reqdir.mkdir()
        shutil.copy(helpers.DATA_ROOT / "results.json", reqdir / "results.json")

    __main__._parse_and_main(
        [*args, "queue", "remove", reqid1],
        __file__,
    )

    content = json.loads(queue_file.read_text())
    assert content["jobs"] == [reqid2]

    assert (
        "Removing job req-compile-bench-1664291728-nobody-mac from the queue..."
        in caplog.text
    )


def test_config_show(tmp_path, capsys):
    args = helpers.setup_temp_env(tmp_path)

    __main__._parse_and_main([*args, "config"], __file__)

    captured = capsys.readouterr()

    assert json.loads((tmp_path / "jobs.json").read_text()) == json.loads(captured.out)


def test_run_bench(tmp_path, monkeypatch):
    # Just a basic smoke test

    def dummy(*args, **kwargs):
        return

    monkeypatch.setitem(__main__.COMMANDS, "attach", dummy)

    __main__._parse_and_main(
        [
            *helpers.setup_temp_env(tmp_path),
            "run-bench",
            "--worker",
            "mac",
            "--benchmarks",
            "deepcopy",
            "caf63ec5",
        ],
        __file__,
    )

    queue = json.loads(
        (tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json").read_text()
    )
    (reqid,) = queue["jobs"]
    assert reqid.endswith("-mac")
    assert reqid.startswith("req-compile-bench")

    (reqdir,) = list((tmp_path / "BENCH" / "REQUESTS").iterdir())
    assert reqdir.name == reqid

    files = sorted(x.name for x in reqdir.iterdir())

    assert [
        "benchmarks.manifest",
        "pyperformance.ini",
        "request.json",
        "results.json",
        "run.sh",
        "send.sh",
    ] == files


def test_cancel(tmp_path, caplog):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    (tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json").write_text(
        json.dumps({"jobs": [reqid], "paused": False})
    )
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
    reqdir.mkdir()
    shutil.copy(helpers.DATA_ROOT / "results.json", reqdir / "results.json")

    __main__._parse_and_main(
        [*args, "cancel", reqid],
        __file__,
    )

    assert (
        "Removing job req-compile-bench-1664291728-nobody-mac from the queue..."
        in caplog.text
    )


def test_cancel_no_reqid_specified(tmp_path, caplog, monkeypatch):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    (tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json").write_text(
        json.dumps({"jobs": [reqid], "paused": False})
    )
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
    reqdir.mkdir()
    shutil.copy(helpers.DATA_ROOT / "results-activated.json", reqdir / "results.json")

    (reqdir / "send.pid").write_text("1234")

    from jobs import _current

    def get_staged_request(jobsfs, symlink=None):
        if not symlink:
            symlink = _current.symlink_from_jobsfs(jobsfs)
        return _current._read_staged(symlink, jobsfs)

    monkeypatch.setattr(_current, "get_staged_request", get_staged_request)

    current = tmp_path / "BENCH" / "REQUESTS" / "CURRENT"
    current.symlink_to(reqdir)

    __main__._parse_and_main(
        [*args, "cancel"],
        __file__,
    )

    assert (
        "job req-compile-bench-1664291728-nobody-mac no longer"
        in caplog.text
    )


def test_cancel_reqid_missing(tmp_path, caplog):
    args = helpers.setup_temp_env(tmp_path)

    reqid = "req-compile-bench-1664291728-nobody-mac"
    (tmp_path / "BENCH" / "QUEUES" / "mac" / "queue.json").write_text(
        json.dumps({"jobs": [reqid], "paused": False})
    )
    reqdir = tmp_path / "BENCH" / "REQUESTS" / reqid
    reqdir.mkdir()
    shutil.copy(helpers.DATA_ROOT / "results.json", reqdir / "results.json")

    with pytest.raises(SystemExit) as exc:
        __main__._parse_and_main(
            [*args, "cancel", "req-compile-bench-99999999990-nobody-mac"],
            __file__,
        )

    assert exc.value.code == 1

    assert (
        "expected a valid reqid, got 'req-compile-bench-99999999990-nobody-mac'"
        in caplog.text
    )


# TODO: cmd_wait (this is tricky one to test)


# TODO: cmd_upload (partially covered by test_run_bench)
