import pytest


from jobs import __main__, _pyperformance, _utils


from . import helpers


def test_compare(tmp_path, capsys):
    # This is just a simple smoke test

    __main__._parse_and_main(
        helpers.setup_temp_env(tmp_path) + [
            "compare",
            "cpython-3.12.0a0-c20186c397-fc_linux-b2cf916db80e-pyperformance",
            "cpython-3.10.4-9d38120e33-fc_linux-b2cf916db80e-pyperformance",
        ],
        __file__
    )

    expected_start = """
| Benchmark               | cpython-3.12.0a0-c20186c397-fc_linux-b2cf916db80e-pyperformance | cpython-3.10.4-9d38120e33-fc_linux-b2cf916db80e-pyperformance |
    """
    expected_end = """
| Geometric mean          | (ref)                                                           | 1.31x slower                                                  |
+-------------------------+-----------------------------------------------------------------+---------------------------------------------------------------+

Benchmark hidden because not significant (1): pickle
Ignored benchmarks (6) of cpython-3.10.4-9d38120e33-fc_linux-b2cf916db80e-pyperformance.json: genshi_text, genshi_xml, gevent_hub, pylint, sqlalchemy_declarative, sqlalchemy_imperative
    """

    captured = capsys.readouterr()

    assert expected_start.strip() in captured.out
    assert captured.out.strip().endswith(expected_end.strip())


def test_show(tmp_path):
    # This is just a simple smoke test

    with pytest.raises(SystemExit) as exc:
        __main__._parse_and_main(
            helpers.setup_temp_env(tmp_path) + [
                "show",
            ],
            __file__
        )

    assert exc.value.code == 1
