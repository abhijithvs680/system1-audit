"""The report quotes a script, so the script has to keep running.

`REPORT.md` is a harness-only write-up whose figures all come from
`examples/report_numbers.py`. If that script breaks, the report becomes a set of
numbers nobody can regenerate, which is the failure mode the script exists to
prevent. These tests are cheap because the script is seeded and small.
"""

import io
import pathlib
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "examples"))

import report_numbers  # noqa: E402


class TestReportNumbers(unittest.TestCase):
    def _run(self) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report_numbers.main()
        return buffer.getvalue()

    def test_it_runs_and_prints_every_section(self):
        output = self._run()
        for heading in (
            "1. A CURVE POINT IS NOT AN OPERATING POINT",
            "2. A VOTE SHARE IS A COARSE GATE",
            "3. AN ECE WITHOUT n AND A BIN COUNT IS UNREADABLE",
            "4. THE ITEM BUDGET CAPS THE FINDING BEFORE THE RUN",
        ):
            self.assertIn(heading, output)

    def test_output_is_identical_across_runs(self):
        # Every measurement is seeded. An audit that reports a different number
        # on a second run is not an audit, and the report would be unquotable.
        self.assertEqual(self._run(), self._run())

    def test_output_is_identical_across_processes(self):
        # Catches a figure that depends on PYTHONHASHSEED, which redirect_stdout
        # in one process cannot detect.
        def once() -> str:
            completed = subprocess.run(
                [sys.executable, "examples/report_numbers.py"],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src", "PYTHONHASHSEED": "0"},
                capture_output=True,
                text=True,
                check=True,
            )
            return completed.stdout

        first = once()
        second = subprocess.run(
            [sys.executable, "examples/report_numbers.py"],
            cwd=REPO_ROOT,
            env={"PYTHONPATH": "src", "PYTHONHASHSEED": "1"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(first, second)

    def test_the_report_exists_and_points_at_this_script(self):
        report = (REPO_ROOT / "REPORT.md").read_text()
        self.assertIn("examples/report_numbers.py", report)

    def test_the_defect_figure_the_report_quotes_still_holds(self):
        # The report's headline number: a threshold reported as meeting a zero
        # error budget while carrying 0.3333 realised risk. If the harness ever
        # stops reproducing it, the report's section 2 is stale.
        output = self._run()
        self.assertIn("0.3333", output)


if __name__ == "__main__":
    unittest.main()
