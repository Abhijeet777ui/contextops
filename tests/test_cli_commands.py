from click.testing import CliRunner
from contextops.cli.main import cli


def test_badge_command_green():
    runner = CliRunner()
    result = runner.invoke(cli, ["badge", "--score", "92"])
    assert result.exit_code == 0
    assert "https://img.shields.io/badge/ContextOps-92-green" in result.output
    assert "[![ContextOps]" in result.output


def test_badge_command_yellow():
    runner = CliRunner()
    result = runner.invoke(cli, ["badge", "--score", "75"])
    assert result.exit_code == 0
    assert "ContextOps-75-yellow" in result.output


def test_badge_command_red():
    runner = CliRunner()
    result = runner.invoke(cli, ["badge", "--score", "50"])
    assert result.exit_code == 0
    assert "ContextOps-50-red" in result.output


def test_telemetry_status_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry", "status"])
    assert result.exit_code == 0
    assert "Telemetry is" in result.output
    assert "File path:" in result.output


def test_telemetry_log_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry", "log"])
    assert result.exit_code == 0
