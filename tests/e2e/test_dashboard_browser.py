"""Real-browser end-to-end tests for the Streamlit dashboard.

These tests are opt-in because they need a browser binary and start real
Streamlit servers:

    set CHURNWISE_BROWSER_E2E=1
    python -m pytest tests/e2e -m browser -q

By default Playwright's bundled Chromium is used. If it is not installed,
system Edge/Chrome is used; override with ``CHURNWISE_BROWSER_CHANNEL``.
No external network call is made: every server points at a temporary fixture
tree and ``CHURNWISE_OFFLINE`` is forced on.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Browser, Page, expect, sync_playwright  # noqa: E402

from tests.dashboard_helpers import APP_PATH, install_dashboard_project  # noqa: E402
from tests.risk_helpers import build_risk_fixture  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = PROJECT_ROOT / "reports" / "post_phase13" / "workstream4"
LOG_DIR = EVIDENCE_ROOT / "browser-logs"
FAILURE_DIR = EVIDENCE_ROOT / "browser-failures"

LABELS = [
    "Executive Dashboard",
    "Customer Risk Explorer",
    "Model Laboratory",
    "Retention Workbench",
    "Campaign Simulator",
]

TITLES = {
    "Executive Dashboard": "Executive Dashboard",
    "Customer Risk Explorer": "Customer Risk Explorer",
    "Model Laboratory": "Model Laboratory",
    "Retention Workbench": "Retention Workbench",
    "Campaign Simulator": "Campaign Simulator",
}

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(
        os.environ.get("CHURNWISE_BROWSER_E2E") != "1",
        reason="browser E2E is opt-in; set CHURNWISE_BROWSER_E2E=1",
    ),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _log_tail(path: Path, limit: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<no server log>"
    return "\n".join(lines[-limit:])


def _wait_for_health(url: str, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Streamlit exited with code {process.returncode} before becoming ready.\n"
                f"Server log: {log_path}\n{_log_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.25)
    raise RuntimeError(
        f"Streamlit did not become ready at {url} within 60 s.\n"
        f"Server log: {log_path}\n{_log_tail(log_path)}"
    )


@contextmanager
def _dashboard_server(root: Path, name: str) -> Iterator[str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    log_path = LOG_DIR / f"{name}-{port}.log"
    env = os.environ.copy()
    env.update(
        {
            "CHURNWISE_DATA_DIR": str(root / "data"),
            "CHURNWISE_ARTIFACT_DIR": str(root / "artifacts"),
            "CHURNWISE_REPORT_DIR": str(root / "reports"),
            "CHURNWISE_OFFLINE": "true",
        }
    )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_for_health(url, process, log_path)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.fixture(scope="session")
def dashboard_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("dashboard-browser") / "project"
    root.mkdir()
    install_dashboard_project(root)
    with _dashboard_server(root, "full-fixture") as url:
        yield url


@pytest.fixture
def assessment_url(tmp_path: Path) -> Iterator[str]:
    root = tmp_path / "assessment-project"
    root.mkdir()
    build_risk_fixture(root, rows=240, seed=9)
    with _dashboard_server(root, "assessment-fixture") as url:
        yield url


@pytest.fixture
def empty_url(tmp_path: Path) -> Iterator[str]:
    root = tmp_path / "empty-project"
    for name in ("data", "artifacts", "reports"):
        (root / name).mkdir(parents=True)
    with _dashboard_server(root, "empty-fixture") as url:
        yield url


def _launch_browser(playwright) -> Browser:
    channel = os.environ.get("CHURNWISE_BROWSER_CHANNEL", "").strip()
    attempts = [{"channel": channel}] if channel else [{}, {"channel": "msedge"}, {"channel": "chrome"}]
    errors: list[str] = []
    for options in attempts:
        try:
            return playwright.chromium.launch(headless=True, **options)
        except Exception as exc:  # pragma: no cover - depends on local browser install
            errors.append(f"{options or 'bundled chromium'}: {exc}")
    raise RuntimeError(
        "No Playwright browser is available. Run `python -m playwright install chromium` "
        "or set CHURNWISE_BROWSER_CHANNEL=msedge/chrome.\n" + "\n".join(errors)
    )


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        instance = _launch_browser(playwright)
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser: Browser, request: pytest.FixtureRequest) -> Iterator[Page]:
    context = browser.new_context(
        viewport={"width": 1440, "height": 1200},
        accept_downloads=True,
    )
    current = context.new_page()
    page_errors: list[str] = []
    console_errors: list[str] = []
    current.on("pageerror", lambda exc: page_errors.append(str(exc)))
    current.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
    )
    try:
        yield current
    finally:
        report = getattr(request.node, "rep_call", None)
        if report is not None and report.failed:
            FAILURE_DIR.mkdir(parents=True, exist_ok=True)
            stem = request.node.name.replace("/", "_").replace("\\", "_")
            try:
                current.screenshot(path=str(FAILURE_DIR / f"{stem}.png"), full_page=True)
            except Exception:  # pragma: no cover - best-effort evidence
                pass
            (FAILURE_DIR / f"{stem}-console.log").write_text(
                "PAGE ERRORS\n" + "\n".join(page_errors) + "\n\nCONSOLE ERRORS\n" + "\n".join(console_errors),
                encoding="utf-8",
            )
        context.close()
        if report is not None and not report.failed:
            assert not page_errors, f"uncaught browser exception(s): {page_errors}"


def _open(page: Page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    expect(page.locator(".cw-title")).to_be_visible(timeout=60_000)


def _navigate(page: Page, label: str) -> None:
    page.locator('[data-testid="stSidebar"]').get_by_text(label, exact=True).click()
    # Streamlit can briefly keep the previous page's title in the DOM while the
    # new page streams in. Wait until exactly one title with the expected text
    # remains; asserting on a locator can race with that transient duplicate.
    page.wait_for_function(
        "expected => { const titles = [...document.querySelectorAll('main .cw-title')]"
        ".map(node => node.textContent.trim());"
        "return titles.length === 1 && titles[0] === expected; }",
        arg=TITLES[label],
        timeout=60_000,
    )
    page.wait_for_timeout(300)


def _body(page: Page) -> str:
    return page.locator("body").inner_text()


def test_landing_page_and_complete_navigation_inventory(page: Page, dashboard_url: str) -> None:
    _open(page, dashboard_url)
    expect(page.locator(".cw-title")).to_have_text("Executive Dashboard")
    sidebar = page.locator('[data-testid="stSidebar"]')
    for label in LABELS:
        expect(sidebar.get_by_text(label, exact=True)).to_be_visible()
    body = _body(page)
    for identity in ("RUN ID", "MODEL FAMILY", "DATASET HASH", "GENERATED AT"):
        assert identity in body
    assert page.locator('[data-testid="stException"]').count() == 0


def test_every_registered_page_renders_without_exceptions(page: Page, dashboard_url: str) -> None:
    _open(page, dashboard_url)
    for label in LABELS:
        _navigate(page, label)
        assert page.locator('[data-testid="stException"]').count() == 0, f"{label} raised an exception"
        body = _body(page)
        assert "Traceback" not in body, f"{label} exposed a traceback"
        assert TITLES[label] in body


def test_laboratory_metrics_match_the_stored_fixture(page: Page, dashboard_url: str) -> None:
    _open(page, dashboard_url)
    _navigate(page, "Model Laboratory")
    # The CV section renders after the performance table and plots; wait for it
    # before reading the page so the assertion does not race the last section.
    expect(page.get_by_text("Cross-validation stability", exact=True)).to_be_visible(timeout=60_000)
    body = _body(page)
    assert "Test F1 0.8800" in body
    assert "Test ROC-AUC 0.9400" in body
    assert "Selection used validation F1 only" in body
    assert "Cross-validation stability" in body


def test_risk_filter_and_csv_download(page: Page, dashboard_url: str) -> None:
    _open(page, dashboard_url)
    _navigate(page, "Retention Workbench")
    expect(page.get_by_text("6 selected of 6 matching.", exact=False).first).to_be_visible(timeout=30_000)

    selector = page.locator('[data-testid="stMultiSelect"]').filter(has_text="Risk tier")
    selector.click()
    page.get_by_role("option", name="critical", exact=True).click()
    expect(page.get_by_text("2 selected of 2 matching.", exact=False).first).to_be_visible(timeout=30_000)

    download_button = (
        page.locator('[data-testid="stDownloadButton"]')
        .filter(has_text="Download CSV")
        .locator("button")
    )
    download_button.scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    with page.expect_download(timeout=20_000) as download_info:
        download_button.click(timeout=20_000)
    download = download_info.value
    assert download.suggested_filename.startswith("retention_selected_")
    assert download.suggested_filename.endswith(".csv")


def test_explorer_shows_the_stored_probability(page: Page, dashboard_url: str) -> None:
    _open(page, dashboard_url)
    _navigate(page, "Customer Risk Explorer")
    body = _body(page)
    assert "fixture_ref_000" in body
    assert "0.9000" in body
    assert "Critical" in body
    page.get_by_text("Attribution details and artifact metadata", exact=True).click()
    expect(page.get_by_text("Fixture local explanation summary", exact=False).first).to_be_visible(
        timeout=30_000
    )


def test_campaign_simulator_runs_a_projection(page: Page, dashboard_url: str) -> None:
    _open(page, dashboard_url)
    _navigate(page, "Campaign Simulator")
    page.get_by_text("Assumptions and targeting", exact=True).click()
    run_button = page.get_by_role("button", name="Run projection")
    expect(run_button).to_be_visible(timeout=30_000)
    run_button.click()
    expect(page.get_by_text("Showing a projection computed in this session", exact=False)).to_be_visible(
        timeout=30_000
    )
    # The banner can stream in before the projection body; wait for the section.
    expect(page.get_by_text("Projected outcome", exact=True)).to_be_visible(timeout=60_000)
    body = _body(page)
    assert "Projected outcome" in body
    assert "Hypothetical projection" in body


def test_assessment_without_a_model_is_actionable(page: Page, dashboard_url: str) -> None:
    _open(page, dashboard_url)
    _navigate(page, "Customer Risk Explorer")
    page.get_by_role("tab", name="Score a customer").click()
    page.get_by_role("button", name="Assess this customer").click()
    expect(page.get_by_text("CW-XAI-006", exact=False).first).to_be_visible(timeout=60_000)
    expect(page.get_by_text("Rerun evaluate", exact=False).first).to_be_visible(timeout=60_000)
    assert page.locator('[data-testid="stException"]').count() == 0


def test_individual_assessment_scores_a_record(page: Page, assessment_url: str) -> None:
    _open(page, assessment_url)
    _navigate(page, "Customer Risk Explorer")
    page.get_by_role("tab", name="Score a customer").click()
    form = page.locator('[data-testid="stForm"]').first
    for field, value in (("tenure_months", "1"), ("monthly_charges", "90"), ("support_calls", "4")):
        form.get_by_label(field, exact=False).fill(value)
    page.get_by_role("button", name="Assess this customer").click()
    expect(page.get_by_text("Individual assessment", exact=False).first).to_be_visible(timeout=60_000)
    labels = page.locator('[data-testid="stMetricLabel"]')
    expect(labels.filter(has_text=re.compile("churn probability", re.IGNORECASE)).first).to_be_visible(
        timeout=60_000
    )
    expect(labels.filter(has_text=re.compile("^risk tier$", re.IGNORECASE)).first).to_be_visible(timeout=30_000)
    expect(page.get_by_text("Local SHAP drivers were not requested", exact=False).first).to_be_visible(
        timeout=30_000
    )
    expect(
        page.locator('[data-testid="stDownloadButton"]')
        .filter(has_text="Download this assessment (JSON)")
    ).to_be_visible()
    assert page.locator('[data-testid="stException"]').count() == 0


def test_batch_csv_upload_scores_and_exports(page: Page, assessment_url: str, tmp_path: Path) -> None:
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "tenure_months,monthly_charges,support_calls,contract\n"
        "12,70.5,3,\n"
        "24,80.0,1,\n",
        encoding="utf-8",
    )
    _open(page, assessment_url)
    _navigate(page, "Customer Risk Explorer")
    page.get_by_role("tab", name="Batch scoring").click()
    page.locator('input[type="file"]').set_input_files(str(csv_path))
    expect(page.get_by_text("Parsed 2 rows and 4 columns.", exact=False)).to_be_visible(timeout=30_000)
    page.get_by_role("button", name="Assess uploaded records").click()
    expect(page.get_by_text("Assessed 2 of 2 rows.", exact=False)).to_be_visible(timeout=60_000)
    labels = page.locator('[data-testid="stMetricLabel"]')
    for label in ("received", "scored", "rejected"):
        expect(labels.filter(has_text=re.compile(f"^{label}$", re.IGNORECASE)).first).to_be_visible(
            timeout=30_000
        )

    download_button = (
        page.locator('[data-testid="stDownloadButton"]')
        .filter(has_text="Download batch results (CSV)")
        .locator("button")
    )
    download_button.scroll_into_view_if_needed()
    with page.expect_download(timeout=20_000) as download_info:
        download_button.click(timeout=20_000)
    assert download_info.value.suggested_filename == "churnwise_batch_results.csv"


def test_empty_artifacts_are_actionable_on_every_page(page: Page, empty_url: str) -> None:
    _open(page, empty_url)
    for label in LABELS:
        _navigate(page, label)
        assert page.locator('[data-testid="stException"]').count() == 0, f"{label} raised an exception"
        body = _body(page)
        assert "churnwise.cli" in body, f"{label} did not expose a next-step command"
        assert "not available" in body.lower() or "no " in body.lower()
