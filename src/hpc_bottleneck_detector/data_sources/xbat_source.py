"""
XBAT REST API Data Source Implementation

This module provides a data source that fetches HPC job metrics directly
from the XBAT REST API.

Authentication flow:
    1. Load a cached OAuth token from a local file (if it exists)
    2. Validate the token via GET /api/v1/current_user
    3. If invalid/missing, request a new one via POST /oauth/token
       using the Resource Owner Password Credentials grant

CSV endpoint:
    GET /api/v1/measurements/{job_id}/csv
        ?group=<group>   (optional)
        &metric=<metric> (optional, only when group is set)
        &level=<level>   (default: 'job')
        &node=<node>     (only when level='node')

Credentials / environment variables:
    The recommended way to supply credentials is via a ``.env`` file (which
    should be git-ignored) and the :meth:`XBATDataSource.from_env` factory.
    Copy ``.env.example`` to ``.env`` and fill in your values::

        XBAT_API_BASE=https://xbat-master:7000
        XBAT_USERNAME=your_username
        XBAT_PASSWORD=your_password
        XBAT_CLIENT_ID=your_client_id
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .interface import IDataSource
from ..data.manager import DataManager
from ..data.job_context import JobContext


# ---------------------------------------------------------------------------
# Default fallback credentials (public demo instance)
# ---------------------------------------------------------------------------
_DEFAULT_API_BASE = "https://demo.xbat.dev"
_DEFAULT_USERNAME = "demo"
_DEFAULT_PASSWORD = "demo"
_DEFAULT_CLIENT_ID = "demo"
_DEFAULT_TOKEN_FILE = ".env.xbat"


class XBATDataSource(IDataSource):
    """
    Data source that fetches job metrics from the XBAT REST API.

    Attributes:
        api_base:    Base URL of the XBAT instance (no trailing slash).
        username:    Username used for the password-grant OAuth flow.
        password:    Corresponding password.
        client_id:   OAuth client ID.
        group:       Metric group to filter (empty → all groups, job-level only).
        metric:      Metric name within the group (must be empty when group is empty).
        level:       Aggregation level: ``'job'`` | ``'node'`` | ``'core'``.
        node:        Node identifier (required only when level is ``'node'``).
        token_file:  Path where the cached access token is stored.
        proxies:     Optional proxy mapping forwarded to the underlying
                     ``requests.Session``, e.g.
                     ``{'http': 'socks5h://localhost:xxx', 'https': 'socks5h://localhost:xxx'}``.
        session:     Underlying ``requests.Session`` (created automatically).
    """

    def __init__(
        self,
        api_base: str = _DEFAULT_API_BASE,
        username: str = _DEFAULT_USERNAME,
        password: str = _DEFAULT_PASSWORD,
        client_id: str = _DEFAULT_CLIENT_ID,
        group: str = "",
        metric: str = "",
        level: str = "job",
        node: str = "",
        token_file: str = _DEFAULT_TOKEN_FILE,
        proxies: Optional[dict] = None,
        verify_ssl: bool = True,
    ) -> None:
        # Validate argument combinations
        if not group and metric:
            raise ValueError("'metric' must be empty when 'group' is not set.")
        if level == "node" and not node:
            raise ValueError("'node' must be provided when level is 'node'.")

        self.api_base = api_base.rstrip("/")
        self.username = username
        self.password = password
        self.client_id = client_id
        self.group = group
        self.metric = metric
        self.level = level or "job"
        self.node = node
        self.token_file = Path(token_file)
        self.proxies = proxies or {}
        self.verify_ssl = verify_ssl

        self._access_token: Optional[str] = None
        self.session = requests.Session()
        if self.proxies:
            self.session.proxies.update(self.proxies)
        if not self.verify_ssl:
            self.session.verify = False
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Attempt to reuse a cached token on construction
        self._load_token()
        if not self._validate_token():
            self._request_new_token()

    # ------------------------------------------------------------------
    # Environment-based factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        env_file: str = ".env",
        group: str = "",
        metric: str = "",
        level: str = "job",
        node: str = "",
        token_file: str = _DEFAULT_TOKEN_FILE,
        proxies: Optional[dict] = None,
        verify_ssl: bool = True,
    ) -> "XBATDataSource":
        """
        Construct an :class:`XBATDataSource` from environment variables.

        Reads the following variables from the environment (after optionally
        loading *env_file* via ``python-dotenv``):

        +--------------------+------------------------------------------+
        | Variable           | Description                              |
        +====================+==========================================+
        | ``XBAT_API_BASE``  | Base URL (``https://xbat-master:7000``)  |
        | ``XBAT_USERNAME``  | OAuth username                           |
        | ``XBAT_PASSWORD``  | OAuth password                           |
        | ``XBAT_CLIENT_ID`` | OAuth client ID                          |
        | ``XBAT_PROXY``     | Proxy URL, (``socks5h://localhost:xxx``) |
        | ``XBAT_VERIFY_SSL``| Set to ``false`` to skip TLS validation  |
        +--------------------+------------------------------------------+

        Each variable falls back to the public demo credentials if not set.
        ``XBAT_PROXY``, when set, is applied to both ``http`` and ``https``.
        Copy ``.env.example`` to ``.env`` and fill in your real values.

        Args:
            env_file:   Path to the ``.env`` file to load. Silently ignored if
                        the file does not exist or ``python-dotenv`` is not
                        installed.
            group:      Forwarded to :meth:`__init__`.
            metric:     Forwarded to :meth:`__init__`.
            level:      Forwarded to :meth:`__init__`.
            node:       Forwarded to :meth:`__init__`.
            token_file: Forwarded to :meth:`__init__`.
            proxies:    Explicit proxy dict; overrides ``XBAT_PROXY`` when given.
            verify_ssl: Whether to verify TLS certificates. Set to ``False`` for
                        servers with self-signed certs; overrides ``XBAT_VERIFY_SSL``.

        Returns:
            A fully initialised :class:`XBATDataSource` instance.

        Example::

            # Copy .env.example → .env and fill in real credentials, then:
            source = XBATDataSource.from_env()
        """
        try:
            from dotenv import load_dotenv  # optional dependency
            load_dotenv(dotenv_path=env_file, override=False)
        except ImportError:
            pass  # python-dotenv not installed; rely on already-set env vars

        # Build proxy dict from XBAT_PROXY env var when not supplied explicitly
        if proxies is None:
            proxy_url = os.environ.get("XBAT_PROXY", "")
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}

        # Resolve verify_ssl from XBAT_VERIFY_SSL env var
        env_verify = os.environ.get("XBAT_VERIFY_SSL", "").strip().lower()
        if env_verify in ("false", "0", "no"):
            verify_ssl = False

        return cls(
            api_base=os.environ.get("XBAT_API_BASE", _DEFAULT_API_BASE),
            username=os.environ.get("XBAT_USERNAME", _DEFAULT_USERNAME),
            password=os.environ.get("XBAT_PASSWORD", _DEFAULT_PASSWORD),
            client_id=os.environ.get("XBAT_CLIENT_ID", _DEFAULT_CLIENT_ID),
            group=group,
            metric=metric,
            level=level,
            node=node,
            token_file=token_file,
            proxies=proxies,
            verify_ssl=verify_ssl,
        )

    # ------------------------------------------------------------------
    # IDataSource interface
    # ------------------------------------------------------------------

    def fetch_job_data(self, job_id: str) -> DataManager:
        """
        Download the CSV for *job_id* from XBAT and return a DataManager.

        The query parameters (group, metric, level, node) are taken from the
        values set during construction.

        Args:
            job_id: XBAT job identifier.

        Returns:
            DataManager wrapping the downloaded metrics.

        Raises:
            ValueError: If the server returns 404 (job / combination not found).
            IOError:    On any other non-200 HTTP status or parse failure.
        """
        url = self._build_url(job_id)

        response = self.session.get(
            url,
            headers={
                "accept": "text/csv",
                "Authorization": f"Bearer {self._access_token}",
            },
        )

        if response.status_code == 401:
            # Token may have expired mid-session; refresh once and retry
            self._request_new_token()
            response = self.session.get(
                url,
                headers={
                    "accept": "text/csv",
                    "Authorization": f"Bearer {self._access_token}",
                },
            )

        if response.status_code == 404:
            raise ValueError(
                f"Job ID '{job_id}' or the requested group/metric/level combination "
                "was not found on the XBAT server."
            )
        if response.status_code != 200:
            raise IOError(
                f"XBAT API returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            df = pd.read_csv(io.StringIO(response.text))
        except Exception as exc:
            raise IOError(f"Failed to parse CSV response from XBAT: {exc}") from exc

        job_context = self._fetch_job_context(job_id)
        return DataManager(df.reset_index(drop=True), job_context=job_context)

    # ------------------------------------------------------------------
    # Job context
    # ------------------------------------------------------------------

    def _fetch_job_context(self, job_id: str) -> Optional[JobContext]:
        """
        Build a :class:`~hpc_bottleneck_detector.data.job_context.JobContext`
        for *job_id* by calling the XBAT metadata endpoints.

        Flow:
            1. ``GET /api/v1/jobs?short=true`` - find the entry for *job_id*
               and collect the node hashes used by that job.
            2. ``GET /api/v1/nodes?node_hashes=<h1>,<h2>,…`` - fetch the
               hardware description for each unique hash.
            3. Extract relevant fields (benchmarks, CPU, memory, OS) and
               return a populated :class:`JobContext`.

        Returns:
            :class:`JobContext` on success, ``None`` if the job cannot be
            found or the metadata endpoints are unavailable.
        """
        try:
            job_entry = self._find_job_entry(job_id)
            if job_entry is None:
                return None

            node_hashes = list({
                meta.get("hash")
                for meta in job_entry.get("nodes", {}).values()
                if meta.get("hash")
            })
            if not node_hashes:
                return None

            node_hardware_raw = self._fetch_node_hardware(node_hashes)
            return JobContext.from_xbat(job_id, job_entry, node_hardware_raw)

        except Exception:  # pragma: no cover – best-effort, never fatal
            return None

    def _find_job_entry(self, job_id: str) -> Optional[dict]:
        """
        Return the job dict from ``GET /api/v1/jobs?short=true`` for *job_id*.

        Returns ``None`` if the job is not present in the listing.
        """
        resp = self.session.get(
            f"{self.api_base}/api/v1/jobs",
            params={"short": "true"},
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=30,
        )
        if resp.status_code != 200:
            return None

        jobs: list = resp.json().get("data", [])
        for entry in jobs:
            if str(entry.get("jobId")) == str(job_id):
                return entry
        return None

    def _fetch_node_hardware(self, node_hashes: list) -> dict:
        """
        Call ``GET /api/v1/nodes?node_hashes=<h1>,<h2>,…`` and return the
        raw response dict (keyed by hash).
        """
        hashes_param = ",".join(node_hashes)
        resp = self.session.get(
            f"{self.api_base}/api/v1/nodes",
            params={"node_hashes": hashes_param},
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=30,
        )
        if resp.status_code != 200:
            return {}
        return resp.json()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _load_token(self) -> None:
        """Load a previously cached access token from *token_file*, if present."""
        if not self.token_file.exists():
            return
        with self.token_file.open() as fh:
            for line in fh:
                key, _, value = line.strip().partition("=")
                if key == "ACCESS_TOKEN":
                    self._access_token = value
                    break

    def _save_token(self) -> None:
        """Persist the current access token to *token_file*."""
        with self.token_file.open("w") as fh:
            fh.write(f"ACCESS_TOKEN={self._access_token}\n")
        # Restrict file permissions so the token is not world-readable
        self.token_file.chmod(0o600)

    def _validate_token(self) -> bool:
        """
        Return ``True`` if the current token is accepted by the server.

        A missing token is immediately treated as invalid without a network
        round-trip.
        """
        if not self._access_token:
            return False
        try:
            resp = self.session.get(
                f"{self.api_base}/api/v1/current_user",
                headers={"Authorization": f"Bearer {self._access_token}"},
                timeout=10,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _request_new_token(self) -> None:
        """
        Obtain a fresh access token via the Resource Owner Password Credentials OAuth Grant.

        Raises:
            IOError: If the server does not return a valid access_token.
        """
        resp = self.session.post(
            f"{self.api_base}/oauth/token",
            data={
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": self.client_id,
            },
            timeout=10,
        )
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise IOError(
                f"Failed to obtain XBAT access token. "
                f"Server response: {payload}"
            )
        self._access_token = token
        self._save_token()

    # ------------------------------------------------------------------
    # URL builder
    # ------------------------------------------------------------------

    def _build_url(self, job_id: str) -> str:
        """Construct the full CSV endpoint URL including query parameters."""
        base = f"{self.api_base}/api/v1/measurements/{job_id}/csv"
        params: dict[str, str] = {}

        if self.group:
            params["group"] = self.group
        if self.metric:
            params["metric"] = self.metric
        if self.level:
            params["level"] = self.level
        if self.node:
            params["node"] = self.node

        if not params:
            return base

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base}?{query}"
