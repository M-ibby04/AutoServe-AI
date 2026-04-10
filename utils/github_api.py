"""GitHub API helper utilities for LaunchMind."""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional

import requests


DEFAULT_TIMEOUT = 45
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class GitHubAPIError(Exception):
    """Raised when a GitHub API operation fails."""


class GitHubAPI:
    """Small wrapper around the GitHub REST API for LaunchMind flows."""

    def __init__(self) -> None:
        """Load GitHub configuration from environment variables."""
        self.token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("GITHUB_REPO")
        self.api_base_url = os.getenv("GITHUB_API_BASE_URL", "https://api.github.com").rstrip("/")
        self.timeout = _read_int_env("GITHUB_TIMEOUT", DEFAULT_TIMEOUT)
        self.max_retries = _read_int_env("GITHUB_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        self.commit_author_name = os.getenv("GITHUB_COMMIT_AUTHOR_NAME", "EngineerAgent")
        self.commit_author_email = os.getenv(
            "GITHUB_COMMIT_AUTHOR_EMAIL",
            "agent@launchmind.ai",
        )
        self.committer_name = os.getenv(
            "GITHUB_COMMITTER_NAME",
            self.commit_author_name,
        )
        self.committer_email = os.getenv(
            "GITHUB_COMMITTER_EMAIL",
            self.commit_author_email,
        )

        if not self.token:
            raise GitHubAPIError("Missing GITHUB_TOKEN environment variable.")
        if not self.repo or "/" not in self.repo:
            raise GitHubAPIError(
                "Missing or invalid GITHUB_REPO. Expected format: owner/repo."
            )

        self.owner, self.repo_name = self.repo.split("/", 1)
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def create_issue(self, title: str, body: str) -> Dict[str, Any]:
        """Create a GitHub issue."""
        return self._request(
            method="POST",
            endpoint=f"/repos/{self.owner}/{self.repo_name}/issues",
            json={
                "title": title,
                "body": body,
            },
        )

    def get_default_branch(self) -> str:
        """Fetch the repository's default branch."""
        response = self._request(
            method="GET",
            endpoint=f"/repos/{self.owner}/{self.repo_name}",
        )
        default_branch = response.get("default_branch")
        if not isinstance(default_branch, str) or not default_branch.strip():
            raise GitHubAPIError("Could not determine the repository default branch.")
        return default_branch.strip()

    def get_branch_head_sha(self, branch_name: str) -> str:
        """Fetch the latest commit SHA for a branch."""
        response = self._request(
            method="GET",
            endpoint=f"/repos/{self.owner}/{self.repo_name}/git/ref/heads/{branch_name}",
        )
        sha = response.get("object", {}).get("sha")
        if not isinstance(sha, str) or not sha.strip():
            raise GitHubAPIError(f"Could not determine head SHA for branch '{branch_name}'.")
        return sha.strip()

    def create_branch(self, branch_name: str, base_branch: Optional[str] = None) -> Dict[str, Any]:
        """Create a branch from the repository default branch or a provided base."""
        base_branch = base_branch or self.get_default_branch()
        base_sha = self.get_branch_head_sha(base_branch)
        try:
            return self._request(
                method="POST",
                endpoint=f"/repos/{self.owner}/{self.repo_name}/git/refs",
                json={
                    "ref": f"refs/heads/{branch_name}",
                    "sha": base_sha,
                },
            )
        except GitHubAPIError as exc:
            if "Reference already exists" in str(exc):
                return {"ref": f"refs/heads/{branch_name}", "object": {"sha": base_sha}}
            raise

    def upsert_file(
        self,
        path: str,
        content: str,
        commit_message: str,
        branch: str,
    ) -> Dict[str, Any]:
        """Create or update a file in the repository on a specific branch."""
        existing_sha = self._get_file_sha(path=path, branch=branch)
        encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        payload: Dict[str, Any] = {
            "message": commit_message,
            "content": encoded_content,
            "branch": branch,
            "author": {
                "name": self.commit_author_name,
                "email": self.commit_author_email,
            },
            "committer": {
                "name": self.committer_name,
                "email": self.committer_email,
            },
        }
        if existing_sha:
            payload["sha"] = existing_sha

        return self._request(
            method="PUT",
            endpoint=f"/repos/{self.owner}/{self.repo_name}/contents/{path}",
            json=payload,
        )

    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Open a pull request."""
        base_branch = base_branch or self.get_default_branch()
        return self._request(
            method="POST",
            endpoint=f"/repos/{self.owner}/{self.repo_name}/pulls",
            json={
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch,
            },
        )

    def find_open_pull_request(
        self,
        head_branch: str,
        base_branch: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return an existing open pull request for a branch, if one exists."""
        base_branch = base_branch or self.get_default_branch()
        response = self._request(
            method="GET",
            endpoint=f"/repos/{self.owner}/{self.repo_name}/pulls",
            params={
                "state": "open",
                "head": f"{self.owner}:{head_branch}",
                "base": base_branch,
            },
        )
        if not isinstance(response, list):
            raise GitHubAPIError("Expected a list response when looking up pull requests.")
        if not response:
            return None
        return response[0]

    def create_pr_review_comment(
        self,
        pull_number: int,
        body: str,
        commit_id: str,
        path: str,
        line: int,
    ) -> Dict[str, Any]:
        """Post an inline review comment on a pull request."""
        return self._request(
            method="POST",
            endpoint=f"/repos/{self.owner}/{self.repo_name}/pulls/{pull_number}/comments",
            json={
                "body": body,
                "commit_id": commit_id,
                "path": path,
                "line": line,
                "side": "RIGHT",
            },
        )

    def create_issue_comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        """Post a non-inline comment on a PR or issue."""
        return self._request(
            method="POST",
            endpoint=f"/repos/{self.owner}/{self.repo_name}/issues/{issue_number}/comments",
            json={"body": body},
        )

    def get_pull_request_head_sha(self, pull_number: int) -> str:
        """Fetch the head commit SHA for a pull request."""
        response = self._request(
            method="GET",
            endpoint=f"/repos/{self.owner}/{self.repo_name}/pulls/{pull_number}",
        )
        sha = response.get("head", {}).get("sha")
        if not isinstance(sha, str) or not sha.strip():
            raise GitHubAPIError(f"Could not fetch head SHA for PR #{pull_number}.")
        return sha.strip()

    def _get_file_sha(self, path: str, branch: str) -> Optional[str]:
        """Fetch the current SHA for a file if it already exists."""
        try:
            response = self._request(
                method="GET",
                endpoint=f"/repos/{self.owner}/{self.repo_name}/contents/{path}",
                params={"ref": branch},
            )
        except GitHubAPIError as exc:
            if "404" in str(exc):
                return None
            raise

        sha = response.get("sha")
        if isinstance(sha, str) and sha.strip():
            return sha.strip()
        return None

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a GitHub REST API request with retry handling."""
        url = f"{self.api_base_url}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=json,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise GitHubAPIError(
                        f"Temporary GitHub API failure: {response.status_code} {response.text}"
                    )
                if response.status_code >= 400:
                    raise GitHubAPIError(
                        f"GitHub API request failed: {response.status_code} {response.text}"
                    )
                return response.json()
            except (requests.RequestException, GitHubAPIError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))

        raise GitHubAPIError(f"GitHub request failed after retries: {last_error}") from last_error


def _read_int_env(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise GitHubAPIError(f"Environment variable {name} must be an integer.") from exc


__all__ = ["GitHubAPI", "GitHubAPIError"]
