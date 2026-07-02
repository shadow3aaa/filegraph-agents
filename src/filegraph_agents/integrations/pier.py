"""Pier / Harbor agent adapter for FileGraph Agents.

Lets FGA run as an installed coding agent inside Pier's sandboxed task
environments (e.g. the DeepSWE benchmark). FGA is installed into the task
container from its public git repo, then invoked on the repo at ``/app``.

Register it with Pier via an import path in your job config::

    agents:
      - import_path: filegraph_agents.integrations.pier:FGAAgent
        model_name: openai/deepseek-v4-flash
        env:
          FGA_BASE_URL: https://opencode.ai/zen/go/v1
          FGA_API_KEY: ${FGA_API_KEY}

The task repo lives at ``/app`` and is a real git repo. Pier extracts the
submission as ``git diff <base> HEAD``, so this agent commits FGA's changes
before returning; uncommitted work would be invisible to the grader.
"""

from __future__ import annotations

import shlex
from urllib.parse import urlparse

from pier.agents.installed.base import BaseInstalledAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

# Public repo Pier installs FGA from inside the task container.
FGA_GIT_URL = "git+https://github.com/shadow3aaa/filegraph-agents.git"

# Where Pier/Harbor tasks check out the target repository.
REPO_DIR = "/app"


class FGAAgent(BaseInstalledAgent):
    """Runs FileGraph Agents against the task repo inside a Pier sandbox."""

    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    @staticmethod
    def name() -> str:
        return "filegraph-agents"

    def get_version_command(self) -> str | None:
        return '. "$HOME/.local/bin/env" 2>/dev/null; fga --help >/dev/null 2>&1 && echo 0.1.0'

    # --- install -----------------------------------------------------------

    def install_spec(self) -> AgentInstallSpec:
        # Ensure git + curl + a Python toolchain exist (task images vary widely:
        # Go/Rust/TS images may not ship Python), then install FGA via uv as an
        # isolated tool so it does not disturb the project's own environment.
        root_run = (
            "set -e; "
            "if command -v apt-get >/dev/null 2>&1; then "
            "  apt-get update && apt-get install -y curl git python3 python3-venv; "
            "elif command -v apk >/dev/null 2>&1; then "
            "  apk add --no-cache curl git python3 py3-pip bash; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "  dnf install -y curl git python3; "
            "elif command -v yum >/dev/null 2>&1; then "
            "  yum install -y curl git python3; "
            "fi"
        )
        agent_run = (
            "set -e\n"
            "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh\n"
            'if ! grep -q ".local/bin" "$HOME/.bashrc" 2>/dev/null; then '
            "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> \"$HOME/.bashrc\"; fi\n"
            'source "$HOME/.local/bin/env"\n'
            f"uv tool install --python 3.11 {shlex.quote(FGA_GIT_URL)}\n"
            "fga --help >/dev/null\n"
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(user="root", env={"DEBIAN_FRONTEND": "noninteractive"}, run=root_run),
                InstallStep(user="agent", run=agent_run),
            ],
            verification_command=self.get_version_command(),
        )

    # --- network -----------------------------------------------------------

    def network_allowlist(self) -> NetworkAllowlist:
        """Allow the LLM endpoint (parsed from FGA_BASE_URL) plus PyPI/astral for
        install. Runtime tasks are air-gapped except for these domains."""
        domains: list[str] = [
            "astral.sh",
            "github.com",
            "codeload.github.com",
            "pypi.org",
            "files.pythonhosted.org",
        ]
        base_url = self._get_env("FGA_BASE_URL")
        if base_url:
            host = urlparse(base_url).hostname
            if host:
                domains.append(host)
        return NetworkAllowlist(domains=domains)

    # --- run ---------------------------------------------------------------

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        env = self.build_process_env(
            {
                "FGA_MODEL": self._fga_model(),
                "FGA_BASE_URL": self._get_env("FGA_BASE_URL"),
                "FGA_API_KEY": self._get_env("FGA_API_KEY"),
                # FGA writes run logs under this dir; keep it out of the repo so
                # it does not pollute the captured patch.
                "FGA_LOG_DIR": "/tmp/fga-logs",
            }
        )

        quoted_instruction = shlex.quote(instruction)
        command = (
            '. "$HOME/.local/bin/env"; '
            f"cd {REPO_DIR}; "
            # git identity is required for the commit that follows; task images
            # do not configure one.
            "git config --global --add safe.directory /app 2>/dev/null || true; "
            'git config user.email "fga@example.com" 2>/dev/null || true; '
            'git config user.name "FileGraph Agents" 2>/dev/null || true; '
            f"fga {REPO_DIR} {quoted_instruction} 2>&1 | tee /logs/agent/fga.txt; "
            # Commit whatever FGA changed so Pier's diff-based artifact capture
            # (git diff <base> HEAD) sees it. No-op if nothing changed.
            'git add -A && (git commit -m "FileGraph Agents solution" || true)'
        )
        await self.exec_as_agent(environment, command=command, env=env)

    def populate_context_post_run(self, context: AgentContext) -> None:
        # FGA does not emit an ATIF trajectory; grading is patch-based, so token
        # accounting is best-effort and left empty here.
        return None

    # --- helpers -----------------------------------------------------------

    def _fga_model(self) -> str | None:
        """FGA's model name. Prefer the explicit FGA_MODEL env; otherwise fall
        back to Pier's --model, stripping any provider prefix since FGA adds its
        own openai/ prefix for OpenAI-compatible endpoints."""
        if model := self._get_env("FGA_MODEL"):
            return model
        if self.model_name:
            return self.model_name.split("/", 1)[-1]
        return None
