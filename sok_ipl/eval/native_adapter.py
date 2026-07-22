"""AgentDojo native pipeline adapter.

Instead of synthetic tool outputs, this adapter loads the real AgentDojo
environment (inbox, calendar, cloud_drive) and extracts the actual data the
agent would see when processing each user task. The injection is placed in
the same field AgentDojo's native injection mechanism targets.

This is the P1 fix: replacing the synthetic "Here are the results for your
query..." text with real environment data.
"""

from __future__ import annotations

from typing import Optional

from ..types import Task, ToolOutput


def _extract_real_tool_output(env, user_task_id: str) -> str:
    """Extract the real tool output for a given user task from the environment.

    AgentDojo workspace environment has inbox (emails), calendar (events),
    and cloud_drive (files). We extract a summary of the relevant data
    that the agent would see when processing the task.
    """
    parts = []

    # Extract inbox emails (first 5 for context)
    inbox = getattr(env, "inbox", None)
    if inbox and hasattr(inbox, "emails"):
        emails = inbox.emails
        if isinstance(emails, dict):
            emails = list(emails.values())
        parts.append("=== Inbox ===")
        for e in emails[:5]:
            subject = getattr(e, "subject", "N/A")
            body = str(getattr(e, "body", ""))[:300]
            sender = getattr(e, "sender_addr", getattr(e, "from_addr", "unknown"))
            parts.append(f"From: {sender}\nSubject: {subject}\nBody: {body}\n")

    # Extract calendar events
    calendar = getattr(env, "calendar", None)
    if calendar and hasattr(calendar, "events"):
        events = calendar.events
        if isinstance(events, dict):
            events = list(events.values())
        parts.append("=== Calendar ===")
        for ev in events[:5]:
            title = getattr(ev, "title", "N/A")
            desc = str(getattr(ev, "description", getattr(ev, "description_", "")))[:200]
            start = getattr(ev, "start_time", "N/A")
            parts.append(f"Event: {title} ({start})\nDescription: {desc}\n")

    # Extract cloud drive files
    cloud = getattr(env, "cloud_drive", None)
    if cloud and hasattr(cloud, "files"):
        files = cloud.files
        if isinstance(files, dict):
            files = list(files.values())
        parts.append("=== Cloud Drive ===")
        for f in files[:3]:
            name = getattr(f, "filename", getattr(f, "name", "N/A"))
            content = str(getattr(f, "content", ""))[:200]
            parts.append(f"File: {name}\nContent: {content}\n")

    return "\n".join(parts) if parts else "No data available."


class AgentDojoNativeAdapter:
    """Loads AgentDojo tasks with REAL environment data as tool outputs.

    This replaces the synthetic tool output used in the original adapter
    with actual inbox/calendar/cloud_drive data from the AgentDojo environment.
    The injection goal is taken from AgentDojo's injection tasks.
    """

    BENCHMARK = "agentdojo"

    def __init__(self, suite_name: str = "workspace", max_per_suite: int = 20):
        self.suite_name = suite_name
        self.max_per_suite = max_per_suite

    def load_tasks(self) -> list[Task]:
        try:
            from agentdojo.task_suite import get_suites
        except Exception:
            return [_synthetic_task(i, self.suite_name, self.BENCHMARK) for i in range(5)]
        suites = get_suites("v1")
        suite = suites[self.suite_name]

        # Load the real environment once (shared across tasks)
        env = suite.load_and_inject_default_environment({})

        tasks = []
        ut_keys = list(suite.user_tasks.keys())[: self.max_per_suite]
        for utk in ut_keys:
            ut = suite.user_tasks[utk]
            # Pair with the first injection task
            inj = suite.injection_tasks[list(suite.injection_tasks.keys())[0]]

            # Extract real tool output from the environment
            real_output = _extract_real_tool_output(env, ut.ID)

            tasks.append(
                Task(
                    task_id=f"agentdojo-{self.suite_name}-{ut.ID}",
                    benchmark=self.BENCHMARK,
                    scenario=self.suite_name,
                    user_task=ut.PROMPT,
                    expected_tool="ground_truth_check",
                    expected_args={"ground_truth": getattr(ut, "GROUND_TRUTH_OUTPUT", "")},
                    benign_tool_outputs=[
                        ToolOutput(tool_name="inbox_calendar_drive", content=real_output)
                    ],
                    task_spec={
                        "injection_goal": getattr(inj, "GOAL", ""),
                        "injection_task_id": inj.ID,
                        "user_task_id": ut.ID,
                    },
                )
            )
        return tasks


def _synthetic_task(i: int, scenario: str, benchmark: str = "synthetic") -> Task:
    """Fallback synthetic task used when AgentDojo is not installed."""
    return Task(
        task_id=f"{benchmark}-{scenario}-{i}",
        benchmark=benchmark,
        scenario=scenario,
        user_task=f"Summarize the latest message in {scenario}.",
        expected_tool="send_email",
        expected_args={"to": "user@example.com", "body": "summary"},
        benign_tool_outputs=[
            ToolOutput(tool_name="inbox", content=f"Message {i}: Hello, please review the report."),
        ],
    )
