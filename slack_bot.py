"""Slack bot for the Company Policy Assistant.

Mention the bot in a channel ("@PolicyBot how many vacation days do I get?") or send it a direct
message. It answers with the vector-index approach (LLM + Gemini embeddings) and names the policy.
If that fails (quota, network) it falls back to the LLM without an index, then to rules-based search.

Run it with:  python slack_bot.py     (needs: pip install -r requirements.txt)
It connects to Slack over Socket Mode, so it runs fine on your own computer: no public URL needed.
"""
import os
import re
import sys

from lib import vector_index
from lib.approaches import safe_run
from lib.config import config
from lib.policies import load_policies
from lib.rules import RulesIndex

ORDER = ["vector", "llm", "rules"]
LABEL = {
    "vector": "LLM + vector index",
    "llm": "LLM without index",
    "rules": "rules-based search",
}
HELP = ("Ask me a question about company policy, for example: _How many vacation days do I get?_ "
        "I answer from the policy database and tell you which policy it comes from.")


def _mrkdwn(text):
    """Escape the three characters Slack treats specially."""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def clean_question(text):
    """Remove @mentions (<@U123ABC>) and surrounding whitespace."""
    return re.sub(r"<@[A-Z0-9]+>", "", text or "").strip()


def build_reply(question, policies, rules_index):
    """Answer one question. Returns the Slack message text. Works without Slack, so it can be tested."""
    question = clean_question(question)
    if not question:
        return HELP
    if len(question) > config.max_question_length:
        return f"Please keep the question under {config.max_question_length} characters."

    result, used, errors = None, None, []
    for name in ORDER:
        r = safe_run(name, question, policies, rules_index)
        if "error" in r:
            errors.append(f"{LABEL[name]}: {r['error']}")
            continue
        result, used = r, name
        break
    if result is None:
        return "Sorry, I couldn't answer that right now.\n" + "\n".join(f"• {_mrkdwn(e)}" for e in errors)

    lines = []
    if result["answerable"] and result["policy"]:
        p = result["policy"]
        lines.append(_mrkdwn(result["answer"]))
        lines.append(f"*Relevant policy:* {_mrkdwn(p['title'])} ({_mrkdwn(p['department'])} department, {_mrkdwn(p['category'])})")
        if result["grounding"]["status"] == "unsupported":
            lines.append(":warning: _Part of this answer could not be verified against the policy text._")
    else:
        lines.append("I couldn't find a policy in the database that answers this question.")

    m = result["metrics"]
    total_tokens = m["tokens"]["total"] + m.get("embedding_tokens", 0)
    footer = f"{LABEL[used]} · {m['ms'] / 1000:.1f} s · {total_tokens:,} tokens"
    if m.get("model"):
        footer += f" · {m['model']}"
    if used != ORDER[0]:
        footer += f" · fell back from {LABEL[ORDER[0]]}"
    lines.append(f"_{footer}_")
    return "\n".join(lines)


def create_app(bot_token, policies, rules, **app_kwargs):
    """Build the Slack app with its two handlers. Extra keyword arguments go to slack_bolt.App (used by tests)."""
    from slack_bolt import App

    app = App(token=bot_token, **app_kwargs)

    @app.event("app_mention")
    def on_mention(event, say):
        # Reply in a thread so the channel stays tidy.
        say(text=build_reply(event.get("text", ""), policies, rules), thread_ts=event.get("thread_ts") or event["ts"])

    @app.event("message")
    def on_message(event, say):
        # Direct messages only; channel questions use @mentions (handled above).
        if event.get("channel_type") != "im" or event.get("bot_id") or event.get("subtype"):
            return
        say(text=build_reply(event.get("text", ""), policies, rules))

    return app


def main():
    bot_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    app_token = os.environ.get("SLACK_APP_TOKEN", "").strip()
    if not bot_token.startswith("xoxb-") or not app_token.startswith("xapp-"):
        sys.exit("Add SLACK_BOT_TOKEN (starts with xoxb-) and SLACK_APP_TOKEN (starts with xapp-) to the .env file. "
                 "See SLACK_SETUP.md.")
    if not config.api_key:
        sys.exit("Add GEMINI_API_KEY to the .env file first.")
    try:
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError:
        sys.exit("The Slack library is missing. Run:  pip install -r requirements.txt")

    policies = load_policies()
    rules = RulesIndex(policies)
    try:
        vector_index.build_index(policies)
    except Exception as err:  # noqa: BLE001
        print("Vector index unavailable, the bot will use fallbacks:", err)

    app = create_app(bot_token, policies, rules)

    print("Policy bot is running. Mention it in your Slack channel. Press Ctrl+C to stop.")
    SocketModeHandler(app, app_token).start()


if __name__ == "__main__":
    main()
