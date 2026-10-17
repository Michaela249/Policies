# Slack bot setup

The bot runs on your own computer (the assignment allows this) and connects to Slack over Socket Mode, so you don't need a public URL. It answers with the vector-index approach and names the policy it used.

## 1. Create your own workspace and test channel

1. Go to https://slack.com/get-started and choose **Create a new workspace** (free plan is fine).
2. Name the workspace so it identifies you, for example `Anna Novak Policy Bot Test`.
3. Create a channel with your name in it, for example `#anna-novak-policy-test`.

## 2. Create the Slack app (about 2 minutes)

1. Go to https://api.slack.com/apps and click **Create New App**, then **From a manifest**.
2. Pick your new workspace, switch to the **JSON** tab, and paste the contents of `slack-app-manifest.json` from this folder. Click **Next**, then **Create**.
3. **App-level token:** on the **Basic Information** page, scroll to **App-Level Tokens**, click **Generate Token and Scopes**, name it `socket`, add the scope `connections:write`, and click **Generate**. Copy the token (it starts with `xapp-`).
4. **Bot token:** open **Install App** in the left menu, click **Install to Workspace**, then **Allow**. Copy the **Bot User OAuth Token** (it starts with `xoxb-`).

## 3. Put both tokens in `.env`

Add two lines to the same `.env` file that holds your Gemini key:

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

## 4. Run the bot

```
pip install -r requirements.txt
python slack_bot.py
```

(On Mac or Linux use `pip3` and `python3`.) This installs one package, `slack_bolt`, which only the bot needs. The first start loads the vector indexes (or builds them, about a minute, if you haven't run the website yet). When it prints `Policy bot is running`, leave that window open.

## 5. Use it

1. In your test channel type `/invite @Policy Bot`.
2. Ask a question by mentioning it, for example: `@Policy Bot How many vacation days do I get per year?`
3. It replies in a thread with the answer, the relevant policy, and the approach, time and tokens used. You can also send it a direct message without mentioning it.

## 6. Invite raz@sdu.dk and take the screenshot

1. Invite `raz@sdu.dk` to your workspace by email (click the workspace name, then **Invite people to ...**). If Slack lets you pick channels, choose your test channel; otherwise add them to the channel once they have joined.
2. Take a screenshot of the channel that shows your question mentioning the bot and the bot's answer with the policy name. That is the screenshot for the assignment.

## Troubleshooting

- **The bot doesn't answer:** make sure `python slack_bot.py` is still running, and that you invited the bot to the channel (`/invite @Policy Bot`).
- **"invalid_auth" or a token error:** the bot token must start with `xoxb-` and the app token with `xapp-`. Don't mix them up, and don't put quotes or spaces around them.
- **Changed the manifest later?** On the app's page, reinstall it (**Install App**, **Reinstall to Workspace**).
