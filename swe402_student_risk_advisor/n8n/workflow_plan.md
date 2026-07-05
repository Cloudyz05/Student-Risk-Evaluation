# n8n Workflow Plan

Workflow name: Student Academic Risk Advisor

## Node 1 - Webhook Trigger

Purpose: receive a student profile from a form, Streamlit frontend, Postman, or manual test request.

Example URL after n8n creates it:

```text
https://YOUR-N8N-DOMAIN/webhook/student-risk-advisor
```

## Node 2 - HTTP Request to FastAPI

Purpose: send the student profile to the model API.

Method:

```text
POST
```

URL during local testing:

```text
http://host.docker.internal:8000/predict
```

URL after deployment:

```text
https://YOUR-RENDER-APP.onrender.com/predict
```

Body type: JSON.

## Node 3 - AI Agent

Purpose: reason over the model output and produce a practical support recommendation.

Suggested system message:

```text
You are an academic support assistant. Use the model prediction and student details to recommend a responsible support action. Keep the recommendation clear, specific, and suitable for a lecturer or academic advisor. Do not make medical, legal, or disciplinary claims.
```

Suggested user message:

```text
The student risk prediction result is:
{{ JSON.stringify($json) }}

Write:
1. risk summary
2. likely reason based on available features
3. recommended action
4. short message that can be sent to the student
```

## Node 4 - Output / Notification

Simple first version:

- Respond to Webhook with the AI Agent recommendation.

Better later version:

- Send email to advisor.
- Save the result into Google Sheets.
- Send Telegram/Discord notification.

## Required Assignment Mapping

- At least one trigger node: Webhook Trigger.
- Integration with deployed predictive model: HTTP Request to FastAPI `/predict`.
- At least one AI Agent node: AI Agent with Gemini/OpenAI chat model.
- At least one automated output or notification action: Respond to Webhook or email/Google Sheets.
- Successful complete workflow: test with sample student payload.
