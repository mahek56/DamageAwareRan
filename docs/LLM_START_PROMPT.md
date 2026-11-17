# Paste this at the start of every new chat (fill the role number)
You are helping me, role R<N>, on the DamageAware research project.
First read, and only these: docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, roles/R<N>_*.md, and the last 40 lines of docs/STATUS.md.
Follow the rules in .agent/rules/damageaware.md. Work in my branch. Edit only files my role owns.
Plan first, show me the plan, then implement. Run pytest before saying done.
Finish by appending a handoff entry to docs/STATUS.md using the template, then commit.
Do one task per chat. If this chat gets long, tell me to start a new one.
My task now: <one sentence from my role file>
