# Building Trust in an AI Harness

## Philosophy

A harness is **not** trusted because it uses an LLM.

A harness is trusted because it makes AI **predictable, transparent, controllable, and recoverable**.

Think of it like aviation.

> People don't trust airplanes because the pilot is perfect.
>
> They trust airplanes because there are checklists, flight recorders, warning systems, autopilot, and safety procedures surrounding the pilot.

The LLM is the pilot.

The harness is everything that makes the pilot trustworthy.

---

# Trust Principles

## 1. Transparency

Never hide what the agent is doing.

Instead of

```text
Working...
```

Show

```text
Reading project structure...
Loading configuration...
Searching authentication code...
Analyzing error logs...
Planning changes...
```

The user should always know:

- What is happening
- Why it is happening
- What comes next

---

## 2. Explain Every Decision

Every action should include:

- Reason
- Evidence
- Expected Outcome

Example

```text
Action:
Update authentication middleware

Reason:
Expired JWT tokens are not validated.

Evidence:
- auth.ts line 83
- login.test.ts failure
- Stack trace

Expected Result:
Expired sessions will now be rejected.
```

Don't just say what changed.

Explain why.

---

## 3. Show the Plan Before Execution

Users should know the complete plan before actions begin.

Example

```text
Goal:
Fix login failure.

Plan:
✓ Inspect logs
✓ Identify root cause
✓ Update middleware
✓ Run tests
✓ Commit changes

Estimated Time:
45 seconds
```

A visible plan reduces uncertainty.

---

## 4. Confidence Scoring

Not every decision is equally reliable.

Example

```text
Rename variable
Confidence: 99%

Delete production data
Confidence: 12%

Recommendation:
Human approval required.
```

High confidence can be automated.

Low confidence should require approval.

---

## 5. Risk-Based Permissions

Every action should have a risk level.

| Risk | Examples | Approval |
|------|----------|----------|
| Low | Read files, search logs | No |
| Medium | Format code, update comments | Optional |
| High | Modify business logic | Required |
| Critical | Delete data, production deployment | Always Required |

The harness—not the LLM—decides the risk level.

---

## 6. Human Approval Gates

The harness should interrupt only when necessary.

Bad

```text
Approve reading package.json?
Approve searching files?
Approve opening README?
```

Good

```text
This command will delete 2,430 files.

Approval Required.
```

Frequent interruptions reduce trust.

Meaningful interruptions build trust.

---

## 7. Complete Audit Trail

Every action should be recorded.

Example

```text
10:14
Observed failing test

10:15
Read auth.ts

10:16
Hypothesis:
JWT expiration missing

10:17
Updated middleware

10:18
Executed tests

10:19
All tests passed
```

Users should be able to replay every decision.

---

## 8. Reversible Actions

Every change should be undoable.

Example

```text
Applied patch.

Undo available.

Rollback Command:
git restore auth.ts
```

Mistakes become far less frightening when they can be reversed instantly.

---

## 9. Evidence-Driven Conclusions

Every recommendation should include supporting evidence.

Instead of

```text
Upgrade React.
```

Show

```text
Recommendation:
Upgrade React

Evidence:
- Security advisory
- Deprecated APIs
- 14 failing tests
- Community migration guide

Confidence:
94%
```

Evidence builds trust.

Assertions alone do not.

---

## 10. Honest Uncertainty

Never pretend certainty.

Bad

```text
Issue fixed.
```

Better

```text
Most likely cause:
Database connection timeout

Confidence:
68%

Additional logs recommended before production deployment.
```

Admitting uncertainty increases credibility.

---

## 11. Learn User Preferences

The harness should observe recurring patterns.

Example

```text
Observed:

User always uses:

- bun
- rg
- bat
- uv

Future commands will follow these preferences.
```

The harness becomes personalized over time.

---

## 12. Progressive Autonomy

Trust should be earned gradually.

### Level 0 — Observe

Only watches.

No suggestions.

---

### Level 1 — Suggest

Suggests commands.

User copies manually.

---

### Level 2 — Assisted

Runs commands only after approval.

---

### Level 3 — Trusted

Automatically performs low-risk actions.

Requests approval for medium/high-risk tasks.

---

### Level 4 — Autonomous

Operates independently.

Reports results afterward.

Only possible after extensive successful history.

---

# Trust Metrics

Measure trust objectively.

| Metric | Description |
|---------|-------------|
| Approval Rate | Percentage of suggestions accepted |
| Undo Rate | Percentage of actions reverted |
| User Edits | How often users modify AI changes |
| Recovery Time | Time required to recover mistakes |
| Repeat Usage | Frequency of returning users |
| Intervention Rate | How often humans interrupt the agent |
| Success Rate | Tasks completed without rollback |
| Confidence Accuracy | Correlation between confidence and correctness |

Improving these metrics increases trust over time.

---

# Trust Architecture

```text
                    User
                      │
                      ▼
              Intent Planner
                      │
                      ▼
              Risk Classifier
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
   Safe Action                Risky Action
        │                           │
        ▼                           ▼
 Execute Automatically      Human Approval
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Evidence Collector
                      ▼
               Audit Journal
                      ▼
             Rollback Manager
                      ▼
             Preference Learner
                      ▼
              Knowledge Memory
```

Notice that the LLM is **not** responsible for trust.

The harness provides trust through:

- Risk analysis
- Permission checks
- Evidence collection
- Rollback support
- Logging
- Preference learning
- Memory
- Policy enforcement

---

# Trust Flow Example

```text
User:
Fix login issue.

↓

Planner:
Determine affected files.

↓

Retriever:
Read auth.ts
Read middleware.ts

↓

Reasoner:
JWT expiration missing.

Confidence:
92%

↓

Risk Engine:
Medium Risk

↓

Approval:
Required

↓

Execution:
Patch file

↓

Verification:
Run tests

↓

Evidence:
All tests passed.

↓

Journal:
Store reasoning and actions.

↓

Memory:
User prefers bun test.
```

---

# Core Components of a Trustworthy Harness

## Planner

Breaks the goal into executable steps.

---

## Retriever

Collects relevant context.

---

## Reasoner

Generates possible solutions.

---

## Risk Engine

Determines whether actions are safe.

---

## Policy Engine

Applies organizational and user rules.

Examples:

- Never delete production data.
- Never expose secrets.
- Never run destructive commands without approval.

---

## Approval Manager

Requests permission only when required.

---

## Executor

Runs approved actions.

---

## Validator

Confirms the outcome.

Examples:

- Tests pass
- Build succeeds
- Deployment healthy

---

## Evidence Collector

Stores proof supporting each action.

---

## Audit Journal

Maintains a complete timeline of actions.

---

## Rollback Manager

Allows instant recovery.

---

## Preference Learner

Learns user workflows and conventions.

Examples:

- Uses bun instead of npm
- Prefers ripgrep
- Formats with prettier
- Uses conventional commits

---

## Memory

Stores long-term project knowledge.

Examples:

- Repository structure
- Team coding conventions
- Frequently used commands
- Historical fixes

---

# Design Principles

A trustworthy harness should always be:

- Predictable
- Transparent
- Explainable
- Honest
- Recoverable
- Observable
- Evidence-based
- Policy-driven
- Human-controlled
- Continuously learning

---

# What Users Should Feel

At every step, the user should be able to answer these questions:

1. What is the agent doing?
2. Why is it doing it?
3. What evidence supports this?
4. What will happen next?
5. Can I stop it?
6. Can I undo it?
7. How confident is it?
8. What risks exist?
9. What changed?
10. Can I verify the outcome?

If the harness consistently answers these questions, users will naturally develop trust.

---

# The Trust Equation

```
Trust =
Predictability
+ Transparency
+ Explainability
+ Evidence
+ Human Control
+ Recoverability
+ Consistency
+ Competence
```

Without any one of these elements, trust decreases.

A highly capable AI that is opaque and irreversible will often be trusted less than a slightly less capable AI that is transparent, explainable, and easy to control.

---

# Final Principle

> **The LLM generates intelligence.**
>
> **The harness generates trust.**

A great harness is not judged by how smart its model is.

It is judged by how safe, understandable, reliable, and predictable the overall system feels to the people who use it.
