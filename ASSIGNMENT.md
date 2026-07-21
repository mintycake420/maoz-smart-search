# MAOZ — Take-Home Assignment (Source of Truth)

> This file is the verbatim assignment brief received from MAOZ. It is the
> authoritative spec for this repository. Every deliverable must trace back to a
> requirement here.

**Received:** 2026-07-20 · **Deadline:** 48 hours from receipt
**Submission format:** PDF document or link to a shared document

---

## Cover Note

Dear Candidate,

As part of the MAOZ selection process, we ask you to complete a take-home
assignment that will allow us to become familiar with your way of thinking, your
work process, and the way you approach solving a real organizational challenge.

As part of the assignment, you will be asked to design and build an initial Proof
of Concept (POC) for a smart search system based on the information described
below.

- There is no single "correct" answer. We are interested in your thought process,
  decision-making approach, and the considerations that led to your proposed
  solution.
- You may use AI tools while completing the assignment. At the end of the
  assignment, please specify which tools you used and how they assisted you.
- Completion time: Up to 48 hours from the time you receive the assignment.
- Should any questions arise or clarification be required during the assignment,
  you may contact us by email, and we will be happy to assist.
- Submission format: A PDF document or a link to a shared document.

## Background and Context

MAOZ works to strengthen the social and economic resilience of the State of
Israel by building trust-based networks of leaders from all sectors of Israeli
society.

Currently, information about network members is managed in Salesforce. One of the
main challenges is enabling network members and MAOZ staff to quickly identify
relevant people for collaboration, consultation, and relationship-building — not
only according to their role or organization, but also based on areas of
activity, experience, interests, shared challenges, and values.

## Part A – Solution Definition

### Understanding the Need

To define the solution, what five questions would you ask the network managers
before beginning development?

For each question, briefly explain how the answer would affect the solution
design or the user experience.

### Solution and Architecture Definition

Define the proposed solution and present a data flow diagram, covering the
process from the information stored in Salesforce to the presentation of search
results to the user.

As part of your solution definition, briefly explain the considerations behind
the choices you made. Address, among other things:

- The development and working environment, models, and AI tools you would choose
  to use.
- The use of Retrieval-Augmented Generation (RAG), where relevant.
- The database or storage mechanism.
- The mechanism used to generate embeddings.
- The system components and the relationships between them.

### Information Security

Information security must be an integral part of the system's design.

Explain how you would incorporate information-security and authorization
considerations into the proposed solution, and how you would ensure that
sensitive information is not exposed to unauthorized individuals, systems, or
models.

## Part B – Proof of Concept (POC) Development

Based on the solution defined in Part A, implement an initial Proof of Concept
for a semantic search mechanism, in accordance with your proposed architecture.

The system should be able to identify connections between semantically similar
terms — for example, between "informal education" and "youth movements" — even
when the exact words do not appear in the person's profile.

As part of the submission:

- Implement the semantic search component.
- Attach the source code.
- Where necessary, include brief explanations that will assist in understanding
  the solution.

## Part C – Presentation of the Assignment

As part of the next stage of the selection process, candidates who advance will
be asked to present their assignment during a short Zoom meeting.

During the meeting, you will be asked to:

- Present the solution you developed.
- Explain the professional and technical choices you made throughout the process.
- Explain the considerations that led to the proposed solution.
- Answer professional questions regarding your thought process, solution
  definition, and implementation.

---

## Derived Deliverables Checklist

- [x] **A.1** Five questions for network managers + design impact of each answer
- [x] **A.2** Proposed solution + data flow diagram (Salesforce → results in UI)
- [x] **A.2a** Rationale: dev environment, models, AI tools
- [x] **A.2b** Rationale: RAG usage (where relevant)
- [x] **A.2c** Rationale: database / storage mechanism
- [x] **A.2d** Rationale: embedding generation mechanism
- [x] **A.2e** Rationale: system components and their relationships
- [x] **A.3** Information security & authorization design
- [x] **B.1** Working semantic search POC (demonstrates "informal education" ↔
      "youth movements" style matching)
- [x] **B.2** Source code attached
- [x] **B.3** Brief explanations aiding comprehension
- [x] **X** Statement of which AI tools were used and how
- [ ] **C** Presentation-ready narrative for the Zoom session
