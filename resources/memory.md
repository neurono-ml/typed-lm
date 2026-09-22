# GreenLeaf Store — Refund and Support Policy (fictional demo memory)

This file is the loadable store memory evaluated together with every
request. All facts and rules below are fictional and exist only for
demos and tests.

## Facts

1. GreenLeaf Store is a fictional online shop selling home goods.
2. Defective or damaged items are eligible for a full refund within 30 days of delivery.
3. Changed-mind returns receive store credit only, within 14 days of delivery, and only when the item is unopened.
4. A double charge (the customer was billed twice for the same order) is always fully refundable, regardless of how much time has passed, because it is a billing error.
5. Gift cards and perishable goods are never refundable under any circumstance.
6. The billing department handles double charges and payment errors.
7. The logistics department handles damaged, lost, or late shipments.
8. The product support department handles defective-item troubleshooting, replacements, and setup help.
9. Urgency levels are Routine (general questions), Urgent (orders over 500 dollars or unresolved billing errors older than 30 days), and Emergency (medical devices that a customer depends on daily).
10. Refunds over 500 dollars require a manager approval note, but approval does not change eligibility.

## Rules

R1. When a case matches more than one fact, the most specific fact wins (for example, fact 4 overrides the 30-day window of fact 2 for double charges).
R2. Eligibility questions are answered "yes" only when a fact explicitly grants a refund for the case.
R3. Routing questions always select the single department whose responsibility matches the case.
