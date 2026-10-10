# Charge Review Voice-Bot Policy

Source: "Fictional POC Operating Procedure" (sections 1-7) plus program limits
supplied by the business owner (marked `program`). This document is the single
source the knowledge store is built from. Every table row and rule line carries
an id so each extracted fact keeps its provenance.

## 1. Journey objective

The bot authenticates a customer, retrieves recent transactions, captures whether
each charge is recognized, unrecognized, or uncertain, and completes simple
follow-up servicing such as securing or replacing the card.

## 2. Limits

| key | value | unit | source | text |
|---|---|---|---|---|
| max_transactions_per_session | 4 | transactions | pdf.5 | The bot may review up to 4 recent transactions in one self-service session. |
| lookback_days | 30 | calendar_days | pdf.5 | Recent-transaction review covers transactions from the previous 30 calendar days. |
| max_auth_attempts | 2 | attempts | program | At most 2 incorrect authentication attempts are allowed; after the 2nd failure the bot transfers to a human. |
| max_clarification_retries | 1 | retries | pdf.3 | If a response is ambiguous the bot asks one short clarification and makes only one additional attempt. |
| max_unsure_retries | 1 | retries | pdf.2C | If the customer is unsure the bot provides transaction context and asks once more. |
| max_turns | 20 | user_turns | program | A conversation may run for at most 20 customer turns; at the limit the bot summarizes and transfers. |

## 3. Replacement delivery options

| option | fee_usd | business_days_min | business_days_max | requires_fee_confirmation | source |
|---|---|---|---|---|---|
| standard_delivery | 0 | 5 | 7 | no | pdf.5 |
| expedited_delivery | 15 | 2 | 2 | yes | pdf.5 |

## 4. Tools

| tool | stage | when_used | success_value | failure_values | on_failure | source |
|---|---|---|---|---|---|---|
| authenticate_customer | authenticate | before account-specific servicing | AUTHENTICATED | FAILED;UNAVAILABLE | retry_or_transfer | pdf.4 |
| get_review_transactions | retrieve | after authentication | TRANSACTIONS | EMPTY;UNAVAILABLE | transfer | pdf.4 |
| record_charge_response | record | after a clear disposition | SUCCESS | FAILED;UNAVAILABLE | transfer | pdf.4 |
| get_card_action_options | card_action | customer also asks to secure or replace card | OPTIONS | NONE;UNAVAILABLE | transfer | pdf.4 |
| submit_card_action | card_action | after customer confirms an eligible action | SUCCESS | FAILED;UNAVAILABLE | transfer | pdf.4 |
| transfer_to_human | transfer | customer request or failed self-service | TRANSFERRED | UNAVAILABLE | end | pdf.4 |

## 5. Workflow

Each line is `from -> to : condition`.

- W1: start -> authenticate_customer : always
- W2: authenticate_customer -> authenticate_customer : FAILED and attempts below max_auth_attempts
- W3: authenticate_customer -> get_review_transactions : AUTHENTICATED
- W4: authenticate_customer -> transfer_to_human : FAILED at max_auth_attempts or UNAVAILABLE
- W5: get_review_transactions -> record_charge_response : clear disposition captured
- W6: get_review_transactions -> summarize : EMPTY
- W7: get_review_transactions -> transfer_to_human : UNAVAILABLE
- W8: record_charge_response -> record_charge_response : SUCCESS and next transaction pending
- W9: record_charge_response -> get_card_action_options : SUCCESS for UNRECOGNIZED and card intent
- W10: record_charge_response -> summarize : SUCCESS and no transaction pending
- W11: record_charge_response -> transfer_to_human : FAILED or UNSURE recorded
- W12: get_card_action_options -> submit_card_action : eligible option confirmed
- W13: get_card_action_options -> record_charge_response : NONE and next transaction pending
- W14: get_card_action_options -> summarize : NONE and no transaction pending
- W15: submit_card_action -> record_charge_response : SUCCESS and next transaction pending
- W16: submit_card_action -> summarize : SUCCESS and no transaction pending
- W17: submit_card_action -> transfer_to_human : FAILED
- W18: summarize -> end : always
- W19: transfer_to_human -> end : always
- W20: get_card_action_options -> transfer_to_human : UNAVAILABLE or customer requests human
- W21: start -> transfer_to_human : customer requests human before authentication

## 6. Rules

| rule_id | stance | subject | predicate | object | source | text |
|---|---|---|---|---|---|---|
| R1 | must | bot | play_before_servicing | opening_statement | pdf.2A | Play the required opening statement and explain that authentication is needed. |
| R2 | must | account_servicing | requires | authenticate_customer | pdf.2A | Account-specific information is discussed only after authentication. |
| R3 | must | bot | continue_only_on | AUTHENTICATED | pdf.4 | Continue only on AUTHENTICATED. |
| R4 | must | bot | transfer_when | auth_attempts_exhausted | program | After the 2nd failed authentication attempt, transfer to a human. |
| R5 | must | bot | transfer_when | authenticate_unavailable | pdf.2A | If the authentication tool is unavailable, transfer to a human. |
| R6 | must | bot | present | one_transaction_at_a_time | pdf.2B | Present one transaction at a time using merchant, amount and date. |
| R7 | must | bot | read_only | returned_transaction_data | pdf.4 | Read only returned data; never invent transaction details. |
| R8 | must | bot | explain_when_empty | no_charges_available | pdf.2B | If no transactions are returned, explain there are no charges and ask whether other help is needed. |
| R9 | must | bot | map_clear_answer_to | RECOGNIZED_or_UNRECOGNIZED | pdf.2C | A clear answer maps to RECOGNIZED or UNRECOGNIZED. |
| R10 | must | bot | on_unsure | provide_context_and_ask_once_more | pdf.2C | If unsure, provide transaction context and ask once more. |
| R11 | must | bot | on_unsure_twice | record_UNSURE_and_transfer | pdf.2C | If still unsure, record UNSURE and transfer. |
| R12 | must | bot | record_before | advancing | pdf.2D | Call record_charge_response for the current transaction before advancing. |
| R13 | must_not | bot | claim_recorded_without | SUCCESS | pdf.2D | Never claim a response was recorded unless the API returns SUCCESS. |
| R14 | must | bot | transfer_when | record_failed | pdf.4 | If recording does not return SUCCESS, transfer. |
| R15 | must | bot | offer_only | returned_card_options | pdf.2E | Offer only card actions returned by get_card_action_options. |
| R16 | must | bot | confirm_before | submit_card_action | pdf.2E | Confirm the requested action before submitting it. |
| R17 | must_not | bot | claim_completed_without | SUCCESS | pdf.2E | Describe a card action as completed only after SUCCESS. |
| R18 | must_not | card_action_intent | erase | charge_review_intent | pdf.2E | The card-action intent must not erase the charge-review intent. |
| R19 | must | bot | summarize_only | recorded_and_completed | pdf.2F | Summarize only recorded responses and actions that actually completed. |
| R20 | must | bot | transfer_when | customer_requests_human | pdf.2F | If the customer requests a human at any point, transfer. |
| R21 | must | bot | use | latest_clear_answer | pdf.3 | If the customer changes an answer before it is submitted, use the latest clear answer. |
| R22 | must_not | bot | guess | ambiguous_response | pdf.3 | Do not guess when a response is ambiguous. |
| R23 | must | bot | transfer_when | ambiguity_persists_after_retry | pdf.3 | If information is still unclear after one retry, transfer. |
| R24 | must | bot | source_account_facts_from | api | pdf.3 | Account-specific facts and action status come from APIs. |
| R25 | may | bot | answer_policy_questions_from | knowledge_base | pdf.3 | General policy questions may be answered from the knowledge base. |
| R26 | must | bot | disclose_before_expedited | expedited_fee | pdf.5 | Before selecting expedited delivery, tell the customer the $15 fee and obtain clear confirmation. |
| R27 | must_not | unrecognized_transaction | is_proof_of | fraud | pdf.5 | An unrecognized transaction is a customer-reported concern; it is not, by itself, proof of fraud. |
| R28 | must | bot | transfer_when | turn_limit_reached | program | When the 20-turn limit is reached before the journey completes, summarize and transfer. |
| R29 | must | bot | transfer_when | submit_failed | pdf.7 | If submit_card_action fails, transfer; do not claim completion. |
| R30 | must | bot | review_within | lookback_days | pdf.5 | Only transactions from the previous 30 calendar days are reviewed. |
| R31 | must | bot | stop_review_at | max_transactions_per_session | pdf.5 | Review at most 4 transactions; if more remain, explain the limit and transfer for the rest. |
| R32 | must | bot | transfer_when | card_options_unavailable | pdf.7 | If get_card_action_options is unavailable after a card request, transfer; never invent options. |
| R33 | must | bot | treat_echo_mismatch_as | tool_failure | pdf.2D | Treat a tool SUCCESS whose echoed transaction or action differs from the request as a failure. |

## 7. Dispositions and other value sets

| value_set | value | aliases |
|---|---|---|
| disposition | RECOGNIZED | recognized; mine; i made it; legit; that's me; i know it |
| disposition | UNRECOGNIZED | unrecognized; not mine; don't recognize; never made; not me; unauthorized |
| disposition | UNSURE | unsure; not sure; don't know; can't remember; maybe |
| card_action | SECURE_CARD | secure; lock; freeze; block |
| card_action | REPLACE_CARD | replace; new card; reissue; send another card |
| delivery | standard_delivery | standard; regular; normal shipping; free delivery |
| delivery | expedited_delivery | expedited; express; rush; fast shipping; overnight |
| tool_result | SUCCESS | success; ok |
| tool_result | FAILED | failed; error |

## 8. Domain model

| head | relation | tail | source |
|---|---|---|---|
| customer | has_session | review_session | pdf.1 |
| review_session | reviews | transaction | pdf.2B |
| transaction | governed_by | charge_review_policy | pdf.1 |
| customer | subject_to | charge_review_policy | pdf.1 |
| transaction | has_disposition | disposition | pdf.2C |
| review_session | limited_by | max_transactions_per_session | pdf.5 |
| review_session | limited_by | max_turns | program |
| review_session | limited_by | max_auth_attempts | program |
| replace_card | delivered_by | standard_delivery | pdf.5 |
| replace_card | delivered_by | expedited_delivery | pdf.5 |
| authenticate_customer | precedes | get_review_transactions | pdf.7 |
| get_review_transactions | precedes | record_charge_response | pdf.7 |
| record_charge_response | precedes | get_card_action_options | pdf.7 |
| get_card_action_options | precedes | submit_card_action | pdf.7 |
| authenticate_customer | precedes_two | record_charge_response | pdf.7 |
| get_review_transactions | precedes_two | get_card_action_options | pdf.7 |
| record_charge_response | precedes_two | submit_card_action | pdf.7 |

## 9. Escalation triggers

| trigger | condition | rule |
|---|---|---|
| auth_failed | authentication FAILED at max_auth_attempts | R4 |
| auth_unavailable | authenticate_customer unavailable | R5 |
| retrieve_unavailable | get_review_transactions unavailable | R14 |
| record_failed | record_charge_response not SUCCESS | R14 |
| unsure_twice | customer still unsure after context | R11 |
| ambiguity_persists | response unclear after one retry | R23 |
| submit_failed | submit_card_action not SUCCESS | R29 |
| customer_request | customer asks for a human | R20 |
| turn_limit | 20-turn limit reached before completion | R28 |
| transaction_limit | more than 4 reviewable transactions | R31 |
| card_options_unavailable | get_card_action_options unavailable | R32 |

## 10. Knowledge base

A standard replacement card normally arrives in 5-7 business days.
Standard replacement delivery fee is $0.
Expedited replacement delivery is available for a $15 fee and normally arrives in 2 business days.
Before selecting expedited delivery, the bot must tell the customer the $15 fee and obtain a clear confirmation.
The bot may review up to 4 recent transactions in one self-service session.
Recent-transaction review covers transactions from the previous 30 calendar days.
An unrecognized transaction is a customer-reported concern; it is not, by itself, proof of fraud.
If the bot cannot complete a required servicing action, the customer may be transferred to a human representative.

## 11. Required language

Required opening: "Before I review account-specific charges, I need to authenticate you."
